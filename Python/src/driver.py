"""
Simple problem driver that ties modules together for a static analysis run.

This is a minimal convenience wrapper used by tests and examples. It performs:
- assembly via `assemble.assemble_global`
- application of Neumann and convection BCs
- application of Dirichlet BCs
- solving with dense solver
- basic postprocessing

The driver intentionally keeps a clear sequence to mirror the Fortran program flow.
"""
from typing import Optional, Tuple, Any
import numpy as np
from pathlib import Path
from .assemble import assemble_global, assemble_mass_global
from .boundary import apply_neumann_bc, apply_convection_bc, apply_dirichlet_bc, _map_dof
from .solver import solve_dense
from .post import extract_nodal_field, write_fortran_style_output_txt
from .io import BoundaryConditions, ConvectionBC, read_inp
from .mesh import generate_mesh_from_config
from .material import compute_cmat_from_config
from . import elements


def solve_static(nod: np.ndarray,
                 glxy: np.ndarray,
                 npe: int,
                 ndf: int,
                 itype: int = 0,
                 element_func=None,
                 material: Optional[np.ndarray] = None,
                 coeffs: Optional[dict] = None,
                 bc: Optional[BoundaryConditions] = None,
                 convection: Optional[ConvectionBC] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Run a static analysis: assemble, apply BCs, solve, postprocess.

    Returns (u, field) where `u` is the global solution vector and `field` is
    the nodal array with shape (nnm, ndf).
    """
    # Assemble global K and F
    K, F = assemble_global(nod, glxy, npe, ndf, element_func, itype, material, coeffs)# type: ignore

    # Apply natural BCs (Neumann)
    if bc is not None:
        apply_neumann_bc(F, bc, ndf)

    # Apply convection BCs
    if convection is not None:
        apply_convection_bc(K, F, convection, nod=nod, glxy=glxy, ndf=ndf)

    # Apply Dirichlet BCs (essential)
    if bc is not None:
        apply_dirichlet_bc(K, F, bc, ndf)

    # Solve
    u = solve_dense(K, F)

    # Postprocess: reshape to nodal field
    field = extract_nodal_field(u, ndf)
    return u, field


def solve_dynamic(nod: np.ndarray,
                  glxy: np.ndarray,
                  npe: int,
                  ndf: int,
                  itype: int = 0,
                  element_func=None,
                  mass_func=None,
                  material: Optional[np.ndarray] = None,
                  coeffs: Optional[dict] = None,
                  bc: Optional[BoundaryConditions] = None,
                  convection: Optional[ConvectionBC] = None,
                  dyn: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Perform dynamic integration following the Fortran TEMPORAL() workflow."""
    K, F = assemble_global(nod, glxy, npe, ndf, element_func, itype, material, coeffs)  # type: ignore
    if mass_func is None:
        raise ValueError("mass_func must be provided for dynamic analysis")
    M = assemble_mass_global(nod, glxy, npe, ndf, mass_func, itype, coeffs, dyn)  # type: ignore

    C = None
    if dyn is not None and 'damping' in dyn:
        damp = dyn['damping']
        alpha = damp.get('alpha', 0.0)
        beta = damp.get('beta', 0.0)
        if alpha != 0.0 or beta != 0.0:
            C = alpha * M + beta * K

    if bc is not None:
        apply_neumann_bc(F, bc, ndf)
    if convection is not None:
        apply_convection_bc(K, F, convection, nod=nod, glxy=glxy, ndf=ndf)
    if bc is not None:
        apply_dirichlet_bc(K, F, bc, ndf)
        for i in range(bc.ispv.shape[0]):
            g = _map_dof(bc.ispv[i, 0], bc.ispv[i, 1], ndf)
            M[g, :] = 0.0
            M[:, g] = 0.0
            M[g, g] = 1.0
            if C is not None:
                C[g, :] = 0.0
                C[:, g] = 0.0
                C[g, g] = 1.0

    u = dyn.get('initial_u', np.zeros(K.shape[0])) if dyn else np.zeros(K.shape[0])
    v = dyn.get('initial_v', np.zeros_like(u)) if dyn else np.zeros_like(u)
    a = dyn.get('initial_a', None) if dyn else None
    intial = dyn.get('intial', 0) if dyn else 0
    if a is None:
        if intial == 0:
            a = np.zeros_like(u)
        else:
            damping_term = C @ v if C is not None else 0.0
            a = solve_dense(M, F - K @ u - damping_term)
    dt = dyn.get('dt', 1.0) if dyn else 1.0
    alfa = dyn.get('alfa', 0.25) if dyn else 0.25
    gama = dyn.get('gama', 0.5) if dyn else 0.5
    item_mode = dyn.get('item', 2) if dyn else 2
    ntime = dyn.get('ntime', 1) if dyn else 1
    nstp = dyn.get('nstp', ntime + 1) if dyn else ntime + 1

    A1 = alfa * dt
    A2 = (1.0 - alfa) * dt
    if item_mode == 1:
        # Fortran TEMPORAL(), ITEM=1 (alfa-family for parabolic equations).
        Khat = M + A1 * K
        for step in range(1, ntime + 1):
            F_step = F if step < nstp else np.zeros_like(F)
            rhs = (A1 + A2) * F_step + (M - A2 * K) @ u
            if C is not None:
                rhs = rhs + C @ u
            u = solve_dense(Khat, rhs)
    else:
        # Fortran TEMPORAL(), ITEM=2 (Newmark family for hyperbolic equations).
        DT2 = dt * dt
        A3 = 2.0 / (gama * DT2)
        A4 = A3 * dt
        A5 = 1.0 / gama - 1.0

        Khat = K + A3 * M
        if C is not None:
            Khat = Khat + A4 * C

        for step in range(1, ntime + 1):
            F_step = F if step < nstp else np.zeros_like(F)
            rhs = F_step + M @ (A3 * u + A4 * v + A5 * a)
            if C is not None:
                rhs = rhs + C @ v
            u_next = solve_dense(Khat, rhs)
            a_next = A3 * (u_next - u) - A4 * v - A5 * a
            v_next = v + A1 * a_next + A2 * a
            u, v, a = u_next, v_next, a_next

    field = extract_nodal_field(u, ndf)
    return u, field


def solve_from_inp(filepath: str, output_txt_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Parse an INP file, solve, and write a Fortran-style text output file."""
    cfg = read_inp(filepath)
    mesh = generate_mesh_from_config(cfg)

    coeffs = {}
    if cfg.source_loads is not None:
        coeffs = {'F0': cfg.source_loads.f0, 'FX': cfg.source_loads.fx, 'FY': cfg.source_loads.fy}

    material = None
    if cfg.material is not None and cfg.problem_type.itype >= 2:
        cm = compute_cmat_from_config(
            cfg.problem_type.itype,
            cfg.material.lnstrs,
            cfg.material.e1,
            cfg.material.e2,
            cfg.material.anu12,
            cfg.material.g12,
            cfg.material.thkns,
            cfg.material.g13,
            cfg.material.g23,
        )
        if cfg.problem_type.itype in (3, 4):
            material = {'CMAT': cm.cmat, 'C44': cm.c44 or 0.0, 'C55': cm.c55 or 0.0}
        else:
            material = cm.cmat

    if cfg.problem_type.item != 0:
        dyn = {
            'item': cfg.problem_type.item,
            'neign': cfg.problem_type.neign,
            'C0': cfg.dynamic.c0 if cfg.dynamic else 1.0,
            'CX': cfg.dynamic.cx if cfg.dynamic else 0.0,
            'CY': cfg.dynamic.cy if cfg.dynamic else 0.0,
            'ntime': cfg.dynamic.ntime if cfg.dynamic else 1,
            'nstp': cfg.dynamic.nstp if cfg.dynamic else 2,
            'intial': cfg.dynamic.intial if cfg.dynamic else 0,
            'dt': cfg.dynamic.dt if cfg.dynamic else 1.0,
            'alfa': cfg.dynamic.alfa if cfg.dynamic else 0.25,
            'gama': cfg.dynamic.gama if cfg.dynamic else 0.5,
        }
        if cfg.dynamic and cfg.dynamic.initial_u is not None:
            dyn['initial_u'] = cfg.dynamic.initial_u
        if cfg.dynamic and cfg.dynamic.initial_v is not None:
            dyn['initial_v'] = cfg.dynamic.initial_v
        if cfg.dynamic and cfg.dynamic.initial_a is not None:
            dyn['initial_a'] = cfg.dynamic.initial_a

        mass_func = elements.triangle_element_mass if cfg.element_mesh.npe == 3 else elements.quad_element_mass
        u, field = solve_dynamic(
            mesh.nod,
            mesh.glxy,
            cfg.element_mesh.npe,
            cfg.ndf,
            cfg.problem_type.itype,
            element_func=elements.element_matrices,
            mass_func=mass_func,
            material=material,
            coeffs=coeffs,
            bc=cfg.boundary_conditions,
            convection=cfg.convection,
            dyn=dyn,
        )
    else:
        u, field = solve_static(
            mesh.nod,
            mesh.glxy,
            cfg.element_mesh.npe,
            cfg.ndf,
            cfg.problem_type.itype,
            element_func=elements.element_matrices,
            material=material,
            coeffs=coeffs,
            bc=cfg.boundary_conditions,
            convection=cfg.convection,
        )

    if output_txt_path is None:
        inp = Path(filepath)
        output_txt_path = str(inp.with_name(f"{inp.stem}_python_output.txt"))
    final_time = None
    final_step = None
    if cfg.problem_type.item != 0 and cfg.dynamic is not None:
        final_step = cfg.dynamic.ntime
        final_time = cfg.dynamic.ntime * cfg.dynamic.dt
    write_fortran_style_output_txt(output_txt_path, cfg, mesh.glxy, field, final_time, final_step)

    return u, field, cfg

