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
from typing import Optional, Tuple
import numpy as np
from .assemble import assemble_global, assemble_mass_global
from .boundary import apply_neumann_bc, apply_convection_bc, apply_dirichlet_bc, _map_dof
from .solver import solve_dense
from .post import extract_nodal_field
from .io import BoundaryConditions, ConvectionBC


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
    """Perform simple Newmark-beta time integration for dynamic problems.

    `dyn` is expected to contain keys from `DynamicParameters` dataclass such
    as c0,cx,cy,ntime,nstp,dt,alfa,gama,initial_u,initial_v,initial_a.

    Returns the final displacement vector and nodal field; history can be
    obtained by calling this repeatedly or modifying the function to store it.
    """
    # assemble stiffness and mass
    K, F = assemble_global(nod, glxy, npe, ndf, element_func, itype, material, coeffs)  # type: ignore
    if mass_func is None:
        raise ValueError("mass_func must be provided for dynamic analysis")
    M = assemble_mass_global(nod, glxy, npe, ndf, mass_func, itype, coeffs, dyn)  # type: ignore
    # mass_func cannot be element_func; user should pass separate function? use element_mass...
    # to keep simple we assume element_func has an attribute 'mass' or external

    # apply BCs to K,F,M
    if bc is not None:
        apply_neumann_bc(F, bc, ndf)
    if convection is not None:
        apply_convection_bc(K, F, convection, nod=nod, glxy=glxy, ndf=ndf)
    if bc is not None:
        apply_dirichlet_bc(K, F, bc, ndf)
        # also modify mass: zero rows/cols of fixed DOFs
        for i in range(bc.ispv.shape[0]):
            g = _map_dof(bc.ispv[i,0], bc.ispv[i,1], ndf)
            M[g,:] = 0.0
            M[:,g] = 0.0
            M[g,g] = 1.0

    # initial conditions
    u = dyn.get('initial_u', np.zeros(K.shape[0])) if dyn else np.zeros(K.shape[0])
    v = dyn.get('initial_v', np.zeros_like(u)) if dyn else np.zeros_like(u)
    a = dyn.get('initial_a', None) if dyn else None
    if a is None:
        a = solve_dense(M, F - K @ u)
    dt = dyn.get('dt', 1.0) if dyn else 1.0
    beta = dyn.get('alfa', 0.25) if dyn else 0.25
    gamma = dyn.get('gama', 0.5) if dyn else 0.5
    ntime = dyn.get('ntime', 1) if dyn else 1
    # coefficients
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = dt * (gamma / (2 * beta) - 1)
    Keff = K + a0 * M
    u_next = u.copy()
    v_next = v.copy()
    a_next = a.copy()
    for step in range(ntime):
        rhs = F + M @ (a0 * u + a2 * v + a3 * a)
        u_next = solve_dense(Keff, rhs)
        v_next = a1 * (u_next - u) - a4 * v - a5 * a
        a_next = a0 * (u_next - u) - a2 * v - a3 * a
        u, v, a = u_next, v_next, a_next
    field = extract_nodal_field(u, ndf)
    return u, field
