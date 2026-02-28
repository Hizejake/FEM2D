"""
Postprocessing helpers for FEM2D.

Provides utilities to convert solution vectors into nodal fields, compute
simple derived quantities, and write Fortran-style text output.
"""
from typing import Any, Optional
import numpy as np


def _append_input_echo(lines: list[str], cfg: Any) -> None:
    """Append a concise, labeled echo of parsed input data."""
    itype_map = {
        0: "Heat transfer / Poisson",
        1: "Viscous incompressible flow",
        2: "Plane elasticity",
        3: "Plate bending (FSDT)",
        4: "Plate bending (Classical)",
        5: "Plate bending (CST conforming)",
    }
    item_map = {
        0: "Steady-state",
        1: "Transient (single variable)",
        2: "Transient (with velocity/acceleration)",
    }
    mesh_map = {
        0: "User-provided mesh",
        1: "Rectangular auto-generated mesh",
        2: "General auto-generated mesh (MESH2DG)",
    }

    lines.append("INPUT ECHO DATA")
    lines.append("------------------------------------------------------------")
    lines.append(f"TITLE : {cfg.title}  (Problem title)")
    lines.append("")

    pt = cfg.problem_type
    lines.append("PROBLEM TYPE FLAGS")
    lines.append(f"ITYPE : {pt.itype}  ({itype_map.get(pt.itype, 'Unknown')})")
    lines.append(f"IGRAD : {pt.igrad}  (0=no gradient/stress output, 1=compute)")
    lines.append(f"ITEM  : {pt.item}  ({item_map.get(pt.item, 'Unknown')})")
    lines.append(f"NEIGN : {pt.neign}  (0=non-eigen, non-zero=eigen analysis)")
    if pt.neign != 0:
        lines.append(f"NVALU : {pt.nvalu}  (Number of eigenvalues requested)")
        lines.append(f"NVCTR : {pt.nvctr}  (Eigenvector print flag)")
    lines.append("")

    em = cfg.element_mesh
    lines.append("ELEMENT / MESH CONTROL")
    lines.append(f"IELTYP : {em.ieltyp}  (0=tri, >0=quad)")
    lines.append(f"NPE    : {em.npe}  (Nodes per element)")
    lines.append(f"MESH   : {em.mesh}  ({mesh_map.get(em.mesh, 'Unknown')})")
    lines.append(f"NPRNT  : {em.nprnt}  (Print control)")
    if em.nem is not None:
        lines.append(f"NEM    : {em.nem}  (Number of elements)")
    if em.nnm is not None:
        lines.append(f"NNM    : {em.nnm}  (Number of nodes)")
    lines.append(f"NDF    : {cfg.ndf}  (DOF per node)")
    lines.append(f"NEQ    : {cfg.neq}  (Total equations)")
    lines.append("")

    if cfg.mesh_data is not None:
        rm = cfg.mesh_data
        lines.append("RECTANGULAR MESH INPUT")
        lines.append(f"NX, NY : {rm.nx}, {rm.ny}  (Subdivisions in x and y)")
        lines.append(f"X0, Y0 : {rm.x0}, {rm.y0}  (Mesh origin)")
        lines.append(f"DX     : {np.array2string(rm.dx, precision=6)}  (x segment lengths)")
        lines.append(f"DY     : {np.array2string(rm.dy, precision=6)}  (y segment lengths)")
        lines.append("")

    bc = cfg.boundary_conditions
    if bc is not None:
        lines.append("BOUNDARY CONDITIONS")
        nspv = int(bc.ispv.shape[0]) if bc.ispv is not None else 0
        nssv = int(bc.issv.shape[0]) if bc.issv is not None else 0
        lines.append(f"NSPV   : {nspv}  (Specified primary DOFs / Dirichlet)")
        lines.append(f"NSSV   : {nssv}  (Specified secondary DOFs / Neumann)")
        if nspv > 0:
            lines.append("ISPV   : [node, local_dof] pairs for primary constraints")
            lines.append(f"         {bc.ispv[:min(12, nspv)].tolist()}"
                         + (" ... (truncated)" if nspv > 12 else ""))
            if bc.vspv is not None:
                lines.append("VSPV   : Prescribed values for ISPV entries")
                lines.append(f"         {bc.vspv[:min(12, nspv)].tolist()}"
                             + (" ... (truncated)" if nspv > 12 else ""))
        if nssv > 0 and bc.vssv is not None:
            lines.append("VSSV   : Specified secondary values")
            lines.append(f"         {bc.vssv[:min(12, nssv)].tolist()}"
                         + (" ... (truncated)" if nssv > 12 else ""))
        lines.append("")

    if cfg.material is not None:
        m = cfg.material
        lines.append("MATERIAL PROPERTIES")
        lines.append("E1, E2     : "
                     f"{m.e1}, {m.e2}  (Young's moduli)")
        lines.append(f"ANU12      : {m.anu12}  (Poisson ratio)")
        lines.append(f"G12        : {m.g12}  (In-plane shear modulus)")
        if m.g13 is not None:
            lines.append(f"G13        : {m.g13}  (Transverse shear modulus)")
        if m.g23 is not None:
            lines.append(f"G23        : {m.g23}  (Transverse shear modulus)")
        lines.append(f"THKNS      : {m.thkns}  (Thickness)")
        if m.lnstrs is not None:
            lines.append(f"LNSTRS     : {m.lnstrs}  (0=plane strain, 1=plane stress)")
        lines.append("")

    if cfg.source_loads is not None:
        s = cfg.source_loads
        lines.append("SOURCE / LOAD COEFFICIENTS")
        lines.append(f"F0, FX, FY : {s.f0}, {s.fx}, {s.fy}  (f(x,y)=F0+FX*x+FY*y)")
        lines.append("")

    if cfg.dynamic is not None:
        d = cfg.dynamic
        lines.append("DYNAMIC PARAMETERS")
        lines.append(f"C0, CX, CY : {d.c0}, {d.cx}, {d.cy}  (Inertia/dynamic coefficients)")
        lines.append(f"NTIME      : {d.ntime}  (Number of time steps)")
        lines.append(f"NSTP       : {d.nstp}  (Load removal step)")
        lines.append(f"INTVL      : {d.intvl}  (Output interval)")
        lines.append(f"INTIAL     : {d.intial}  (Initial-condition flag)")
        lines.append(f"DT         : {d.dt}  (Time increment)")
        lines.append(f"ALFA, GAMA : {d.alfa}, {d.gama}  (Time integration parameters)")
        lines.append(f"EPSLN      : {d.epsln}  (Convergence tolerance)")
        lines.append("")


def extract_nodal_field(u: np.ndarray, ndf: int) -> np.ndarray:
    """Convert global DOF vector `u` into nodal field array of shape (nnm, ndf).

    For scalar problems (`ndf==1`) returns a (nnm,1) array. For vector problems
    returns (nnm, ndf) with ordering consistent with global DOF mapping.
    """
    u = np.asarray(u, dtype=float)
    if ndf == 1:
        return u.reshape((-1, 1))
    if u.size % ndf != 0:
        raise ValueError("Length of u is not divisible by ndf")
    nnm = u.size // ndf
    return u.reshape((nnm, ndf))


def max_abs_field(u: np.ndarray, ndf: int) -> float:
    """Return the maximum absolute value across all DOFs."""
    field = extract_nodal_field(u, ndf)
    return float(np.max(np.abs(field)))


def write_fortran_style_output_txt(
    output_path: str,
    cfg: Any,
    glxy: np.ndarray,
    field: np.ndarray,
    time_value: Optional[float] = None,
    time_step: Optional[int] = None,
) -> None:
    """Write a Fortran-like text output file for easier side-by-side comparison."""
    glxy = np.asarray(glxy, dtype=float)
    field = np.asarray(field, dtype=float)
    ndf = int(field.shape[1])

    if ndf == 1:
        cols = "    Node    x-coord.      y-coord.      Value of u"
    elif ndf == 2:
        cols = "    Node    x-coord.      y-coord.     Value of u     Value of v"
    else:
        cols = "    Node    x-coord.      y-coord.     deflec. w    x-rotation    y-rotation"

    line = "  _____________________________________________________________________________"
    lines = []
    lines.append(str(cfg.title))
    lines.append(line)
    lines.append("")
    _append_input_echo(lines, cfg)
    if time_value is not None and time_step is not None:
        lines.append(f"     *TIME* = {time_value:0.5E}     Time Step Number = {time_step:2d}")
        lines.append("")
        lines.append("     S O L U T I O N :")
        lines.append("")
    lines.append(line)
    lines.append("")
    lines.append(cols)
    lines.append(line)
    lines.append("")

    for i in range(glxy.shape[0]):
        node = i + 1
        x = glxy[i, 0]
        y = glxy[i, 1]
        if ndf == 1:
            lines.append(f"{node:8d}   {x:0.5E}   {y:0.5E}   {field[i,0]:0.5E}")
        elif ndf == 2:
            lines.append(
                f"{node:8d}   {x:0.5E}   {y:0.5E}   {field[i,0]:0.5E}   {field[i,1]:0.5E}"
            )
        else:
            lines.append(
                f"{node:8d}   {x:0.5E}   {y:0.5E}   {field[i,0]:0.5E}   {field[i,1]:0.5E}   {field[i,2]:0.5E}"
            )
    lines.append(line)
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
