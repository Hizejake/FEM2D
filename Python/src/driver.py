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
from .assemble import assemble_global
from .boundary import apply_neumann_bc, apply_convection_bc, apply_dirichlet_bc
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
