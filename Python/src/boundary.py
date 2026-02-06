"""
Boundary condition application utilities.

Functions:
- apply_dirichlet_bc(K, F, bc, ndf): apply essential BCs (Dirichlet) using elimination
- apply_neumann_bc(F, bc, ndf): add natural BC contributions to RHS
- apply_convection_bc(K, F, convection, nod, glxy, ndf): simple edge convection model
"""
from typing import Optional
import numpy as np
from .io import BoundaryConditions, ConvectionBC


def _map_dof(node: int, local_dof: int, ndf: int) -> int:
    # Convert Fortran-style 1-based node/local_dof to 0-based global DOF
    node_idx = int(node) - 1
    ld = int(local_dof) - 1
    return node_idx * ndf + ld


def apply_dirichlet_bc(K: np.ndarray, F: np.ndarray, bc: BoundaryConditions, ndf: int) -> None:
    """Apply essential (Dirichlet) boundary conditions in-place.

    K and F are modified so that prescribed DOFs take the specified values.
    This uses row/column elimination: zero row/col, set diag to 1, set RHS to value.
    """
    if bc is None:
        return
    ispv = bc.ispv
    vspv = bc.vspv
    if ispv is None or ispv.size == 0:
        return
    if vspv is None:
        vspv = np.zeros(ispv.shape[0], dtype=float)

    for i in range(ispv.shape[0]):
        node = ispv[i, 0]
        ldof = ispv[i, 1]
        g = _map_dof(node, ldof, ndf)
        # zero row and column
        K[g, :] = 0.0
        K[:, g] = 0.0
        K[g, g] = 1.0
        F[g] = float(vspv[i])


def apply_neumann_bc(F: np.ndarray, bc: BoundaryConditions, ndf: int) -> None:
    """Apply natural (Neumann) BCs by adding to RHS vector F in-place.

    `issv` contains rows of (node, local_dof) and `vssv` holds specified values.
    """
    if bc is None:
        return
    issv = bc.issv
    vssv = bc.vssv
    if issv is None or issv.size == 0:
        return
    if vssv is None:
        vssv = np.zeros(issv.shape[0], dtype=float)

    for i in range(issv.shape[0]):
        node = issv[i, 0]
        ldof = issv[i, 1]
        g = _map_dof(node, ldof, ndf)
        F[g] += float(vssv[i])


def apply_convection_bc(K: np.ndarray, F: np.ndarray, convection: ConvectionBC,
                        nod: np.ndarray, glxy: np.ndarray, ndf: int) -> None:
    """Apply simple convection BCs on element boundary edges.

    This implements a linear edge model: for each boundary entry, add
    local 2x2 edge stiffness = (beta*L/6)*[[2,1],[1,2]] and RHS = beta*tinf*(L/2)*[1,1]
    mapped to the global DOFs for scalar problems (ndf==1). For vector problems
    the function applies the scalar convection to the specified local DOF only.
    """
    if convection is None:
        return
    if not hasattr(convection, 'iconv') or convection.iconv == 0:
        return
    if convection.nbe is None or convection.inod is None:
        return

    for i in range(convection.nbe):
        # element number (1-based in Fortran) but we only need node pair indices
        node_a = int(convection.inod[i, 0]) - 1
        node_b = int(convection.inod[i, 1]) - 1
        xa, ya = glxy[node_a]
        xb, yb = glxy[node_b]
        L = float(np.hypot(xb - xa, yb - ya))
        beta = float(convection.beta[i]) if convection.beta is not None else 0.0
        tinf = float(convection.tinf[i]) if convection.tinf is not None else 0.0
        # local 2x2 stiffness
        k_edge = (beta * L / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        f_edge = (beta * tinf * L / 2.0) * np.array([1.0, 1.0])
        # map to global DOFs; assume DOF index 1 (scalar) or use local DOF mapping if provided
        if ndf == 1:
            g1 = node_a
            g2 = node_b
            F[g1] += f_edge[0]
            F[g2] += f_edge[1]
            K[g1, g1] += k_edge[0, 0]
            K[g1, g2] += k_edge[0, 1]
            K[g2, g1] += k_edge[1, 0]
            K[g2, g2] += k_edge[1, 1]
        else:
            # apply to first DOF of each node (user may supply local dof mapping in future)
            g1 = node_a * ndf
            g2 = node_b * ndf
            F[g1] += f_edge[0]
            F[g2] += f_edge[1]
            K[g1, g1] += k_edge[0, 0]
            K[g1, g2] += k_edge[0, 1]
            K[g2, g1] += k_edge[1, 0]
            K[g2, g2] += k_edge[1, 1]
