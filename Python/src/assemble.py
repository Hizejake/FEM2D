"""
Global assembly utilities for FEM2D.
Produces dense global stiffness matrix and load vector from element matrices.

API:
- assemble_global(nod, glxy, npe, ndf, element_func, itype, material=None, coeffs=None)

Returns (K, F) where K is (neq, neq) ndarray and F is (neq,) ndarray.
"""
from typing import Callable, Tuple, Optional
import numpy as np
from .io import UserMesh


def assemble_global(nod: np.ndarray,
                    glxy: np.ndarray,
                    npe: int,
                    ndf: int,
                    element_func: Callable,
                    itype: int = 0,
                    material: Optional[np.ndarray] = None,
                    coeffs: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble dense global stiffness matrix and load vector.

    nod: (nem, npe) connectivity array (0-based node indices)
    glxy: (nnm, 2) coordinates
    npe: nodes per element
    ndf: DOF per node (1 for scalar problems, 2 for plane elasticity)
    element_func: function that returns (ELK, ELF) for a single element given (coords, npe, itype, ...)
    material, coeffs: forwarded to element_func

    Notes: This routine uses dense assembly (NumPy arrays). For large models
    replace with sparse assembly (e.g. SciPy CSR) if needed.
    """
    nod = np.asarray(nod, dtype=int)
    glxy = np.asarray(glxy, dtype=float)
    nem = nod.shape[0]
    nnm = glxy.shape[0]
    neq = nnm * ndf

    K = np.zeros((neq, neq), dtype=float)
    F = np.zeros(neq, dtype=float)

    for e in range(nem):
        el_nodes = nod[e]
        coords = glxy[el_nodes]
        # element_func signature expected like element_matrices(coords, npe, itype, material, coeffs)
        ELK, ELF = element_func(coords, npe=npe, itype=itype, material=material, coeffs=coeffs)  # type: ignore
        ndof_e = ELK.shape[0]
        # map local DOFs to global DOF indices
        if ndf == 1:
            gidx = el_nodes
        else:
            gidx = np.zeros(ndof_e, dtype=int)
            for a, node in enumerate(el_nodes):
                for d in range(ndf):
                    gidx[ndf * a + d] = node * ndf + d
        # assemble
        for i_local, I in enumerate(gidx):
            F[I] += ELF[i_local]
            for j_local, J in enumerate(gidx):
                K[I, J] += ELK[i_local, j_local]

    return K, F
