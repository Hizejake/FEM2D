"""
Postprocessing helpers for FEM2D.

Provides small utilities to convert solution vectors into nodal fields and
compute simple derived quantities.
"""
from typing import Tuple
import numpy as np


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
