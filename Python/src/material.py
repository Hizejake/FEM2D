"""
Material utilities for FEM2D: compute constitutive matrix CMAT
based on Fortran logic in FEM2DF15.FOR (plane elasticity and plate bending).
"""
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional


@dataclass
class CMATResult:
    cmat: np.ndarray  # 3x3
    c44: Optional[float] = None
    c55: Optional[float] = None


def compute_cmat_elasticity(e1: float, e2: float, anu12: float, g12: float,
                             thkns: float, lnstrs: Optional[int] = 1) -> CMATResult:
    """
    Compute material coefficient matrix CMAT for plane elasticity.

    lnstrs: 0 = plane strain, else plane stress (matches Fortran LNSTRS)
    Returns CMAT (3x3) where indices correspond to (1,2,3)->(xx,yy,xy) terms.
    """
    ANU12 = anu12
    ANU21 = ANU12 * e2 / e1
    DENOM = 1.0 - ANU12 * ANU21

    CMAT = np.zeros((3, 3), dtype=float)
    CMAT[2, 2] = g12 * thkns  # CMAT(3,3) in Fortran

    if lnstrs == 0:
        # Plane strain
        S0 = (1.0 - ANU12 - 2.0 * ANU12 * ANU21)
        CMAT[0, 0] = thkns * e1 * (1.0 - ANU12) / S0
        CMAT[0, 1] = thkns * e1 * ANU21 / S0
        CMAT[1, 1] = thkns * e2 * DENOM / S0 / (1.0 + ANU12)
    else:
        # Plane stress
        CMAT[0, 0] = thkns * e1 / DENOM
        CMAT[0, 1] = ANU21 * CMAT[0, 0]
        CMAT[1, 1] = e2 * CMAT[0, 0] / e1

    # Symmetrize
    CMAT[1, 0] = CMAT[0, 1]
    CMAT[2, 0] = 0.0
    CMAT[2, 1] = 0.0

    return CMATResult(cmat=CMAT)


def compute_cmat_plate(e1: float, e2: float, anu12: float, g12: float,
                       g13: float, g23: float, thkns: float, itype: int) -> CMATResult:
    """
    Compute CMAT for plate bending per Fortran logic.

    itype is used to distinguish FSDT vs classical but here both use same CMAT.
    Returns CMAT (3x3) and shear coefficients C44, C55.
    """
    ANU12 = anu12
    ANU21 = ANU12 * e2 / e1
    DENOM = 1.0 - ANU12 * ANU21

    CMAT = np.zeros((3, 3), dtype=float)
    CMAT[0, 0] = (thkns ** 3) * e1 / DENOM / 12.0
    CMAT[0, 1] = ANU21 * CMAT[0, 0]
    CMAT[1, 1] = e2 * CMAT[0, 0] / e1
    CMAT[2, 2] = g12 * (thkns ** 3) / 12.0

    SCF = 5.0 / 6.0
    C44 = SCF * g23 * thkns
    C55 = SCF * g13 * thkns

    # Symmetrize
    CMAT[1, 0] = CMAT[0, 1]
    CMAT[2, 0] = 0.0
    CMAT[2, 1] = 0.0

    return CMATResult(cmat=CMAT, c44=C44, c55=C55)


# Thin wrapper to select appropriate computation based on ITYPE and LNSTRS
def compute_cmat_from_config(itype: int, lnstrs: Optional[int],
                             e1: float, e2: float, anu12: float,
                             g12: float, thkns: float,
                             g13: Optional[float] = None,
                             g23: Optional[float] = None) -> CMATResult:
    if itype == 2:
        return compute_cmat_elasticity(e1, e2, anu12, g12, thkns, lnstrs)
    else:
        # Plate bending
        return compute_cmat_plate(e1, e2, anu12, g12, g13 or 0.0, g23 or 0.0, thkns, itype)
