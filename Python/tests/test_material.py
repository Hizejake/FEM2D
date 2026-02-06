import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import src.material as material


def test_cmat_elasticity_plane_stress():
    # Simple isotropic case: E1=E2=E, ANU12=nu, G12 = E/(2(1+nu)), thk=1
    E = 210e9
    nu = 0.3
    G = E / (2 * (1 + nu))
    res = material.compute_cmat_elasticity(E, E, nu, G, 1.0, lnstrs=1)
    cmat = res.cmat
    # Check symmetry (CMAT should be symmetric in the 2x2 sub-block)
    assert np.allclose(cmat[0:2, 0:2], cmat[0:2, 0:2].T, atol=1e-12)


def test_cmat_plate_bending():
    E = 70e9
    nu = 0.25
    G12 = E / (2 * (1 + nu))
    G13 = G12
    G23 = G12
    thk = 0.01
    res = material.compute_cmat_plate(E, E, nu, G12, G13, G23, thk, itype=3)
    assert res.cmat.shape == (3, 3)
    assert res.c44 is not None and res.c55 is not None
