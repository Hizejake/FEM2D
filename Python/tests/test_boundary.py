import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src import boundary
from src.io import BoundaryConditions


def test_apply_dirichlet():
    # simple 3x3 K, F; set DOF 2 to value 5.0
    K = np.arange(9.0).reshape(3,3)
    K = (K + K.T) * 0.5
    F = np.array([0.0, 0.0, 0.0])
    # Fortran-style node/local_dof: use node=2, local=1 mapping to global DOF index 1 (with ndf=1)
    bc = BoundaryConditions(ispv = np.array([[2,1]]), vspv = np.array([5.0]))
    boundary.apply_dirichlet_bc(K, F, bc, ndf=1)
    assert K[1,1] == 1.0
    assert F[1] == 5.0
    # other rows zero except diag
    assert np.allclose(K[1,0], 0.0)
    assert np.allclose(K[0,1], 0.0)


def test_apply_dirichlet_nonzero_rhs_correction():
    K = np.array([[2.0, 1.0], [1.0, 2.0]])
    F = np.array([0.0, 0.0])
    bc = BoundaryConditions(ispv=np.array([[2, 1]]), vspv=np.array([5.0]))
    boundary.apply_dirichlet_bc(K, F, bc, ndf=1)
    # Remaining free equation must include -K[:,g]*u_g contribution.
    assert np.isclose(F[0], -5.0)
    assert np.isclose(F[1], 5.0)


def test_apply_neumann():
    F = np.zeros(3)
    bc = BoundaryConditions(ispv = np.array([]).reshape(0,2), issv = np.array([[3,1]]), vssv = np.array([2.5]))
    boundary.apply_neumann_bc(F, bc, ndf=1)
    # node 3 -> index 2
    assert F[2] == 2.5


def test_apply_convection():
    # two nodes at (0,0) and (1,0)
    glxy = np.array([[0.0,0.0],[1.0,0.0]])
    K = np.zeros((2,2))
    F = np.zeros(2)
    class Conv:
        pass
    conv = Conv()
    conv.iconv = 1
    conv.nbe = 1
    conv.inod = np.array([[1,2]])
    conv.beta = np.array([2.0])
    conv.tinf = np.array([3.0])
    boundary.apply_convection_bc(K, F, conv, nod=None, glxy=glxy, ndf=1)
    # edge length =1, k_edge = beta*L/6 * [[2,1],[1,2]] = (2/6)*[[2,1],[1,2]]
    expected_k = (2.0/6.0) * np.array([[2.0,1.0],[1.0,2.0]])
    assert np.allclose(K, expected_k)
    expected_f = (2.0 * 3.0 * 1.0 / 2.0) * np.array([1.0,1.0])
    assert np.allclose(F, expected_f)
