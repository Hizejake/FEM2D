import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src import solver


def test_solve_dense_simple():
    K = np.array([[2.0, 1.0], [1.0, 2.0]])
    F = np.array([3.0, 3.0])
    u = solver.solve_dense(K, F)
    u_ref = np.linalg.solve(K, F)
    assert np.allclose(u, u_ref)


def test_solve_spd():
    K = np.array([[4.0, 1.0], [1.0, 3.0]])
    F = np.array([1.0, 2.0])
    u = solver.solve_spd(K, F)
    assert np.allclose(K @ u, F)


def test_residual_norm():
    K = np.eye(3)
    F = np.array([1.0, 2.0, 3.0])
    u = solver.solve_dense(K, F)
    assert solver.residual_norm(K, u, F) < 1e-12
