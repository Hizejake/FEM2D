import sys
import os
import numpy as np
import pytest

# Ensure local `src` package is importable when running tests from repository root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src import elements


def test_triangle_poisson():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    coeffs = {'A11': 1.0, 'A22': 1.0, 'A00': 0.0, 'F0': 1.0}
    ELK, ELF = elements.element_matrices(coords, npe=3, itype=0, coeffs=coeffs)
    assert ELK.shape == (3, 3)
    # stiffness should be symmetric
    assert np.allclose(ELK, ELK.T, atol=1e-12)
    area = 0.5
    # sum of load vector should equal area * F0 (centroid approx)
    assert pytest.approx(area * coeffs['F0'], rel=1e-6) == ELF.sum()


def test_triangle_elasticity():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    CMAT = np.eye(3) * 1e3
    ELK, ELF = elements.element_matrices(coords, npe=3, itype=2, material=CMAT, coeffs={'F0': 0.0})
    # Elastic triangle returns 6x6
    assert ELK.shape == (6, 6)
    assert np.allclose(ELK, ELK.T, atol=1e-12)


def test_quad_poisson():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    coeffs = {'A11': 1.0, 'A22': 1.0, 'A00': 0.0, 'F0': 2.0}
    ELK, ELF = elements.element_matrices(coords, npe=4, itype=0, coeffs=coeffs)
    assert ELK.shape == (4, 4)
    assert np.allclose(ELK, ELK.T, atol=1e-12)
    area = 1.0
    assert pytest.approx(area * coeffs['F0'], rel=1e-6) == pytest.approx(ELF.sum(), rel=1e-6)
