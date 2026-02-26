import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src import mesh, elements, driver
from src.io import BoundaryConditions


def test_static_nonzero_dirichlet_enforced():
    nod, glxy = mesh.generate_rectangular_mesh(
        nx=1, ny=1, x0=0.0, dx=np.array([1.0]), y0=0.0, dy=np.array([1.0]), ieltyp=0, npe=3
    )
    coeffs = {'A11': 1.0, 'A22': 1.0, 'A00': 0.0, 'F0': 0.0}
    bc = BoundaryConditions(ispv=np.array([[1, 1]]), vspv=np.array([2.5]))
    u, _ = driver.solve_static(
        nod, glxy, npe=3, ndf=1, itype=0,
        element_func=elements.element_matrices,
        material=None, coeffs=coeffs, bc=bc, convection=None
    )
    assert np.isclose(u[0], 2.5)
