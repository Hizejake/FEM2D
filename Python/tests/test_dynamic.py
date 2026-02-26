import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src import mesh, elements, driver
from src.io import BoundaryConditions


def test_dynamic_constant_zero():
    # use same small mesh as before
    nx = 1
    ny = 1
    x0 = 0.0
    y0 = 0.0
    dx = np.array([1.0])
    dy = np.array([1.0])
    ieltyp = 0
    npe = 3
    nod, glxy = mesh.generate_rectangular_mesh(nx, ny, x0, dx, y0, dy, ieltyp, npe)
    # static stiffness coefficients not used in test
    coeffs = {'A11': 1.0, 'A22': 1.0, 'A00': 0.0, 'F0': 0.0}
    # Dirichlet BC at node 1
    bc = BoundaryConditions(ispv=np.array([[1,1]]), vspv=np.array([0.0]))
    # total DOFs = 4 (nodes in 2x2 mesh)
    dyn = {'C0': 1.0, 'CX': 0.0, 'CY': 0.0,
           'ntime': 3, 'dt': 0.5, 'alfa': 0.25, 'gama': 0.5,
           'initial_u': np.zeros(4), 'initial_v': np.zeros(4)}
    u, field = driver.solve_dynamic(nod, glxy, npe, ndf=1, itype=0,
                                    element_func=elements.element_matrices,
                                    mass_func=elements.triangle_element_mass,
                                    coeffs=coeffs, bc=bc, convection=None,
                                    dyn=dyn)
    # with zero load and fixed DOF, solution should remain zero
    assert np.allclose(u, 0.0, atol=1e-12)
