import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src import mesh, elements, driver
from src.io import BoundaryConditions


def test_end_to_end_solve():
    # Build 1x1 square split into 2 triangles
    nx = 1
    ny = 1
    x0 = 0.0
    y0 = 0.0
    dx = np.array([1.0])
    dy = np.array([1.0])
    ieltyp = 0
    npe = 3
    nod, glxy = mesh.generate_rectangular_mesh(nx, ny, x0, dx, y0, dy, ieltyp, npe)

    # Apply Dirichlet BC: fix node 1 (Fortran-style 1) to zero
    # BoundaryConditions expects Fortran-style indices in ispv and issv arrays
    bc = BoundaryConditions(ispv = np.array([[1,1]]), vspv = np.array([0.0]))

    coeffs = {'A11': 1.0, 'A22': 1.0, 'A00': 0.0, 'F0': 1.0}

    u, field = driver.solve_static(nod, glxy, npe, ndf=1, itype=0,
                                   element_func=elements.element_matrices,
                                   material=None, coeffs=coeffs, bc=bc, convection=None)
    # solution vector length = number of nodes
    assert u.size == glxy.shape[0]
    # check that Dirichlet enforced
    assert abs(u[0]) < 1e-12
    # residual should be small
    # compute residual K u - F via re-assembly (sanity check)
    K, F = __import__('src.assemble', fromlist=['assemble']).assemble_global(nod, glxy, npe, 1, elements.element_matrices, 0, None, coeffs)
    # apply same BC and convection
    __import__('src.boundary', fromlist=['boundary']).apply_neumann_bc(F, bc, 1)
    __import__('src.boundary', fromlist=['boundary']).apply_dirichlet_bc(K, F, bc, 1)
    res = K.dot(u) - F
    assert np.linalg.norm(res) < 1e-9
