import sys
import os
import numpy as np

# ensure src package importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src import mesh, elements, assemble


def test_assemble_triangle_poisson():
    # single square split into 2 triangles
    nx = 1
    ny = 1
    x0 = 0.0
    y0 = 0.0
    dx = np.array([1.0])
    dy = np.array([1.0])
    ieltyp = 0
    npe = 3
    nod, glxy = mesh.generate_rectangular_mesh(nx, ny, x0, dx, y0, dy, ieltyp, npe)
    # assemble scalar problem
    coeffs = {'A11': 1.0, 'A22': 1.0, 'A00': 0.0, 'F0': 1.0}
    K, F = assemble.assemble_global(nod, glxy, npe, ndf=1, element_func=elements.element_matrices,
                                     itype=0, material=None, coeffs=coeffs)
    # two triangles covering unit square => total area = 1.0
    assert K.shape[0] == glxy.shape[0]
    assert np.allclose(K, K.T)
    assert np.isclose(F.sum(), 1.0, atol=1e-12)
