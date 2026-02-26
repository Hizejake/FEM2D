import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src import mesh, elements, assemble


def test_assemble_mass_triangle_shape():
    nod, glxy = mesh.generate_rectangular_mesh(
        nx=1, ny=1, x0=0.0, dx=np.array([1.0]), y0=0.0, dy=np.array([1.0]), ieltyp=0, npe=3
    )
    dyn = {'C0': 1.0, 'CX': 0.0, 'CY': 0.0}
    M = assemble.assemble_mass_global(nod, glxy, npe=3, ndf=1, mass_func=elements.triangle_element_mass, itype=0, dyn=dyn)
    assert M.shape == (glxy.shape[0], glxy.shape[0])
    assert np.allclose(M, M.T, atol=1e-12)
