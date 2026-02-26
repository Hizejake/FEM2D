import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src import mesh, elements, driver
from src.io import BoundaryConditions
from src.assemble import assemble_global, assemble_mass_global
from src.boundary import apply_dirichlet_bc, _map_dof


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


def test_dynamic_initial_acceleration_includes_damping():
    nod, glxy = mesh.generate_rectangular_mesh(
        nx=1, ny=1, x0=0.0, dx=np.array([1.0]), y0=0.0, dy=np.array([1.0]), ieltyp=0, npe=3
    )
    coeffs = {'A11': 0.0, 'A22': 0.0, 'A00': 0.0, 'F0': 0.0}
    # Constrain 3/4 nodes -> one free DOF.
    bc = BoundaryConditions(
        ispv=np.array([[1, 1], [3, 1], [4, 1]]),
        vspv=np.zeros(3),
    )
    init_v = np.array([0.0, 1.0, 0.0, 0.0])
    dyn_base = {
        'C0': 1.0, 'CX': 0.0, 'CY': 0.0,
        'ntime': 1, 'dt': 1.0, 'alfa': 0.25, 'gama': 0.5,
        'intial': 1,
        'initial_u': np.zeros(4), 'initial_v': init_v,
        'damping': {'alpha': 1.0, 'beta': 0.0},
    }

    # Build the same linearized system used by solve_dynamic and compute explicit a0.
    K, F = assemble_global(nod, glxy, 3, 1, elements.element_matrices, 0, None, coeffs)
    M = assemble_mass_global(nod, glxy, 3, 1, elements.triangle_element_mass, 0, coeffs, dyn_base)
    apply_dirichlet_bc(K, F, bc, 1)
    for i in range(bc.ispv.shape[0]):
        g = _map_dof(bc.ispv[i, 0], bc.ispv[i, 1], 1)
        M[g, :] = 0.0
        M[:, g] = 0.0
        M[g, g] = 1.0
    C = dyn_base['damping']['alpha'] * M + dyn_base['damping']['beta'] * K
    expected_a0 = np.linalg.solve(M, F - K @ dyn_base['initial_u'] - C @ dyn_base['initial_v'])

    u_auto, _ = driver.solve_dynamic(
        nod, glxy, 3, ndf=1, itype=0,
        element_func=elements.element_matrices,
        mass_func=elements.triangle_element_mass,
        coeffs=coeffs, bc=bc, convection=None, dyn=dyn_base
    )

    dyn_with_a = dict(dyn_base)
    dyn_with_a['initial_a'] = expected_a0
    u_explicit, _ = driver.solve_dynamic(
        nod, glxy, 3, ndf=1, itype=0,
        element_func=elements.element_matrices,
        mass_func=elements.triangle_element_mass,
        coeffs=coeffs, bc=bc, convection=None, dyn=dyn_with_a
    )
    assert np.allclose(u_auto, u_explicit, atol=1e-12)
