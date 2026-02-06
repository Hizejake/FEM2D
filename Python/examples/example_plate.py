import sys
import os
import numpy as np

# Make local `src` package importable when running this example from the Python folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import mesh, elements, driver
from src.io import BoundaryConditions


def main():
    # Simple 2x2 rectangular mesh (each cell split into 2 triangles)
    nx = 2
    ny = 2
    x0 = 0.0
    y0 = 0.0
    dx = np.array([1.0] * nx)
    dy = np.array([1.0] * ny)
    ieltyp = 0  # triangular
    npe = 3

    nod, glxy = mesh.generate_rectangular_mesh(nx, ny, x0, dx, y0, dy, ieltyp, npe)

    # Problem coefficients: Poisson with constant source F0
    coeffs = {'A11': 1.0, 'A22': 1.0, 'A00': 0.0, 'F0': 1.0}

    # Dirichlet BC: fix left edge nodes (x == 0) to zero
    left_nodes = [i for i, (x, y) in enumerate(glxy) if abs(x - 0.0) < 1e-12]
    # BoundaryConditions expects Fortran-style 1-based node numbers
    ispv = np.array([[n + 1, 1] for n in left_nodes], dtype=int)
    vspv = np.zeros(len(left_nodes), dtype=float)
    bc = BoundaryConditions(ispv=ispv, vspv=vspv)

    # Solve
    u, field = driver.solve_static(nod, glxy, npe, ndf=1, itype=0,
                                   element_func=elements.element_matrices,
                                   material=None, coeffs=coeffs, bc=bc, convection=None)

    # Print nodal solution and optionally write CSV
    print("NodeID, X, Y, u")
    for i, (xy, ui) in enumerate(zip(glxy, u)):
        print(f"{i+1}, {xy[0]:.6f}, {xy[1]:.6f}, {ui:.8e}")

    out_csv = os.path.join(PROJECT_ROOT, 'example_solution.csv')
    with open(out_csv, 'w') as f:
        f.write('node,x,y,u\n')
        for i, (xy, ui) in enumerate(zip(glxy, u)):
            f.write(f"{i+1},{xy[0]},{xy[1]},{ui}\n")
    print(f"Saved nodal solution to {out_csv}")


if __name__ == '__main__':
    main()
