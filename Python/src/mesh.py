"""
Mesh generation utilities for FEM2D.
Implements rectangular mesh generator analogous to Fortran MESH2DR.
Outputs `UserMesh` (connectivity and coordinates) using 0-based indices.
"""
from typing import Tuple
import numpy as np
from .io import RectangularMesh, UserMesh, FEM2DConfig


def generate_rectangular_mesh(nx: int, ny: int, x0: float, dx: np.ndarray,
                              y0: float, dy: np.ndarray, ieltyp: int,
                              npe: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a rectangular mesh similar to Fortran MESH2DR.

    Returns:
        nod: (nem, npe) int ndarray with 0-based node indices
        glxy: (nnm, 2) float ndarray of node coordinates

    Notes:
    - If ieltyp == 0 (triangles), each rectangle is split into two triangles
      (2 elements per cell).
    - If ieltyp != 0 (quadrilaterals), create one element per cell.
    - The Fortran code numbers nodes such that NXX1 = NXX+1, NYY1 = NYY+1.
      We'll follow row-major ordering (y-major) consistent with the original.
    """
    # Validate inputs
    assert nx >= 1 and ny >= 1, "nx and ny must be >= 1"
    assert dx.shape[0] == nx, "dx length must equal nx"
    assert dy.shape[0] == ny, "dy length must equal ny"

    # Number of nodes along x and y directions for the inflated grid
    # Fortran uses: NXX = IEL * NX; NXX1 = NXX + 1; NEM = NX * NY (or 2*NX*NY for triangles)
    IEL = 1 if ieltyp == 0 else 1  # keep IEL=1 semantics for now
    NXX = IEL * nx
    NYY = IEL * ny
    NXX1 = NXX + 1
    NYY1 = NYY + 1

    # For simple rectangular mesh with linear quads, nodes = (NXX1)*(NYY1)
    nnm = NXX1 * NYY1

    # Build x coordinates by repeating dx segments as necessary.
    # In Fortran MESH2DR, DX has NX entries; each may represent element widths.
    xs = [x0]
    for i in range(nx):
        # each DX(i) may correspond to one element width; if IEL>1 there are subdivisions
        step = dx[i] / 1.0
        # append next node coordinate
        xs.append(xs[-1] + step)
    # If IEL>1 we would subdivide further; for now NX subdivisions produce NXX = NX
    # Build full coordinate arrays by simple cumulative sums and tiling
    xcoords = np.zeros(NXX1)
    xcoords[0] = x0
    # Fortran may repeat patterns; we'll use cumulative sums of dx
    cumulative = x0
    idx = 1
    for i in range(nx):
        cumulative += dx[i]
        xcoords[idx] = cumulative
        idx += 1
    # If more nodes expected (due to IEL), replicate last coordinate
    while idx < NXX1:
        xcoords[idx] = xcoords[idx - 1]
        idx += 1

    ycoords = np.zeros(NYY1)
    ycoords[0] = y0
    cumulative = y0
    idx = 1
    for j in range(ny):
        cumulative += dy[j]
        ycoords[idx] = cumulative
        idx += 1
    while idx < NYY1:
        ycoords[idx] = ycoords[idx - 1]
        idx += 1

    # Create grid of node coordinates (row-major: y changes slowest or fastest?)
    # We'll follow standard: node index = ix + iy * NXX1
    glxy = np.zeros((nnm, 2), dtype=float)
    n = 0
    for iy in range(NYY1):
        for ix in range(NXX1):
            glxy[n, 0] = xcoords[ix]
            glxy[n, 1] = ycoords[iy]
            n += 1

    # Build element connectivity
    if ieltyp == 0:
        # Triangular: split each rectangular cell into two triangles
        nem = 2 * nx * ny
        nod = np.zeros((nem, 3), dtype=int)
        e = 0
        for j in range(ny):
            for i in range(nx):
                # lower-left node index
                n1 = i + j * NXX1
                n2 = (i + 1) + j * NXX1
                n3 = i + (j + 1) * NXX1
                n4 = (i + 1) + (j + 1) * NXX1
                # Triangle 1: (n1, n2, n4)
                nod[e, 0] = n1
                nod[e, 1] = n2
                nod[e, 2] = n4
                e += 1
                # Triangle 2: (n1, n4, n3)
                nod[e, 0] = n1
                nod[e, 1] = n4
                nod[e, 2] = n3
                e += 1
    else:
        # Quadrilateral element: 4 nodes per element
        nem = nx * ny
        nod = np.zeros((nem, 4), dtype=int)
        e = 0
        for j in range(ny):
            for i in range(nx):
                n1 = i + j * NXX1
                n2 = (i + 1) + j * NXX1
                n3 = (i + 1) + (j + 1) * NXX1
                n4 = i + (j + 1) * NXX1
                nod[e, 0] = n1
                nod[e, 1] = n2
                nod[e, 2] = n3
                nod[e, 3] = n4
                e += 1

    return nod, glxy


def generate_mesh_from_config(config: FEM2DConfig) -> UserMesh:
    """
    Generate or normalize mesh according to `config` produced by `io.read_inp()`.

    If `config.element_mesh.mesh == 1` (rectangular) this will call
    `generate_rectangular_mesh`. If `mesh == 0` it will normalize the
    already provided `UserMesh` from `config.user_mesh` converting to 0-based
    indices.
    """
    em = config.element_mesh
    if em.mesh == 1:
        rm = config.mesh_data
        nod, glxy = generate_rectangular_mesh(rm.nx, rm.ny, rm.x0, rm.dx, # type: ignore
                                             rm.y0, rm.dy, em.ieltyp, em.npe)   # type: ignore
        # Update counts
        em.nem = nod.shape[0]
        em.nnm = glxy.shape[0]
        return UserMesh(nod, glxy)
    elif em.mesh == 0:
        # Normalize user mesh: convert 1-based Fortran indices to 0-based
        if config.user_mesh is None:
            raise ValueError("user_mesh expected but not provided in config")
        nod = config.user_mesh.nod.astype(int) - 1
        glxy = config.user_mesh.glxy.astype(float)
        em.nem = nod.shape[0]
        em.nnm = glxy.shape[0]
        return UserMesh(nod, glxy)
    else:
        raise NotImplementedError("General mesh generation (MESH2DG) not implemented yet")
