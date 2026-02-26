"""
Element routines for FEM2D - simplified implementations for linear
triangular and bilinear quadrilateral elements.

Supports:
- Poisson / diffusion (ITYPE=0) with coefficients A11, A22, A00
- Plane elasticity (ITYPE=2) with constitutive `CMAT` (3x3)

APIs:
- `triangle_element_matrices(coords, itype, material=None, coeffs=None)`
- `quad_element_matrices(coords, itype, material=None, coeffs=None)`

Both return (ELK, ELF) where ELK is (nn,nn) and ELF is (nn,)
"""
from typing import Tuple, Optional
import numpy as np


def _ndf_for_itype(itype: int) -> int:
    if itype == 0:
        return 1
    if itype == 2:
        return 2
    if itype in (3, 4):
        return 3
    raise NotImplementedError(f"ITYPE {itype} not implemented")


def _area_of_triangle(coords: np.ndarray) -> float:
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def triangle_shape_derivatives(coords: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    For linear triangle (3 nodes): compute derivatives of shape functions
    with respect to x,y (constant across element) and area.

    Returns dNdx (2 x 3) and area.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    # Compute area using determinant
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * detJ
    if abs(area) < 1e-12:
        raise ValueError("Triangle area is zero or degenerate")
    # Derivatives from formula for linear triangle
    dNdx = np.zeros((2, 3), dtype=float)
    dNdx[0, 0] = (y2 - y3) / detJ  # dN1/dx
    dNdx[1, 0] = (x3 - x2) / detJ  # dN1/dy
    dNdx[0, 1] = (y3 - y1) / detJ
    dNdx[1, 1] = (x1 - x3) / detJ
    dNdx[0, 2] = (y1 - y2) / detJ
    dNdx[1, 2] = (x2 - x1) / detJ
    return dNdx, abs(area)


def triangle_element_matrices(coords: np.ndarray,
                              itype: int = 0,
                              material: Optional[np.ndarray] = None,
                              coeffs: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
    """
    Compute element stiffness and load vector for a linear triangle.

    coords: (3,2) array of node coordinates
    itype: problem type (0=Poisson,2=Elasticity)
    material: for elasticity, a 3x3 `CMAT` matrix
    coeffs: dictionary with keys for Poisson: 'A11','A22','A00','F0','FX','FY'

    Returns (ELK, ELF)
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (3, 2):
        raise ValueError("coords must be (3,2) for triangle")

    dNdx, area = triangle_shape_derivatives(coords)
    # For elasticity, there are 2 DOF per node => 6x6 element matrix
    if itype == 2:
        ELK = np.zeros((6, 6), dtype=float)  # type: ignore
        ELF = np.zeros(6, dtype=float)  # type: ignore
    else:
        nn = 3
        ELK = np.zeros((nn, nn), dtype=float)
        ELF = np.zeros(nn, dtype=float)

    if itype == 0:
        # Poisson/diffusion: ELK_ij = integral (A11*dNi/dx*dNj/dx + A22*dNi/dy*dNj/dy + A00*Ni*Nj) dA
        A11 = coeffs.get('A11', 1.0)    # type: ignore    
        A22 = coeffs.get('A22', 1.0)    # type: ignore
        A00 = coeffs.get('A00', 0.0)    # type: ignore
        for i in range(nn):
            for j in range(nn):
                S11 = dNdx[0, i] * dNdx[0, j]
                S22 = dNdx[1, i] * dNdx[1, j]
                S00 = (1.0 / 3.0) if i == j else (1.0 / 6.0)  # integral Ni*Nj over triangle
                ELK[i, j] = (A11 * S11 + A22 * S22) * area + A00 * S00 * area
        # Load vector for source f = F0 + FX*x + FY*y
        F0 = coeffs.get('F0', 0.0)  # type: ignore
        FX = coeffs.get('FX', 0.0)  # type: ignore
        FY = coeffs.get('FY', 0.0)  # type: ignore
        # Use centroid for linear source: integral Ni * f dA ≈ f(centroid) * integral Ni dA
        centroid = coords.mean(axis=0)
        fcent = F0 + FX * centroid[0] + FY * centroid[1]
        for i in range(nn):
            ELF[i] = fcent * area / 3.0

    elif itype == 2:
        # Plane elasticity: build B matrix (3 x 6) for 3 nodes
        # For linear triangle, strain = B * u, where u = [u1,v1,u2,v2,u3,v3]
        # B = [[dN1/dx, 0, dN2/dx, 0, dN3/dx, 0], [0, dN1/dy, 0, dN2/dy, 0, dN3/dy], [dN1/dy, dN1/dx, ...]]
        B = np.zeros((3, 6), dtype=float)
        for a in range(3):
            B[0, 2 * a] = dNdx[0, a]
            B[1, 2 * a + 1] = dNdx[1, a]
            B[2, 2 * a] = dNdx[1, a]
            B[2, 2 * a + 1] = dNdx[0, a]
        CMAT = material if material is not None else np.eye(3)
        # Element stiffness: k = area * (B^T * CMAT * B)
        k6 = (B.T @ CMAT @ B) * area
        # Map to nodal DOFs (2 DOF per node): reorder to (u1,u2,...)? Already 6x6 so map accordingly
        ELK = k6
        # ELF: assume body force small; compute from F0/centroid as scalar per DOF (not rigorous)
        F0 = coeffs.get('F0', 0.0) if coeffs else 0.0
        for a in range(3):
            ELF[2 * a] = F0 * area / 3.0
            ELF[2 * a + 1] = 0.0
    else:
        raise NotImplementedError(f"ITYPE {itype} not implemented for triangle")

    return ELK, ELF


def _quad_shape_functions(xi: float, eta: float) -> np.ndarray:
    # order: N1=(-1-xi)(1-eta)/4 ??? We'll use standard bilinear order at corners (-1,-1),(1,-1),(1,1),(-1,1)
    N = np.zeros(4)
    N[0] = 0.25 * (1 - xi) * (1 - eta)
    N[1] = 0.25 * (1 + xi) * (1 - eta)
    N[2] = 0.25 * (1 + xi) * (1 + eta)
    N[3] = 0.25 * (1 - xi) * (1 + eta)
    return N


def _quad_shape_derivatives(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    # returns dN/dxi and dN/deta arrays length 4
    dNdxi = np.zeros(4)
    dNdeta = np.zeros(4)
    dNdxi[0] = -0.25 * (1 - eta)
    dNdxi[1] = 0.25 * (1 - eta)
    dNdxi[2] = 0.25 * (1 + eta)
    dNdxi[3] = -0.25 * (1 + eta)
    dNdeta[0] = -0.25 * (1 - xi)
    dNdeta[1] = -0.25 * (1 + xi)
    dNdeta[2] = 0.25 * (1 + xi)
    dNdeta[3] = 0.25 * (1 - xi)
    return dNdxi, dNdeta


def quad_element_matrices(coords: np.ndarray,
                          itype: int = 0,
                          material: Optional[np.ndarray] = None,
                          coeffs: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
    """
    Bilinear quad (4-node) element matrices using 2x2 Gauss quadrature.

    coords: (4,2) node coordinates ordered [(0,0),(1,0),(1,1),(0,1)] counter-clockwise
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (4, 2):
        raise ValueError("coords must be (4,2) for quad")
    nn = 4
    ndf = _ndf_for_itype(itype)
    ELK = np.zeros((nn * ndf, nn * ndf), dtype=float)
    ELF = np.zeros(nn * ndf, dtype=float)

    # 2x2 Gauss points
    gauss = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
             (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]

    for xi, eta in gauss:
        N = _quad_shape_functions(xi, eta)
        dNdxi, dNdeta = _quad_shape_derivatives(xi, eta)
        # Jacobian
        J = np.zeros((2, 2), dtype=float)
        for a in range(4):
            J[0, 0] += dNdxi[a] * coords[a, 0]
            J[0, 1] += dNdxi[a] * coords[a, 1]
            J[1, 0] += dNdeta[a] * coords[a, 0]
            J[1, 1] += dNdeta[a] * coords[a, 1]
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Jacobian determinant non-positive for quad element")
        invJ = np.linalg.inv(J)
        dNdx = np.zeros((2, 4))
        for a in range(4):
            grad = invJ @ np.array([dNdxi[a], dNdeta[a]])
            dNdx[0, a] = grad[0]
            dNdx[1, a] = grad[1]
        if itype == 0:
            A11 = coeffs.get('A11', 1.0)    # type: ignore
            A22 = coeffs.get('A22', 1.0)    # type: ignore
            A00 = coeffs.get('A00', 0.0)    # type: ignore
            # assemble
            for i in range(nn):
                for j in range(nn):
                    S11 = dNdx[0, i] * dNdx[0, j]
                    S22 = dNdx[1, i] * dNdx[1, j]
                    S00 = N[i] * N[j]
                    ELK[i, j] += (A11 * S11 + A22 * S22 + A00 * S00) * detJ
            # RHS: f at gauss point approximated
            # f = F0 + FX*x + FY*y
            xg = sum(N[a] * coords[a, 0] for a in range(4))
            yg = sum(N[a] * coords[a, 1] for a in range(4))
            F0 = coeffs.get('F0', 0.0)  # type: ignore
            FX = coeffs.get('FX', 0.0)  # type: ignore
            FY = coeffs.get('FY', 0.0)  # type: ignore
            fval = F0 + FX * xg + FY * yg
            for i in range(nn):
                ELF[i] += N[i] * fval * detJ
        elif itype == 2:
            # elasticity: build B (3 x 8)
            B = np.zeros((3, 8))
            for a in range(4):
                B[0, 2*a] = dNdx[0, a]
                B[1, 2*a+1] = dNdx[1, a]
                B[2, 2*a] = dNdx[1, a]
                B[2, 2*a+1] = dNdx[0, a]
            CMAT = material if material is not None else np.eye(3)
            k_local = B.T @ CMAT @ B * detJ
            # map k_local (8x8) into ELK (8x8)
            ELK += k_local
            # RHS: naive body force
            F0 = coeffs.get('F0', 0.0) if coeffs else 0.0
            for a in range(4):
                ELF[2*a] += F0 * N[a] * detJ
                ELF[2*a+1] += 0.0
        elif itype in (3, 4):
            # Mindlin plate with DOFs [w, theta_x, theta_y] at each node.
            Bb = np.zeros((3, 12), dtype=float)
            for a in range(4):
                ia = 3 * a
                Bb[0, ia + 1] = dNdx[0, a]
                Bb[1, ia + 2] = dNdx[1, a]
                Bb[2, ia + 1] = dNdx[1, a]
                Bb[2, ia + 2] = dNdx[0, a]

            if isinstance(material, dict):
                D_b = np.asarray(material.get('CMAT', np.eye(3)), dtype=float)
            else:
                D_b = np.asarray(material if material is not None else np.eye(3), dtype=float)
            ELK += Bb.T @ D_b @ Bb * detJ

            F0 = coeffs.get('F0', 0.0) if coeffs else 0.0
            FX = coeffs.get('FX', 0.0) if coeffs else 0.0
            FY = coeffs.get('FY', 0.0) if coeffs else 0.0
            xg = sum(N[a] * coords[a, 0] for a in range(4))
            yg = sum(N[a] * coords[a, 1] for a in range(4))
            q = F0 + FX * xg + FY * yg
            for a in range(4):
                ELF[3 * a] += q * N[a] * detJ
        else:
            raise NotImplementedError(f"ITYPE {itype} not implemented for quad")

    if itype in (3, 4):
        # Reduced integration for shear terms to limit locking.
        xi = 0.0
        eta = 0.0
        weight = 4.0
        N = _quad_shape_functions(xi, eta)
        dNdxi, dNdeta = _quad_shape_derivatives(xi, eta)
        J = np.zeros((2, 2), dtype=float)
        for a in range(4):
            J[0, 0] += dNdxi[a] * coords[a, 0]
            J[0, 1] += dNdxi[a] * coords[a, 1]
            J[1, 0] += dNdeta[a] * coords[a, 0]
            J[1, 1] += dNdeta[a] * coords[a, 1]
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Jacobian determinant non-positive for quad element")
        invJ = np.linalg.inv(J)
        dNdx = np.zeros((2, 4))
        for a in range(4):
            grad = invJ @ np.array([dNdxi[a], dNdeta[a]])
            dNdx[0, a] = grad[0]
            dNdx[1, a] = grad[1]

        Bs = np.zeros((2, 12), dtype=float)
        for a in range(4):
            ia = 3 * a
            Bs[0, ia] = dNdx[0, a]
            Bs[0, ia + 1] = N[a]
            Bs[1, ia] = dNdx[1, a]
            Bs[1, ia + 2] = N[a]

        if isinstance(material, dict):
            c44 = float(material.get('C44', 0.0))
            c55 = float(material.get('C55', 0.0))
        else:
            c44 = 0.0
            c55 = 0.0
        D_s = np.array([[c55, 0.0], [0.0, c44]], dtype=float)
        ELK += Bs.T @ D_s @ Bs * detJ * weight

    return ELK, ELF


# --- mass matrix routines for dynamic analysis ---
def triangle_element_mass(coords: np.ndarray,
                           itype: int = 0,
                           coeffs: Optional[dict] = None,
                           dyn: Optional[dict] = None) -> np.ndarray:
    """Return element mass matrix for linear triangle.

    * `coeffs` is ignored (stiffness information).
    * `dyn` may provide material density coefficients C0, CX, CY (spatially
      linear density). If `dyn` is None or missing values, density is taken as
      1.0.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (3, 2):
        raise ValueError("coords must be (3,2) for triangle")

    # compute centroid and area
    centroid = coords.mean(axis=0)
    area = _area_of_triangle(coords)
    C0 = dyn.get('C0', 1.0) if dyn else 1.0
    CX = dyn.get('CX', 0.0) if dyn else 0.0
    CY = dyn.get('CY', 0.0) if dyn else 0.0
    CT = C0 + CX * centroid[0] + CY * centroid[1]

    nn = 3
    ELM = np.zeros((nn, nn), dtype=float)
    for i in range(nn):
        for j in range(nn):
            S00 = (1.0 / 3.0) if i == j else (1.0 / 6.0)
            ELM[i, j] = CT * S00 * area
    # if elasticity, expand to 2*nn x 2*nn, mass on diagonal DOFs
    if itype == 2:
        M2 = np.zeros((6, 6), dtype=float)
        for a in range(3):
            for b in range(3):
                M2[2 * a, 2 * b] = ELM[a, b]
                M2[2 * a + 1, 2 * b + 1] = ELM[a, b]
        return M2
    if itype in (3, 4):
        MX = dyn.get('CX', C0) if dyn else C0
        MY = dyn.get('CY', C0) if dyn else C0
        M3 = np.zeros((9, 9), dtype=float)
        for a in range(3):
            for b in range(3):
                S00 = (1.0 / 3.0) if a == b else (1.0 / 6.0)
                ia = 3 * a
                ib = 3 * b
                M3[ia, ib] = CT * S00 * area
                M3[ia + 1, ib + 1] = MX * S00 * area
                M3[ia + 2, ib + 2] = MY * S00 * area
        return M3
    return ELM


def quad_element_mass(coords: np.ndarray,
                       itype: int = 0,
                       coeffs: Optional[dict] = None,
                       dyn: Optional[dict] = None) -> np.ndarray:
    """Return element mass matrix for bilinear quad using 2x2 Gauss rule."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (4, 2):
        raise ValueError("coords must be (4,2) for quad")
    nn = 4
    nvars = nn * _ndf_for_itype(itype)
    ELM = np.zeros((nvars, nvars), dtype=float)
    # default dyn coefficients
    C0 = dyn.get('C0', 1.0) if dyn else 1.0
    CX = dyn.get('CX', 0.0) if dyn else 0.0
    CY = dyn.get('CY', 0.0) if dyn else 0.0

    gauss = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
             (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]

    for xi, eta in gauss:
        N = _quad_shape_functions(xi, eta)
        dNdxi, dNdeta = _quad_shape_derivatives(xi, eta)
        J = np.zeros((2, 2), dtype=float)
        for a in range(4):
            J[0, 0] += dNdxi[a] * coords[a, 0]
            J[0, 1] += dNdxi[a] * coords[a, 1]
            J[1, 0] += dNdeta[a] * coords[a, 0]
            J[1, 1] += dNdeta[a] * coords[a, 1]
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        xg = sum(N[a] * coords[a, 0] for a in range(4))
        yg = sum(N[a] * coords[a, 1] for a in range(4))
        CT = C0 + CX * xg + CY * yg
        # accumulate
        for i in range(nn):
            for j in range(nn):
                mass_ij = N[i] * N[j] * CT * detJ
                if itype == 0:
                    ELM[i, j] += mass_ij
                elif itype == 2:
                    ELM[2 * i, 2 * j] += mass_ij
                    ELM[2 * i + 1, 2 * j + 1] += mass_ij
                elif itype in (3, 4):
                    ELM[3 * i, 3 * j] += mass_ij
                    ELM[3 * i + 1, 3 * j + 1] += N[i] * N[j] * CX * detJ
                    ELM[3 * i + 2, 3 * j + 2] += N[i] * N[j] * CY * detJ
    return ELM

# Convenience wrapper selecting by NPE
def element_matrices(coords: np.ndarray, npe: int, itype: int = 0,
                     material: Optional[np.ndarray] = None,
                     coeffs: Optional[dict] = None):  # type: ignore
    """Dispatch to the appropriate element stiffness/load routine.

    Returns (ELK, ELF) pair.  
    """
    if npe == 3:
        return triangle_element_matrices(coords, itype, material, coeffs or {})
    if npe == 4:
        return quad_element_matrices(coords, itype, material, coeffs or {})
    raise NotImplementedError(f"Element with NPE={npe} not implemented")


