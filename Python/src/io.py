"""
Input/Output module for FEM2D plate static analysis.

Parses FEM2D input files (.INP) following the format specified in FEM2DF15.FOR
and returns structured configuration objects for downstream processing.

Fortran reference: Lines 154-400 of FEM2DF15.FOR (PREPROCESSOR UNIT)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from pathlib import Path


# ============================================================================
# DATACLASSES for structured input representation
# ============================================================================

@dataclass
class ProblemType:
    """
    Fortran: READ(IN,*) ITYPE, IGRAD, ITEM, NEIGN
    Line: 158
    
    Attributes:
        itype: Problem type
            0 = Heat transfer / Poisson problem
            1 = Viscous incompressible flow
            2 = Plane elasticity
            3 = Plate bending (FSDT - First-order Shear Deformation Theory)
            4 = Plate bending (Classical Plate Theory)
            5 = Plate bending (CST Conforming element variant)
        igrad: Gradient computation flag
            0 = Do not compute gradient/stress
            1 = Compute gradient/stress
        item: Transient/dynamic analysis flag
            0 = Steady-state analysis
            1 = Transient analysis (single variable)
            2 = Transient analysis (with velocity/acceleration)
        neign: Eigenvalue problem flag
            0 = No eigenvalue analysis
            1 = Standard eigenvalue problem
            2+ = Generalized eigenvalue problem
        nvalu: (Optional, if neign != 0) Number of eigenvalues to extract
        nvctr: (Optional, if neign != 0) Print eigenvectors flag (0/1)
    """
    itype: int
    igrad: int
    item: int
    neign: int
    nvalu: Optional[int] = None
    nvctr: Optional[int] = None


@dataclass
class ElementMesh:
    """
    Fortran: READ(IN,*) IELTYP, NPE, MESH, NPRNT
    Line: 171
    
    Attributes:
        ieltyp: Element type
            0 = Triangular element
            1+ = Rectangular/quadrilateral element
        npe: Number of nodes per element
            3, 4 = Linear elements
            6, 8, 9 = Quadratic elements
        mesh: Mesh generation flag
            0 = User-provided mesh (read connectivity & coords)
            1 = Rectangular domain auto-generation (via MESH2DR)
            2 = General geometry auto-generation (via MESH2DG)
        nprnt: Print control flag
            0 = Minimal output
            1 = Print element matrices
            2 = Print global matrices
            3 = Print element + global matrices
        nem: Number of elements (computed or read)
        nnm: Number of nodes (computed or read)
    """
    ieltyp: int
    npe: int
    mesh: int
    nprnt: int
    nem: Optional[int] = None
    nnm: Optional[int] = None


@dataclass
class RectangularMesh:
    """
    Fortran: Lines 203-205 (when MESH == 1)
    READ(IN,*) NX, NY
    READ(IN,*) X0, (DX(I), I=1,NX)
    READ(IN,*) Y0, (DY(I), I=1,NY)
    
    Attributes:
        nx, ny: Number of subdivisions in x and y directions
        x0, y0: Starting coordinates
        dx: Array of x-direction segment lengths (length nx)
        dy: Array of y-direction segment lengths (length ny)
    """
    nx: int
    ny: int
    x0: float
    y0: float
    dx: np.ndarray  # shape (nx,)
    dy: np.ndarray  # shape (ny,)


@dataclass
class UserMesh:
    """
    Fortran: Lines 182, 189-190 (when MESH == 0)
    READ(IN,*) NEM, NNM
    READ(IN,*) (NOD(N,I), I=1,NPE)  [loop over NEM elements]
    READ(IN,*) ((GLXY(I,J), J=1,2), I=1,NNM)
    
    Attributes:
        nod: Element connectivity array, shape (nem, npe)
            nod[n, i] = global node number for element n, local node i
        glxy: Global node coordinates, shape (nnm, 2)
            glxy[i, 0] = x-coordinate of node i
            glxy[i, 1] = y-coordinate of node i
    """
    nod: np.ndarray    # shape (nem, npe)
    glxy: np.ndarray   # shape (nnm, 2)


@dataclass
class BoundaryConditions:
    """
    Fortran: Lines 240-251
    
    PRIMARY VARIABLES (Essential BCs / Dirichlet):
        READ(IN,*) NSPV
        READ(IN,*) ((ISPV(I,J), J=1,2), I=1,NSPV)
        READ(IN,*) (VSPV(I), I=1,NSPV)  [only if NEIGN == 0]
    
    SECONDARY VARIABLES (Natural BCs / Neumann):
        READ(IN,*) NSSV
        READ(IN,*) ((ISSV(I,J), J=1,2), I=1,NSSV)
        READ(IN,*) (VSSV(I), I=1,NSSV)  [only if NEIGN == 0]
    
    Attributes:
        ispv: Shape (nspv, 2) - [node_number, local_dof_number]
        vspv: Shape (nspv,) - specified values for primary DOFs
        issv: Shape (nssv, 2) - [node_number, local_dof_number]
        vssv: Shape (nssv,) - specified values for secondary DOFs
    """
    ispv: np.ndarray   # shape (nspv, 2), dtype int
    vspv: Optional[np.ndarray] = None   # shape (nspv,), dtype float
    issv: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 2))
    vssv: Optional[np.ndarray] = None   # shape (nssv,), dtype float


@dataclass
class ConvectionBC:
    """
    Fortran: Lines 269-272 (only if ICONV != 0)
    READ(IN,*) ICONV
    READ(IN,*) NBE
    READ(IN,*) (IBN(I), (INOD(I,J),J=1,2), BETA(I), TINF(I), I=1,NBE)
    
    Attributes:
        iconv: Convection flag (0 = no convection, > 0 = enabled)
        nbe: Number of boundary elements with convection
        ibn: Shape (nbe,) - element numbers with convection
        inod: Shape (nbe, 2) - local node pairs on convection boundary
        beta: Shape (nbe,) - convection coefficient (heat transfer)
        tinf: Shape (nbe,) - external/ambient temperature
    """
    iconv: int
    nbe: Optional[int] = None
    ibn: Optional[np.ndarray] = None    # shape (nbe,), dtype int
    inod: Optional[np.ndarray] = None   # shape (nbe, 2), dtype int
    beta: Optional[np.ndarray] = None   # shape (nbe,), dtype float
    tinf: Optional[np.ndarray] = None   # shape (nbe,), dtype float


@dataclass
class PoissonCoefficients:
    """
    Fortran: Lines 265-267 (only if ITYPE == 0)
    READ(IN,*) A10, A1X, A1Y
    READ(IN,*) A20, A2X, A2Y
    READ(IN,*) A00
    
    For Poisson/Laplace equation:
        -d/dx(A11*du/dx) - d/dy(A22*du/dy) + A00*u = f
    where:
        A11 = A10 + A1X*x + A1Y*y
        A22 = A20 + A2X*x + A2Y*y
    """
    a10: float
    a1x: float
    a1y: float
    a20: float
    a2x: float
    a2y: float
    a00: float


@dataclass
class ElasticityMaterial:
    """
    Fortran: Lines 292-335 (only if ITYPE >= 2)
    READ(IN,*) E1, E2, ANU12, G12, (G13, G23), THKNS
    
    Material properties for plane elasticity and plate bending.
    
    For 2D ELASTICITY:
        E1, E2 = Young's moduli in principal directions
        ANU12 = Poisson's ratio
        G12 = Shear modulus in x-y plane
        THKNS = Thickness
    
    For PLATE BENDING:
        E1, E2 = Young's moduli
        ANU12, G12 = As above
        G13, G23 = Shear moduli (for FSDT theory)
        THKNS = Plate thickness
    
    The code computes CMAT (3x3 material matrix) from these inputs:
        - Plane stress/strain for elasticity
        - Bending stiffness matrix for plates
    
    Attributes:
        e1, e2: Young's moduli
        anu12: Poisson's ratio
        g12: Shear modulus in plane
        g13, g23: Transverse shear moduli (plates only)
        thkns: Thickness
        lnstrs: (Elasticity only) 0 = plane strain, 1 = plane stress
    """
    e1: float
    e2: float
    anu12: float
    g12: float
    thkns: float
    lnstrs: Optional[int] = None  # For elasticity (plane strain/stress)
    g13: Optional[float] = None   # For plate bending FSDT
    g23: Optional[float] = None   # For plate bending FSDT


@dataclass
class SourceAndLoads:
    """
    Fortran: Line ~327 (after material, before time-dependent params)
    READ(IN,*) F0, FX, FY
    
    Body force / source term: f(x,y) = F0 + FX*x + FY*y
    
    Attributes:
        f0: Constant component
        fx: Linear x-component
        fy: Linear y-component
    """
    f0: float
    fx: float
    fy: float


@dataclass
class DynamicParameters:
    """
    Fortran: Lines 321-364+ (only if ITEM != 0)
    
    Damping/inertia coefficients and time-stepping parameters.
    
    Attributes:
        c0: Constant mass/damping parameter
        cx, cy: Spatial coupling parameters (adjusted internally by code)
        ntime: Total number of time steps
        nstp: Time step at which load is removed
        intvl: Print interval (every intvl-th step)
        intial: Initial condition flag (0 = zero, 1 = from file, 2 = custom)
        dt: Time increment
        alfa: Time integration parameter (Newmark-beta family)
        gama: Time integration parameter (Newmark-beta family)
        epsln: Convergence tolerance for steady-state detection
        initial_u: (Optional) Initial displacement vector, shape (neq,)
        initial_v: (Optional) Initial velocity vector, shape (neq,)
        initial_a: (Optional) Initial acceleration vector, shape (neq,)
    """
    c0: float
    cx: float
    cy: float
    ntime: int
    nstp: int
    intvl: int
    intial: int
    dt: float
    alfa: float
    gama: float
    epsln: float
    initial_u: Optional[np.ndarray] = None
    initial_v: Optional[np.ndarray] = None
    initial_a: Optional[np.ndarray] = None


@dataclass
class FEM2DConfig:
    """
    Complete problem configuration parsed from FEM2D input file.
    
    This is the top-level object returned by read_inp() containing all
    problem parameters in a structured, type-safe format.
    """
    title: str
    problem_type: ProblemType
    element_mesh: ElementMesh
    mesh_data: Optional[RectangularMesh] = None
    user_mesh: Optional[UserMesh] = None
    boundary_conditions: Optional[BoundaryConditions] = None
    convection: Optional[ConvectionBC] = None
    poisson_coeff: Optional[PoissonCoefficients] = None
    material: Optional[ElasticityMaterial] = None
    source_loads: Optional[SourceAndLoads] = None
    dynamic: Optional[DynamicParameters] = None
    
    # Derived quantities (computed)
    ndf: int = 1  # Degrees of freedom per node
    neq: int = 1  # Total equations (NNM * NDF)
    nn: int = 1   # DOF per element (NPE * NDF)


# ============================================================================
# PARSER FUNCTION
# ============================================================================

def read_inp(filepath: str) -> FEM2DConfig:
    """
    Parse FEM2D input file and return structured configuration.
    
    Implements exact read sequence from FEM2DF15.FOR lines 154-400.
    Handles all conditional branches (NEIGN, MESH, ITYPE, ITEM, ICONV).
    
    Args:
        filepath: Path to .INP file
        
    Returns:
        FEM2DConfig: Typed configuration object
        
    Raises:
        FileNotFoundError: If input file not found
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)   # type: ignore
    if not filepath.exists():   # type: ignore
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    # Tokenize entire file (whitespace-separated, skip comments)
    tokens = _tokenize_file(filepath)   # type: ignore
    idx = 0
    
    # ========== BLOCK 1: Title and Problem Type ==========
    # Fortran line 154: READ(IN,400) TITLE
    title = tokens[idx]
    idx += 1
    
    # Fortran line 158: READ(IN,*) ITYPE, IGRAD, ITEM, NEIGN
    itype = int(tokens[idx])
    igrad = int(tokens[idx + 1])
    item = int(tokens[idx + 2])
    neign = int(tokens[idx + 3])
    idx += 4
    if item == 0:
        neign = 0
    
    # Fortran line 165 (conditional on NEIGN != 0)
    nvalu = nvctr = None
    if neign != 0:
        nvalu = int(tokens[idx])
        nvctr = int(tokens[idx + 1])
        idx += 2
    
    problem_type = ProblemType(itype, igrad, item, neign, nvalu, nvctr)
    
    # ========== BLOCK 2: Element and Mesh Type ==========
    # Fortran line 171: READ(IN,*) IELTYP, NPE, MESH, NPRNT
    ieltyp = int(tokens[idx])
    npe = int(tokens[idx + 1])
    mesh = int(tokens[idx + 2])
    nprnt = int(tokens[idx + 3])
    idx += 4
    
    element_mesh = ElementMesh(ieltyp, npe, mesh, nprnt)
    
    # ========== BLOCK 3: Mesh Generation/Input ==========
    mesh_data = None
    user_mesh = None
    
    if mesh == 1:
        # Fortran lines 203-205: Rectangular domain
        # READ(IN,*) NX, NY
        nx = int(tokens[idx])
        ny = int(tokens[idx + 1])
        idx += 2
        
        # READ(IN,*) X0, (DX(I), I=1,NX)
        x0 = float(tokens[idx])
        dx = np.array([float(tokens[idx + 1 + i]) for i in range(nx)])
        idx += 1 + nx
        
        # READ(IN,*) Y0, (DY(I), I=1,NY)
        y0 = float(tokens[idx])
        dy = np.array([float(tokens[idx + 1 + i]) for i in range(ny)])
        idx += 1 + ny
        
        mesh_data = RectangularMesh(nx, ny, x0, y0, dx, dy)
        # MESH2DR will compute NEM, NNM internally
        element_mesh.nem = nx * ny if ieltyp != 0 else 2 * nx * ny
        element_mesh.nnm = (nx + 1) * (ny + 1) if ieltyp != 0 else (nx + 1) * (ny + 1)
        
    elif mesh == 0:
        # Fortran lines 182, 189-190: User-provided mesh
        # READ(IN,*) NEM, NNM
        nem = int(tokens[idx])
        nnm = int(tokens[idx + 1])
        idx += 2
        
        element_mesh.nem = nem
        element_mesh.nnm = nnm
        
        # READ(IN,*) (NOD(N,I), I=1,NPE)  [loop over NEM]
        nod = np.zeros((nem, npe), dtype=int)
        for n in range(nem):
            for i in range(npe):
                nod[n, i] = int(tokens[idx])
                idx += 1
        
        # READ(IN,*) ((GLXY(I,J), J=1,2), I=1,NNM)
        glxy = np.zeros((nnm, 2), dtype=float)
        for i in range(nnm):
            glxy[i, 0] = float(tokens[idx])
            glxy[i, 1] = float(tokens[idx + 1])
            idx += 2
        
        user_mesh = UserMesh(nod, glxy)
    
    else:  # mesh >= 2
        # Fortran line 1871: SUBROUTINE MESH2DG (general mesh generation)
        # This reads its own records, skip for now (TODO: implement MESH2DG parser)
        raise NotImplementedError("MESH2DG (general mesh generation) not yet implemented")
    
    # Compute NDF based on ITYPE
    if itype == 0:
        ndf = 1
    elif itype == 1:
        ndf = 2
    else:  # ITYPE >= 2 (elasticity, plate bending)
        ndf = 3 if itype in [3, 4] else (4 if itype == 5 else 2)
    
    # ========== BLOCK 4: Boundary Conditions ==========
    # Fortran line 240: READ(IN,*) NSPV
    nspv = int(tokens[idx])
    idx += 1
    
    ispv = np.array([], dtype=int).reshape(0, 2)
    vspv = None
    
    if nspv > 0:
        # Fortran line 242: READ(IN,*) ((ISPV(I,J), J=1,2), I=1,NSPV)
        ispv = np.zeros((nspv, 2), dtype=int)
        for i in range(nspv):
            ispv[i, 0] = int(tokens[idx])
            ispv[i, 1] = int(tokens[idx + 1])
            idx += 2
        
        # Fortran line 244: (only if NEIGN == 0)
        if neign == 0:
            vspv = np.array([float(tokens[idx + i]) for i in range(nspv)])
            idx += nspv
    
    issv = np.array([], dtype=int).reshape(0, 2)
    vssv = None

    if neign == 0:
        # Fortran line 248: READ(IN,*) NSSV
        nssv = int(tokens[idx])
        idx += 1
        
        if nssv > 0:
            # Fortran line 250: READ(IN,*) ((ISSV(I,J), J=1,2), I=1,NSSV)
            issv = np.zeros((nssv, 2), dtype=int)
            for i in range(nssv):
                issv[i, 0] = int(tokens[idx])
                issv[i, 1] = int(tokens[idx + 1])
                idx += 2
            
            # Fortran line 251
            vssv = np.array([float(tokens[idx + i]) for i in range(nssv)])
            idx += nssv
    
    boundary_conditions = BoundaryConditions(ispv, vspv, issv, vssv)
    
    # ========== BLOCK 5: Problem-Specific Coefficients ==========
    poisson_coeff = None
    material = None
    
    if itype == 0:
        # Fortran lines 265-267: Poisson coefficients
        a10 = float(tokens[idx])
        a1x = float(tokens[idx + 1])
        a1y = float(tokens[idx + 2])
        idx += 3
        
        a20 = float(tokens[idx])
        a2x = float(tokens[idx + 1])
        a2y = float(tokens[idx + 2])
        idx += 3
        
        a00 = float(tokens[idx])
        idx += 1
        
        poisson_coeff = PoissonCoefficients(a10, a1x, a1y, a20, a2x, a2y, a00)
        
    elif itype == 1:
        # Fortran lines ~277-278: Viscous flow
        # READ(IN,*) VISCSITY, PENALTY
        viscsity = float(tokens[idx])
        penalty = float(tokens[idx + 1])
        idx += 2
        # TODO: store in separate dataclass if needed
        
    else:  # ITYPE >= 2: Elasticity or Plate Bending
        if itype == 2:
            # Fortran line ~289: Plane stress/strain flag
            lnstrs = int(tokens[idx])
            idx += 1
            
            # Fortran line 292: READ(IN,*) E1, E2, ANU12, G12, THKNS
            e1 = float(tokens[idx])
            e2 = float(tokens[idx + 1])
            anu12 = float(tokens[idx + 2])
            g12 = float(tokens[idx + 3])
            thkns = float(tokens[idx + 4])
            idx += 5
            
            material = ElasticityMaterial(e1, e2, anu12, g12, thkns, lnstrs)
            
        else:  # ITYPE >= 3: Plate bending
            # Fortran line 292: READ(IN,*) E1, E2, ANU12, G12, G13, G23, THKNS
            e1 = float(tokens[idx])
            e2 = float(tokens[idx + 1])
            anu12 = float(tokens[idx + 2])
            g12 = float(tokens[idx + 3])
            g13 = float(tokens[idx + 4])
            g23 = float(tokens[idx + 5])
            thkns = float(tokens[idx + 6])
            idx += 7
            
            material = ElasticityMaterial(e1, e2, anu12, g12, thkns, g13=g13, g23=g23)
    
    # ========== BLOCK 6: Convection BC (optional; ITYPE==0 only) ==========
    convection = ConvectionBC(0)
    if itype == 0:
        # Fortran line 269: READ(IN,*) ICONV (inside ITYPE==0 branch)
        iconv = int(tokens[idx])
        idx += 1
        convection = ConvectionBC(iconv)

    if itype == 0 and convection.iconv != 0:
        # Fortran line 271: READ(IN,*) NBE
        nbe = int(tokens[idx])
        idx += 1
        
        ibn = np.zeros(nbe, dtype=int)
        inod = np.zeros((nbe, 2), dtype=int)
        beta = np.zeros(nbe, dtype=float)
        tinf = np.zeros(nbe, dtype=float)
        
        # Auto-detect format by trying to parse first record
        # Format B (grouped): IBN(float beta), BETA, TINF, ..., then INOD values
        # Format A (interleaved): IBN, INOD1, INOD2, BETA, TINF, ...
        
        def _int_from_token(tok: str) -> int:
            val = float(tok)
            if not np.isfinite(val) or abs(val - round(val)) > 1e-9:
                raise ValueError(f"Expected integer-valued token, got {tok!r}")
            return int(round(val))

        def _parse_convection_records(start_idx: int, mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
            loc_idx = start_idx
            ibn_arr = np.zeros(nbe, dtype=int)
            inod_arr = np.zeros((nbe, 2), dtype=int)
            beta_arr = np.zeros(nbe, dtype=float)
            tinf_arr = np.zeros(nbe, dtype=float)

            if mode == "interleaved":
                for i in range(nbe):
                    ibn_arr[i] = _int_from_token(tokens[loc_idx])
                    inod_arr[i, 0] = _int_from_token(tokens[loc_idx + 1])
                    inod_arr[i, 1] = _int_from_token(tokens[loc_idx + 2])
                    beta_arr[i] = float(tokens[loc_idx + 3])
                    tinf_arr[i] = float(tokens[loc_idx + 4])
                    loc_idx += 5
            elif mode == "grouped":
                for i in range(nbe):
                    ibn_arr[i] = _int_from_token(tokens[loc_idx])
                    beta_arr[i] = float(tokens[loc_idx + 1])
                    tinf_arr[i] = float(tokens[loc_idx + 2])
                    loc_idx += 3
                for i in range(nbe):
                    inod_arr[i, 0] = _int_from_token(tokens[loc_idx])
                    inod_arr[i, 1] = _int_from_token(tokens[loc_idx + 1])
                    loc_idx += 2
            else:
                raise ValueError(f"Unknown convection parse mode: {mode}")

            return ibn_arr, inod_arr, beta_arr, tinf_arr, loc_idx

        def _score_convection_parse(ibn_arr: np.ndarray, inod_arr: np.ndarray) -> int:
            score = 0
            if element_mesh.nem is not None:
                score += 2 if np.all((ibn_arr >= 1) & (ibn_arr <= element_mesh.nem)) else -2
            if element_mesh.nnm is not None:
                score += 3 if np.all((inod_arr >= 1) & (inod_arr <= element_mesh.nnm)) else -3
            return score

        candidates = []
        for mode in ("interleaved", "grouped"):
            try:
                parsed = _parse_convection_records(idx, mode)
                candidates.append((_score_convection_parse(parsed[0], parsed[1]), mode, parsed))
            except (ValueError, IndexError):
                continue

        if not candidates:
            raise ValueError("Unable to parse convection boundary records")

        # Prefer canonical Fortran interleaved format on ties.
        candidates.sort(key=lambda item: (item[0], item[1] == "interleaved"), reverse=True)
        _, _, best = candidates[0]
        ibn, inod, beta, tinf, idx = best
        
        convection.nbe = nbe
        convection.ibn = ibn
        convection.inod = inod
        convection.beta = beta
        convection.tinf = tinf
    
    # ========== BLOCK 7: Source/Load Terms ==========
    # Fortran reads source terms only when NEIGN == 0.
    source_loads = None
    if neign == 0:
        f0 = float(tokens[idx])
        fx = float(tokens[idx + 1])
        fy = float(tokens[idx + 2])
        idx += 3
        source_loads = SourceAndLoads(f0, fx, fy)
    
    # ========== BLOCK 8: Dynamic/Transient Parameters (if ITEM != 0) ==========
    dynamic = None
    
    if item != 0:
        # Fortran line ~321: READ(IN,*) C0, CX, CY
        c0 = float(tokens[idx])
        cx = float(tokens[idx + 1])
        cy = float(tokens[idx + 2])
        idx += 3

        # Match Fortran scaling of dynamic coefficients for solid/plate problems.
        if material is not None and itype > 1:
            thkns = material.thkns
            if itype == 2:
                c0 = thkns * c0
                cx = thkns * cx
                cy = thkns * cy
            else:
                if neign <= 1:
                    c0 = thkns * c0
                    cx = (thkns ** 2) * c0 / 12.0
                    cy = cx
        
        # Fortran does not read time-marching parameters for eigen runs.
        if neign != 0:
            dynamic = DynamicParameters(
                c0=c0, cx=cx, cy=cy,
                ntime=0, nstp=0, intvl=1, intial=0,
                dt=0.0, alfa=0.0, gama=0.0, epsln=0.0,
                initial_u=None, initial_v=None, initial_a=None,
            )
        else:
            # Fortran line ~332: READ(IN,*) NTIME, NSTP, INTVL, INTIAL
            ntime = int(tokens[idx])
            nstp = int(tokens[idx + 1])
            intvl = int(tokens[idx + 2])
            intial = int(tokens[idx + 3])
            idx += 4
            
            # Fortran line ~334: READ(IN,*) DT, ALFA, GAMA, EPSLN
            dt = float(tokens[idx])
            alfa = float(tokens[idx + 1])
            gama = float(tokens[idx + 2])
            epsln = float(tokens[idx + 3])
            idx += 4
            
            # Optional: Initial conditions (if INTIAL != 0 and ITEM in [1, 2])
            initial_u = None
            initial_v = None
            initial_a = None
            
            if intial != 0:
                neq = element_mesh.nnm * ndf
                
                # Fortran line ~339: READ(IN,*) (GLU(I), I=1,NEQ)
                initial_u = np.array([float(tokens[idx + i]) for i in range(neq)])
                idx += neq
                
                if item == 2:
                    # Fortran line (later): Initial velocity
                    initial_v = np.array([float(tokens[idx + i]) for i in range(neq)])
                    idx += neq
            
            dynamic = DynamicParameters(c0, cx, cy, ntime, nstp, intvl, intial, 
                                         dt, alfa, gama, epsln, initial_u, initial_v, initial_a)
    
    # ========== Assemble Final Configuration ==========
    neq = element_mesh.nnm * ndf if element_mesh.nnm else 0
    nn = npe * ndf
    
    config = FEM2DConfig(
        title=title,
        problem_type=problem_type,
        element_mesh=element_mesh,
        mesh_data=mesh_data,
        user_mesh=user_mesh,
        boundary_conditions=boundary_conditions,
        convection=convection,
        poisson_coeff=poisson_coeff,
        material=material,
        source_loads=source_loads,
        dynamic=dynamic,
        ndf=ndf,
        neq=neq,
        nn=nn
    )
    
    return config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _tokenize_file(filepath: Path) -> List[str]:
    """
    Tokenize input file: split by whitespace, remove comments.
    
    Handles:
    - Fortran fixed-form comments (C in column 1)
    - Inline comments (text beyond 60+ columns or after multiple spaces + letters)
    - Python-style comments (#)
    
    Args:
        filepath: Path to .INP file
        
    Returns:
        List of tokens (strings)
    """
    tokens = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # First line is the title
    if lines:
        title_line = lines[0].strip()
        if title_line and not title_line.startswith('C'):
            tokens.append(title_line)
        lines = lines[1:]
    
    # Process remaining lines
    for line in lines:
        # Remove Fortran fixed-form comments (C in column 1)
        if line.startswith('C'):
            continue
        
        # Remove Python-style comments
        if '#' in line:
            line = line[:line.index('#')]
        
        # Remove inline comments: if there are 4+ consecutive spaces followed by 
        # uppercase letters (typical Fortran comment style), strip that part
        import re
        # Look for pattern: multiple spaces + word characters (comment label)
        line = re.sub(r'\s{4,}[A-Za-z].*$', '', line)
        
        # Split by whitespace and add non-empty tokens
        tokens.extend(line.split())
    
    return tokens


def validate_config(config: FEM2DConfig) -> Tuple[bool, List[str]]:
    """
    Perform basic validation of parsed configuration.
    
    Returns:
        (is_valid, error_messages): Tuple of validation result and any errors found
    """
    errors = []
    
    # Check problem type
    if config.problem_type.itype not in [0, 1, 2, 3, 4, 5]:
        errors.append(f"Invalid ITYPE: {config.problem_type.itype}")
    
    # Check element type (0=triangular, >0=quadrilateral variants)
    if config.element_mesh.ieltyp < 0:
        errors.append(f"Invalid IELTYP: {config.element_mesh.ieltyp}")
    
    # Check NPE
    if config.element_mesh.npe not in [3, 4, 6, 8, 9]:
        errors.append(f"Invalid NPE: {config.element_mesh.npe}")
    
    # Check mesh consistency
    if config.problem_type.itype == 3 and config.element_mesh.ieltyp == 0:
        errors.append("Triangular elements not allowed for plate bending (ITYPE=3)")
    
    # Check mesh data
    if config.mesh_data is None and config.user_mesh is None:
        errors.append("Neither mesh_data nor user_mesh is provided")
    
    if config.user_mesh is not None:
        if config.user_mesh.nod.shape[0] != config.element_mesh.nem:
            errors.append(f"NOD shape mismatch: got {config.user_mesh.nod.shape[0]}, expected {config.element_mesh.nem}")
        if config.user_mesh.glxy.shape[0] != config.element_mesh.nnm:
            errors.append(f"GLXY shape mismatch: got {config.user_mesh.glxy.shape[0]}, expected {config.element_mesh.nnm}")
    
    return len(errors) == 0, errors


# ============================================================================
# Main entry point for testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python io.py <input_file.INP>")
        sys.exit(1)
    
    try:
        config = read_inp(sys.argv[1])
        print(f"✓ Successfully parsed: {config.title}")
        print(f"  Problem type: {config.problem_type.itype}")
        print(f"  Elements: {config.element_mesh.nem}, Nodes: {config.element_mesh.nnm}")
        print(f"  DOF per node: {config.ndf}, Total equations: {config.neq}")
        
        is_valid, errors = validate_config(config)
        if is_valid:
            print("✓ Validation passed")
        else:
            print("✗ Validation errors:")
            for err in errors:
                print(f"  - {err}")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
