"""
Dense solver utilities for FEM2D.

This module provides simple, well-documented dense linear-system solvers suitable
for small- to medium-sized FEM problems. The Fortran original uses a banded
solver tuned for memory efficiency; here we implement a dense (full-matrix)
approach using NumPy/LAPACK for clarity and ease of testing.

Design notes and trade-offs (explained in comments):
- Dense assembly + dense solve is simple and robust but uses O(N^2) memory and
  O(N^3) time, where N = number of DOFs. This is fine for small problems used
  during development and testing.
- NumPy's `linalg.solve` calls optimized LAPACK routines (LU factorization with
  partial pivoting) which are stable for general systems; pivoting is handled
  internally and we do not need to reimplement it.
- When the stiffness matrix is symmetric positive definite (SPD) one can use
  Cholesky (e.g., `np.linalg.cholesky`) for better performance and numerical
  stability; here we use `linalg.solve` for generality but provide a
  convenience `solve_spd` wrapper.
- The functions expect Dirichlet BCs to have been applied already (rows/cols
  modified and RHS adjusted). Use `boundary.apply_dirichlet_bc` before calling
  these solvers.

API:
- `solve_dense(K, F)` -> returns solution vector `u` for K u = F
- `solve_spd(K, F)` -> Cholesky-based solver for symmetric positive-definite K
- `residual_norm(K, u, F)` -> computes ||K u - F||_2 for verification
"""
from typing import Tuple
import numpy as np


def solve_dense(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Solve a linear system K u = F using NumPy's dense solver.

    Expects K to be a square 2D array and F the right-hand side vector.
    Returns the solution vector `u`.

    Implementation details (why/how):
    - Uses `np.linalg.solve`, which wraps optimized LAPACK routines (LU
      factorization with partial pivoting). This provides good numerical
      stability for general (non-symmetric / indefinite) matrices.
    - We intentionally do not modify the inputs; copies are made by
      NumPy/LAPACK when necessary.
    - Caller should ensure Dirichlet BCs are enforced before calling this
      function (i.e., fixed DOFs have been eliminated or rows/columns
      properly modified).
    """
    K = np.asarray(K, dtype=float)
    F = np.asarray(F, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square 2D array")
    if F.ndim != 1 or F.shape[0] != K.shape[0]:
        raise ValueError("F must be a 1D vector with compatible length")

    # Basic singularity/conditioning check: warn if matrix is ill-conditioned.
    # We do not fail automatically on large condition numbers because some
    # nearly-singular systems are intentionally used (e.g., under-constrained
    # before BC application). Caller is responsible for ensuring well-posedness.
    cond = np.linalg.cond(K)
    if cond > 1e12:
        # large condition number — numerical results may be unreliable
        # (do not raise, just note in comments/logging in more advanced code)
        pass

    u = np.linalg.solve(K, F)
    return u


def solve_spd(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Solve for symmetric positive-definite (SPD) K using Cholesky.

    Cholesky decomposition is more efficient and numerically stable for SPD
    matrices compared with general LU decomposition. This wrapper performs a
    simple check for symmetry and then uses `np.linalg.cholesky`.
    """
    K = np.asarray(K, dtype=float)
    F = np.asarray(F, dtype=float)
    if not np.allclose(K, K.T, atol=1e-12):
        raise ValueError("K is not symmetric; use solve_dense for general matrices")
    L = np.linalg.cholesky(K)
    # Solve L y = F
    y = np.linalg.solve(L, F)
    # Solve L^T u = y
    u = np.linalg.solve(L.T, y)
    return u


def residual_norm(K: np.ndarray, u: np.ndarray, F: np.ndarray) -> float:
    """Return the 2-norm of the residual r = K u - F."""
    r = K @ u - F
    return float(np.linalg.norm(r))
