"""
Run modal (natural frequency) analysis from a FEM2D INP file in Python.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "Python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Python"))

from src.io import read_inp
from src.mesh import generate_mesh_from_config
from src.material import compute_cmat_from_config
from src.assemble import assemble_global, assemble_mass_global
from src.boundary import _map_dof
from src import elements

try:
    from scipy.linalg import eigh  # type: ignore
except Exception:  # pragma: no cover
    eigh = None


def solve_modal_from_inp(input_file: str, output_file: str | None = None) -> Path:
    cfg = read_inp(input_file)
    if cfg.problem_type.neign == 0:
        raise ValueError("Input is not an eigenvalue problem (NEIGN must be non-zero).")

    mesh = generate_mesh_from_config(cfg)

    material = None
    if cfg.material is not None and cfg.problem_type.itype >= 2:
        cm = compute_cmat_from_config(
            cfg.problem_type.itype,
            cfg.material.lnstrs,
            cfg.material.e1,
            cfg.material.e2,
            cfg.material.anu12,
            cfg.material.g12,
            cfg.material.thkns,
            cfg.material.g13,
            cfg.material.g23,
        )
        if cfg.problem_type.itype in (3, 4):
            material = {"CMAT": cm.cmat, "C44": cm.c44 or 0.0, "C55": cm.c55 or 0.0}
        else:
            material = cm.cmat

    K, _ = assemble_global(
        mesh.nod,
        mesh.glxy,
        cfg.element_mesh.npe,
        cfg.ndf,
        elements.element_matrices,
        cfg.problem_type.itype,
        material=material,
        coeffs={},
    )

    mass_func = elements.triangle_element_mass if cfg.element_mesh.npe == 3 else elements.quad_element_mass
    dyn = {
        "C0": cfg.dynamic.c0 if cfg.dynamic else 1.0,
        "CX": cfg.dynamic.cx if cfg.dynamic else 0.0,
        "CY": cfg.dynamic.cy if cfg.dynamic else 0.0,
        "NEIGN": cfg.problem_type.neign,
    }
    M = assemble_mass_global(
        mesh.nod,
        mesh.glxy,
        cfg.element_mesh.npe,
        cfg.ndf,
        mass_func,
        itype=cfg.problem_type.itype,
        coeffs={},
        dyn=dyn,
    )

    fixed = set()
    if cfg.boundary_conditions is not None:
        for i in range(cfg.boundary_conditions.ispv.shape[0]):
            fixed.add(_map_dof(cfg.boundary_conditions.ispv[i, 0], cfg.boundary_conditions.ispv[i, 1], cfg.ndf))
    all_dofs = np.arange(K.shape[0], dtype=int)
    free = np.array([d for d in all_dofs if d not in fixed], dtype=int)
    if free.size == 0:
        raise ValueError("No free DOFs after applying boundary conditions.")

    Kff = K[np.ix_(free, free)]
    Mff = M[np.ix_(free, free)]

    # Fortran EGNSOLVR/JACOBI workflow behaves like a mass-lumped
    # generalized eigen solve for these plate vibration runs.
    # Use diagonalized mass to stay consistent with reference output.
    Mff = np.diag(np.diag(Mff))

    if eigh is not None:
        vals, _ = eigh(Kff, Mff)
    else:  # fallback
        vals, _ = np.linalg.eig(np.linalg.solve(Mff, Kff))
        vals = np.real(vals)

    vals = np.real(vals)
    vals = vals[vals > 1e-12]
    vals.sort()

    nvalu = cfg.problem_type.nvalu if cfg.problem_type.nvalu is not None else min(10, vals.size)
    nvalu = min(nvalu, vals.size)
    # Match Fortran EGNSOLVR/JACOBI reporting convention (largest modes first).
    eigvals = vals[-nvalu:][::-1]
    freqs = np.sqrt(eigvals) if cfg.problem_type.item >= 2 and cfg.problem_type.neign == 1 else np.full_like(eigvals, np.nan)

    out = Path(output_file) if output_file else (REPO_ROOT / "outputs" / f"{Path(input_file).stem}_python_modal.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"{cfg.title}\n")
        f.write("MODE    EIGENVALUE          FREQUENCY\n")
        for i, lam in enumerate(eigvals, start=1):
            frq = freqs[i - 1]
            if np.isfinite(frq):
                f.write(f"{i:4d}  {lam: .10e}  {frq: .10e}\n")
            else:
                f.write(f"{i:4d}  {lam: .10e}\n")

    print(f"Modal solve complete: {cfg.title}")
    print(f"Input : {input_file}")
    print(f"Output: {out}")
    if eigvals.size:
        print(f"Mode 1 eigenvalue: {eigvals[0]:.8e}")
        if np.isfinite(freqs[0]):
            print(f"Mode 1 frequency : {freqs[0]:.8e}")
    return out


if __name__ == "__main__":
    arg_inp = sys.argv[1] if len(sys.argv) > 1 else None
    arg_out = sys.argv[2] if len(sys.argv) > 2 else None
    if arg_inp is None:
        raise SystemExit("Usage: python plate_modal_python.py <input.inp> [output.txt]")
    solve_modal_from_inp(arg_inp, arg_out)
