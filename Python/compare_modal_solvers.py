"""
Run and compare Fortran vs Python modal (natural frequency) outputs.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

from plate_modal_python import solve_modal_from_inp
from plate_vibration_fortran_runner import run_fortran_case


_FT_RE = re.compile(
    r"Eigenvalue\(\s*(\d+)\s*\)\s*=\s*([+-]?\d*\.?\d+(?:[EDed][+-]?\d+)?)\s*Frequency\s*=\s*([+-]?\d*\.?\d+(?:[EDed][+-]?\d+)?)"
)


def _parse_fortran_modal(log_path: Path) -> list[tuple[int, float, float]]:
    out: list[tuple[int, float, float]] = []
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    for m in _FT_RE.finditer(txt):
        mode = int(m.group(1))
        lam = float(m.group(2).replace("D", "E").replace("d", "E"))
        frq = float(m.group(3).replace("D", "E").replace("d", "E"))
        out.append((mode, lam, frq))
    return out


def _parse_python_modal(txt_path: Path) -> list[tuple[int, float, float]]:
    out: list[tuple[int, float, float]] = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            mode = int(parts[0])
            lam = float(parts[1])
            frq = float(parts[2]) if len(parts) >= 3 else float("nan")
            out.append((mode, lam, frq))
    return out


def main(inp_file: str) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    inp = Path(inp_file)
    if not inp.is_absolute():
        inp = (repo_root / inp).resolve()

    py_out = solve_modal_from_inp(str(inp), str(repo_root / "outputs" / f"{inp.stem}_python_modal.txt"))
    ft_out = run_fortran_case(str(inp), str(repo_root / "outputs" / f"{inp.stem}_fortran_modal.log"))

    py_modes = sorted(_parse_python_modal(py_out), key=lambda x: x[1])
    ft_modes = sorted(_parse_fortran_modal(ft_out), key=lambda x: x[1])
    n = min(len(py_modes), len(ft_modes))

    print("\nModal comparison (sorted by eigenvalue, first modes):")
    for i in range(n):
        _, lam_py, f_py = py_modes[i]
        _, lam_ft, f_ft = ft_modes[i]
        rel_lam = abs(lam_py - lam_ft) / (abs(lam_ft) if lam_ft != 0.0 else 1.0)
        rel_f = abs(f_py - f_ft) / (abs(f_ft) if f_ft != 0.0 else 1.0)
        print(f"  mode {i+1:2d}: rel(eig)={rel_lam:.6e}, rel(freq)={rel_f:.6e}")

    print("\nOutput files:")
    print(f"  Python : {py_out}")
    print(f"  Fortran: {ft_out}")
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is None:
        raise SystemExit("Usage: python compare_modal_solvers.py <input.inp>")
    raise SystemExit(main(arg))
