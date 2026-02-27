"""
Run Python and Fortran solvers on the same INP and print a compact comparison.
"""
from pathlib import Path
import sys

from plate_vibration_python import run_python_case
from plate_vibration_fortran_runner import run_fortran_case


def _extract_fortran_center_w(log_path: Path) -> float | None:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    time_idxs = [i for i, line in enumerate(lines) if "Time Step Number" in line]
    if not time_idxs:
        return None
    start = time_idxs[-1]
    for line in lines[start:start + 120]:
        parts = line.split()
        if len(parts) >= 6:
            try:
                node = int(parts[0])
                if node == 5:
                    return float(parts[3].replace("D", "E"))
            except ValueError:
                continue
    return None


def _extract_python_center_w(txt_path: Path) -> float | None:
    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) >= 6:
            try:
                node = int(parts[0])
                if node == 5:
                    return float(parts[3])
            except ValueError:
                continue
    return None


def main(inp_file: str | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    inp = Path(inp_file) if inp_file else (repo_root / "inputs" / "rect_plate_dynamic.inp")

    py_out = run_python_case(str(inp))
    ft_out = run_fortran_case(str(inp))

    py_w = _extract_python_center_w(py_out)
    ft_w = _extract_fortran_center_w(ft_out)

    print("\nComparison (node 5, final time step):")
    print(f"  Python  w: {py_w}")
    print(f"  Fortran w: {ft_w}")
    if py_w is not None and ft_w is not None and ft_w != 0.0:
        rel = abs(py_w - ft_w) / abs(ft_w)
        print(f"  Relative error: {rel:.6e}")

    print("\nOutput files:")
    print(f"  Python : {py_out}")
    print(f"  Fortran: {ft_out}")
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(main(arg))
