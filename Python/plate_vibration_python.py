"""
Run the Python FEM2D solver from an INP file and write a Fortran-style TXT.
"""
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "Python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Python"))

from src.driver import solve_from_inp


def run_python_case(input_file: str | None = None, output_file: str | None = None) -> Path:
    inp = Path(input_file) if input_file else (REPO_ROOT / "inputs" / "rect_plate_dynamic.inp")
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    out = Path(output_file) if output_file else (REPO_ROOT / "outputs" / f"{inp.stem}_python.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    u, field, cfg = solve_from_inp(str(inp), output_txt_path=str(out))
    print(f"Python solve complete: {cfg.title}")
    print(f"Input : {inp}")
    print(f"Output: {out}")
    print(f"Center deflection (node 5, w): {field[4,0]:.8e}")
    return out


if __name__ == "__main__":
    arg_inp = sys.argv[1] if len(sys.argv) > 1 else None
    arg_out = sys.argv[2] if len(sys.argv) > 2 else None
    run_python_case(arg_inp, arg_out)
