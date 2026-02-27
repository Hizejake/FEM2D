"""
Run the reference Fortran FEM2D solver for a given INP file.
"""
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
FORTRAN_EXE = REPO_ROOT / "fortran_reference" / "fem2d.exe"


def run_fortran_case(input_file: str | None = None, output_file: str | None = None) -> Path:
    inp = Path(input_file) if input_file else (REPO_ROOT / "inputs" / "rect_plate_dynamic.inp")
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")
    if not FORTRAN_EXE.exists():
        raise FileNotFoundError(f"Fortran executable not found: {FORTRAN_EXE}")

    out = Path(output_file) if output_file else (REPO_ROOT / "outputs" / f"{inp.stem}_fortran.log")
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(inp, "r", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
        result = subprocess.run([str(FORTRAN_EXE)], stdin=f_in, stdout=f_out, stderr=subprocess.STDOUT, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Fortran solver failed with return code {result.returncode}. See {out}")

    print(f"Fortran solve complete")
    print(f"Input : {inp}")
    print(f"Output: {out}")
    return out


if __name__ == "__main__":
    arg_inp = sys.argv[1] if len(sys.argv) > 1 else None
    arg_out = sys.argv[2] if len(sys.argv) > 2 else None
    run_fortran_case(arg_inp, arg_out)
