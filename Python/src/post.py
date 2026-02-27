"""
Postprocessing helpers for FEM2D.

Provides utilities to convert solution vectors into nodal fields, compute
simple derived quantities, and write Fortran-style text output.
"""
from typing import Any, Optional
import numpy as np


def extract_nodal_field(u: np.ndarray, ndf: int) -> np.ndarray:
    """Convert global DOF vector `u` into nodal field array of shape (nnm, ndf).

    For scalar problems (`ndf==1`) returns a (nnm,1) array. For vector problems
    returns (nnm, ndf) with ordering consistent with global DOF mapping.
    """
    u = np.asarray(u, dtype=float)
    if ndf == 1:
        return u.reshape((-1, 1))
    if u.size % ndf != 0:
        raise ValueError("Length of u is not divisible by ndf")
    nnm = u.size // ndf
    return u.reshape((nnm, ndf))


def max_abs_field(u: np.ndarray, ndf: int) -> float:
    """Return the maximum absolute value across all DOFs."""
    field = extract_nodal_field(u, ndf)
    return float(np.max(np.abs(field)))


def write_fortran_style_output_txt(
    output_path: str,
    cfg: Any,
    glxy: np.ndarray,
    field: np.ndarray,
    time_value: Optional[float] = None,
    time_step: Optional[int] = None,
) -> None:
    """Write a Fortran-like text output file for easier side-by-side comparison."""
    glxy = np.asarray(glxy, dtype=float)
    field = np.asarray(field, dtype=float)
    ndf = int(field.shape[1])

    if ndf == 1:
        cols = "    Node    x-coord.      y-coord.      Value of u"
    elif ndf == 2:
        cols = "    Node    x-coord.      y-coord.     Value of u     Value of v"
    else:
        cols = "    Node    x-coord.      y-coord.     deflec. w    x-rotation    y-rotation"

    line = "  _____________________________________________________________________________"
    lines = []
    lines.append(str(cfg.title))
    lines.append(line)
    lines.append("")
    if time_value is not None and time_step is not None:
        lines.append(f"     *TIME* = {time_value:0.5E}     Time Step Number = {time_step:2d}")
        lines.append("")
        lines.append("     S O L U T I O N :")
        lines.append("")
    lines.append(line)
    lines.append("")
    lines.append(cols)
    lines.append(line)
    lines.append("")

    for i in range(glxy.shape[0]):
        node = i + 1
        x = glxy[i, 0]
        y = glxy[i, 1]
        if ndf == 1:
            lines.append(f"{node:8d}   {x:0.5E}   {y:0.5E}   {field[i,0]:0.5E}")
        elif ndf == 2:
            lines.append(
                f"{node:8d}   {x:0.5E}   {y:0.5E}   {field[i,0]:0.5E}   {field[i,1]:0.5E}"
            )
        else:
            lines.append(
                f"{node:8d}   {x:0.5E}   {y:0.5E}   {field[i,0]:0.5E}   {field[i,1]:0.5E}   {field[i,2]:0.5E}"
            )
    lines.append(line)
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
