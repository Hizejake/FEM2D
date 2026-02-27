import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src import driver


def test_plate_bending_dynamic_runs_from_inp():
    inp_path = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'inputs', 'rect_plate_dynamic_load.inp'))
    out_path = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'inputs', 'rect_plate_dynamic_load_python_output_test.txt'))
    u, field, cfg = driver.solve_from_inp(inp_path, output_txt_path=out_path)
    assert cfg.problem_type.itype == 3
    assert u.size == cfg.neq == 27
    assert field.shape == (9, 3)
    # Center deflection should be finite and positive for this load case.
    assert np.isfinite(field[4, 0])
    assert abs(field[4, 0]) > 0.0
    assert os.path.exists(out_path)
    txt = open(out_path, "r", encoding="utf-8").read()
    assert "Time Step Number" in txt
    assert "deflec. w" in txt
