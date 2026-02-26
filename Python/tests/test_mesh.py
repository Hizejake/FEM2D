import sys
import os
import numpy as np

# Ensure local `src` package is importable when running tests from repository root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Import as package `src` to allow relative imports inside modules
import src.io as fem_io
import src.mesh as fem_mesh

read_inp = fem_io.read_inp
generate_mesh_from_config = fem_mesh.generate_mesh_from_config


def test_rectangular_mesh_2d_example():
    inp_path = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'inputs', '2DEXMPL.INP'))
    cfg = read_inp(inp_path)
    mesh = generate_mesh_from_config(cfg)
    # For 2DEXMPL.INP NX=2,NY=2 with ieltyp=0 gave NEM=8 and NNM=9 earlier
    assert mesh.glxy.shape[0] == cfg.element_mesh.nnm
    assert mesh.nod.shape[0] == cfg.element_mesh.nem


def test_convective_mesh_example():
    inp_path = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'inputs', 'FEM2DCNV.INP'))
    cfg = read_inp(inp_path)
    mesh = generate_mesh_from_config(cfg)
    assert mesh.glxy.shape[0] == cfg.element_mesh.nnm
    assert mesh.nod.shape[0] == cfg.element_mesh.nem


def test_membrane_mesh_example():
    inp_path = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'inputs', 'membrane.inp'))
    cfg = read_inp(inp_path)
    mesh = generate_mesh_from_config(cfg)
    assert mesh.glxy.shape[0] == cfg.element_mesh.nnm
    assert mesh.nod.shape[0] == cfg.element_mesh.nem


def test_convection_grouped_integer_tokens(tmp_path):
    content = """GroupedConvection
0 0 0 0
1 4 1 0
1 1
0.0 1.0
0.0 1.0
0
0
1.0 0.0 0.0
1.0 0.0 0.0
0.0
1
2
1 20 30
1 40 50
1 2
2 3
0.0 0.0 0.0
"""
    p = tmp_path / "grouped_conv.inp"
    p.write_text(content)
    cfg = read_inp(str(p))
    assert cfg.convection is not None
    assert cfg.convection.nbe == 2
    assert np.array_equal(cfg.convection.ibn, np.array([1, 1]))
    assert np.array_equal(cfg.convection.inod, np.array([[1, 2], [2, 3]]))
    assert np.allclose(cfg.convection.beta, np.array([20.0, 40.0]))
    assert np.allclose(cfg.convection.tinf, np.array([30.0, 50.0]))
