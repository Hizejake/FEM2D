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
    inp_path = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '2DEXMPL.INP'))
    cfg = read_inp(inp_path)
    mesh = generate_mesh_from_config(cfg)
    # For 2DEXMPL.INP NX=2,NY=2 with ieltyp=0 gave NEM=8 and NNM=9 earlier
    assert mesh.glxy.shape[0] == cfg.element_mesh.nnm
    assert mesh.nod.shape[0] == cfg.element_mesh.nem


def test_convective_mesh_example():
    inp_path = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'FEM2DCNV.INP'))
    cfg = read_inp(inp_path)
    mesh = generate_mesh_from_config(cfg)
    assert mesh.glxy.shape[0] == cfg.element_mesh.nnm
    assert mesh.nod.shape[0] == cfg.element_mesh.nem


def test_membrane_mesh_example():
    inp_path = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'membrane.inp'))
    cfg = read_inp(inp_path)
    mesh = generate_mesh_from_config(cfg)
    assert mesh.glxy.shape[0] == cfg.element_mesh.nnm
    assert mesh.nod.shape[0] == cfg.element_mesh.nem
