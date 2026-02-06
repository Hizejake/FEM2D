File Structure: 

Python/
├─ pyproject.toml
├─ README.md
├─ src/
│  └─ fem2d/
│     ├─ __init__.py
│     ├─ mesh.py              # mesh IO + data structures (Node, Element, Mesh)
│     ├─ elements.py          # element classes (base + implementations)
│     ├─ material.py          # material models (elastic, plastic stubs)
│     ├─ assemble.py          # element -> global matrix/vector assembly
│     ├─ boundary.py          # BC application (Dirichlet, Neumann)
│     ├─ solver.py            # linear solvers + wrappers (scipy, sparse)
│     ├─ post.py              # postprocessing (contours, exports)
│     ├─ io.py                # .inp parser + meshio helpers
│     └─ utils.py             # common helpers, types, constants
├─ examples/
│  └─ example_plate.py
└─ tests/
   ├─ test_elements.py
   ├─ test_assembly.py
   └─ test_integration.py