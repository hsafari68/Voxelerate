from pathlib import Path
import struct

import numpy as np

from voxelerate import load_mesh


CUBE_OBJ = """v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""

ASCII_STL = """solid triangle
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid triangle
"""


def test_load_obj_centers_geometry(tmp_path: Path) -> None:
    path = tmp_path / "cube.obj"
    path.write_text(CUBE_OBJ)

    mesh = load_mesh(path, center=True)

    bounds_min, bounds_max = mesh.bounds
    center = (bounds_min + bounds_max) * 0.5

    np.testing.assert_allclose(center, np.zeros(3, dtype=np.float32), atol=1e-6)
    assert mesh.faces.shape == (12, 3)
    assert mesh.vertex_normals.shape == mesh.vertices.shape
    assert mesh.triangle_count == 12


def test_load_ascii_stl(tmp_path: Path) -> None:
    path = tmp_path / "triangle.stl"
    path.write_text(ASCII_STL)

    mesh = load_mesh(path)

    assert mesh.triangle_count == 1
    np.testing.assert_allclose(mesh.bounds[0], np.array([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(mesh.bounds[1], np.array([1.0, 1.0, 0.0], dtype=np.float32))


def test_load_binary_stl(tmp_path: Path) -> None:
    path = tmp_path / "triangle_binary.stl"
    with path.open("wb") as handle:
        handle.write(b"binary stl".ljust(80, b" "))
        handle.write(struct.pack("<I", 1))
        record = struct.pack(
            "<12fH",
            0.0, 0.0, 1.0,  # normal
            0.0, 0.0, 0.0,  # v0
            1.0, 0.0, 0.0,  # v1
            0.0, 1.0, 0.0,  # v2
            0,              # attribute byte count
        )
        handle.write(record)

    mesh = load_mesh(path)

    assert mesh.triangle_count == 1
    np.testing.assert_allclose(mesh.bounds[0], np.array([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(mesh.bounds[1], np.array([1.0, 1.0, 0.0], dtype=np.float32))
