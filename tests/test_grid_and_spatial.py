from pathlib import Path

import numpy as np

from voxelerate import (
    VoxelGrid,
    build_bvh,
    build_mesh_octree,
    build_voxel_octree,
    build_voxelization_grid,
    load_mesh,
)


TETRA_OBJ = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""


def test_build_voxelization_grid_from_mesh(tmp_path: Path) -> None:
    path = tmp_path / "tetra.obj"
    path.write_text(TETRA_OBJ)
    mesh = load_mesh(path)

    grid = build_voxelization_grid(mesh, voxel_size=0.25)

    assert tuple(grid.grid_size_xyz.tolist()) == (4, 4, 4)
    assert tuple(grid.shape_zyx) == (4, 4, 4)
    np.testing.assert_allclose(grid.bounds_min_xyz, np.array([0.0, 0.0, 0.0], dtype=np.float32))


def test_build_voxelization_grid_from_explicit_bounds_and_grid_size(tmp_path: Path) -> None:
    path = tmp_path / "tetra.obj"
    path.write_text(TETRA_OBJ)
    mesh = load_mesh(path)

    grid = build_voxelization_grid(
        mesh,
        bounds_min_xyz=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        bounds_max_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        grid_size_xyz=np.array([8, 8, 8], dtype=np.int32),
        pixel_size=0.01,
    )

    assert tuple(grid.grid_size_xyz.tolist()) == (8, 8, 8)
    np.testing.assert_allclose(grid.voxel_size_xyz, np.array([0.25, 0.25, 0.25], dtype=np.float32))
    assert grid.pixel_size == 0.01


def test_build_voxelization_grid_from_component_transform(tmp_path: Path) -> None:
    path = tmp_path / "tetra.obj"
    path.write_text(TETRA_OBJ)
    mesh = load_mesh(path)

    grid = build_voxelization_grid(
        mesh,
        voxel_size=0.5,
        translation_xyz=np.array([2.0, -1.0, 0.5], dtype=np.float32),
        rotation_degrees_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        scale_xyz=2.0,
    )

    np.testing.assert_allclose(grid.bounds_min_xyz, np.array([2.0, -1.0, 0.5], dtype=np.float32))
    np.testing.assert_allclose(grid.bounds_max_xyz, np.array([4.0, 1.0, 2.5], dtype=np.float32))
    assert tuple(grid.grid_size_xyz.tolist()) == (4, 4, 4)


def test_bvh_and_octrees_build(tmp_path: Path) -> None:
    path = tmp_path / "tetra.obj"
    path.write_text(TETRA_OBJ)
    mesh = load_mesh(path)

    bvh = build_bvh(mesh, strategy="sah")
    assert bvh.root is not None
    assert len(bvh.collect_boxes()) >= 1

    mesh_octree = build_mesh_octree(mesh, max_depth=4, max_triangles=1)
    assert mesh_octree.root is not None
    assert len(mesh_octree.collect_boxes()) >= 1

    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[:2, :2, :2] = 1
    grid = VoxelGrid(
        data=data,
        bounds_min_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        bounds_max_xyz=np.array([4.0, 4.0, 4.0], dtype=np.float32),
        voxel_size_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    voxel_octree = build_voxel_octree(grid)
    assert voxel_octree.root is not None
    flat = voxel_octree.flatten()
    assert flat.shape[0] >= 1
