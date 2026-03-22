from pathlib import Path

import numpy as np

from voxelerate import (
    VoxelGrid,
    build_bvh,
    build_mesh_octree,
    build_voxel_octree,
    extract_central_slices,
    load_bvh,
    load_mesh,
    load_mesh_octree,
    load_voxel_octree,
)
from voxelerate.slices import slice_overlay_rectangles


TETRA_OBJ = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""


def test_flat_structure_archives_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "tetra.obj"
    path.write_text(TETRA_OBJ)
    mesh = load_mesh(path)

    bvh = build_bvh(mesh, strategy="median", max_leaf_size=1)
    bvh_path = tmp_path / "tetra.bvh.npz"
    bvh.save_npz(bvh_path)
    flat_bvh = load_bvh(bvh_path)
    assert len(flat_bvh.collect_boxes()) >= 1
    assert flat_bvh.metadata["strategy"] == "median"

    mesh_octree = build_mesh_octree(mesh, max_depth=4, max_triangles=1)
    mesh_octree_path = tmp_path / "tetra_mesh_octree.npz"
    mesh_octree.save_npz(mesh_octree_path)
    flat_mesh_octree = load_mesh_octree(mesh_octree_path)
    assert len(flat_mesh_octree.collect_boxes()) >= 1
    assert flat_mesh_octree.metadata["triangle_count"] == mesh.triangle_count

    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[1:3, 1:3, 1:3] = 1
    grid = VoxelGrid(
        data=data,
        bounds_min_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        bounds_max_xyz=np.array([4.0, 4.0, 4.0], dtype=np.float32),
        voxel_size_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    voxel_octree = build_voxel_octree(grid)
    voxel_octree_path = tmp_path / "grid_voxel_octree.npz"
    voxel_octree.save_npz(voxel_octree_path)
    flat_voxel_octree = load_voxel_octree(voxel_octree_path)
    assert len(flat_voxel_octree.collect_boxes()) >= 1
    assert flat_voxel_octree.metadata["grid_shape_zyx"] == [4, 4, 4]


def test_extract_central_slices_and_octree_overlays() -> None:
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[1:3, 1:3, 1:3] = 1
    grid = VoxelGrid(
        data=data,
        bounds_min_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        bounds_max_xyz=np.array([4.0, 4.0, 4.0], dtype=np.float32),
        voxel_size_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    bundle = extract_central_slices(grid)
    assert bundle.x.image.shape == (4, 4)
    assert bundle.y.image.shape == (4, 4)
    assert bundle.z.image.shape == (4, 4)

    octree = build_voxel_octree(grid)
    rectangles = slice_overlay_rectangles(
        octree,
        axis="x",
        coordinate=bundle.x.coordinate,
        max_depth=4,
        occupied_only=True,
    )
    assert len(rectangles) >= 1
