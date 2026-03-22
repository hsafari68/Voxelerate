from pathlib import Path

import numpy as np

from voxelerate import VoxelGrid, load_voxel_grid


def test_voxel_grid_hybrid_pickle_roundtrip(tmp_path: Path) -> None:
    data = np.zeros((4, 3, 2), dtype=np.uint8)
    data[1, 1, 1] = 1

    grid = VoxelGrid(
        data=data,
        bounds_min_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        bounds_max_xyz=np.array([2.0, 3.0, 4.0], dtype=np.float32),
        voxel_size_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        mode="solid",
        pixel_size=0.015,
        metadata={"source": "unit-test"},
    )

    path = tmp_path / "grid.pkl"
    grid.save_pickle(path, format="hybrid")
    loaded = VoxelGrid.load_pickle(path)

    assert loaded.axis_order == "zyx"
    assert loaded.grid_size_xyz == (2, 3, 4)
    assert loaded.mode == "solid"
    assert loaded.pixel_size == 0.015
    assert loaded.metadata["source"] == "unit-test"
    np.testing.assert_array_equal(loaded.data, grid.data)
    np.testing.assert_allclose(loaded.bounds_min_xyz, grid.bounds_min_xyz)
    np.testing.assert_allclose(loaded.bounds_max_xyz, grid.bounds_max_xyz)


def test_voxel_grid_legacy_pickle_roundtrip(tmp_path: Path) -> None:
    data = np.zeros((2, 2, 2), dtype=np.uint8)
    data[0, 0, 0] = 1

    grid = VoxelGrid(
        data=data,
        bounds_min_xyz=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        bounds_max_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        voxel_size_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    path = tmp_path / "grid_legacy.pkl"
    grid.save_pickle(path, format="legacy")
    loaded = load_voxel_grid(path)

    assert loaded.shape_zyx == (2, 2, 2)
    np.testing.assert_array_equal(loaded.data, data)


def test_voxel_grid_npz_roundtrip(tmp_path: Path) -> None:
    data = np.zeros((3, 4, 5), dtype=np.uint8)
    data[2, 1, 3] = 1

    grid = VoxelGrid(
        data=data,
        bounds_min_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        bounds_max_xyz=np.array([6.0, 6.0, 6.0], dtype=np.float32),
        voxel_size_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        metadata={"kind": "npz"},
    )

    path = tmp_path / "grid.npz"
    grid.save_npz(path)
    loaded = load_voxel_grid(path)

    np.testing.assert_array_equal(loaded.data, grid.data)
    assert loaded.metadata["kind"] == "npz"
