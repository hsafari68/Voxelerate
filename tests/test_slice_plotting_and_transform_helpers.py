from pathlib import Path

import numpy as np
import pytest

from voxelerate import VoxelGrid, build_voxel_octree
from voxelerate.math3d import resolve_transform, rotation_matrix_y, scale_matrix, translation_matrix
from voxelerate.slices import SlicePlotStyle, plot_central_slices, plot_slice, slice_overlay_rectangles
from voxelerate.viewer import MeshViewer, _window_title_text, viewer_controls_help


def _example_grid() -> VoxelGrid:
    data = np.zeros((6, 6, 6), dtype=np.uint8)
    data[2:4, 2:4, 2:4] = 1
    return VoxelGrid(
        data=data,
        bounds_min_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        bounds_max_xyz=np.array([6.0, 6.0, 6.0], dtype=np.float32),
        voxel_size_xyz=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_plot_central_slices_can_save_image(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    grid = _example_grid()
    octree = build_voxel_octree(grid)
    output_path = tmp_path / "central_slices.png"

    fig = plot_central_slices(grid, octree=octree, show=False, save_path=output_path)
    assert output_path.exists()
    assert fig is not None


def test_plot_slice_can_save_image(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    grid = _example_grid()
    octree = build_voxel_octree(grid)
    output_path = tmp_path / "z_slice.png"

    fig = plot_slice(grid, axis="z", overlay=octree, show=False, save_path=output_path)
    assert output_path.exists()
    assert fig is not None




def test_plot_central_slices_returns_none_by_default_when_shown(monkeypatch) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    grid = _example_grid()
    result = plot_central_slices(grid, show=True)
    assert result is None


def test_plot_slice_returns_none_by_default_when_shown(monkeypatch) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    grid = _example_grid()
    result = plot_slice(grid, axis="z", show=True)
    assert result is None

def test_resolve_transform_applies_components_after_base_matrix() -> None:
    base = translation_matrix([1.0, 0.0, 0.0])
    resolved = resolve_transform(base, translation_xyz=np.array([0.0, 2.0, 0.0], dtype=np.float32))
    expected = translation_matrix([0.0, 2.0, 0.0]) @ base
    np.testing.assert_allclose(resolved, expected, atol=1e-6)


def test_mesh_viewer_rotation_helper_applies_expected_matrix() -> None:
    viewer = MeshViewer()
    viewer._model_transform = np.eye(4, dtype=np.float32)
    viewer._transform_pivot = np.zeros(3, dtype=np.float32)

    viewer._rotate_model_by_angles(yaw_degrees=90.0)

    np.testing.assert_allclose(viewer._model_transform, rotation_matrix_y(90.0), atol=1e-6)


def test_mesh_viewer_scale_helper_applies_expected_matrix() -> None:
    viewer = MeshViewer()
    viewer._model_transform = np.eye(4, dtype=np.float32)
    viewer._transform_pivot = np.zeros(3, dtype=np.float32)

    viewer._scale_model(1.0)

    np.testing.assert_allclose(viewer._model_transform, scale_matrix(1.08), atol=1e-6)



def test_slice_overlay_rectangles_default_to_full_octree_overlay() -> None:
    grid = _example_grid()
    octree = build_voxel_octree(grid, max_depth=3)

    rectangles = slice_overlay_rectangles(octree, axis="z", coordinate=3.5)

    assert rectangles
    root = rectangles[0]
    assert root.x_min == pytest.approx(0.0)
    assert root.x_max == pytest.approx(6.0)
    assert root.y_min == pytest.approx(0.0)
    assert root.y_max == pytest.approx(6.0)


def test_slice_plot_style_uses_light_voxels_on_dark_background() -> None:
    style = SlicePlotStyle()

    assert style.empty_voxel_color == "#111827"
    assert style.filled_voxel_color == "#f8fafc"
    assert style.overlay_color == "#ef4444"


def test_viewer_controls_help_mentions_current_controls() -> None:
    text = viewer_controls_help()

    assert "Left drag" in text
    assert "Toggle mesh rendering" in text
    assert "x-ray solid voxel preview" in text
    assert "Toggle this help" not in text


def test_window_title_uses_current_control_summary() -> None:
    title = _window_title_text("Viewer")

    assert "Press H" not in title
    assert "X xray" in title
    assert "Mouse: L-orbit R-pan Wheel-zoom" in title



def test_viewer_ground_starts_hidden() -> None:
    from voxelerate.viewer import MeshViewer

    viewer = MeshViewer()
    assert viewer._show_ground is False
