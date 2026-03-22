from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .bvh import BVH
from .octree import MeshOctree, VoxelOctree
from .spatial_io import FlatBVH, FlatMeshOctree, FlatVoxelOctree
from .voxel_grid import VoxelGrid

Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]


@dataclass(slots=True)
class Slice2D:
    axis: str
    index: int
    coordinate: float
    image: UInt8Array
    extent: tuple[float, float, float, float]
    horizontal_label: str
    vertical_label: str


@dataclass(slots=True)
class SliceOverlayRectangle:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    depth: int
    value: int | None = None


@dataclass(slots=True)
class SliceBundle:
    x: Slice2D
    y: Slice2D
    z: Slice2D


@dataclass(slots=True)
class SlicePlotStyle:
    """Visual style for 2D voxel slice figures."""

    figure_facecolor: str = "#0b1020"
    axes_facecolor: str = "#111827"
    empty_voxel_color: str = "#111827"
    filled_voxel_color: str = "#f8fafc"
    overlay_color: str = "#ef4444"
    overlay_halo_color: str = "#020617"
    overlay_alpha: float = 0.98
    overlay_linewidth_base: float = 1.8
    overlay_linewidth_decay: float = 0.07
    spine_color: str = "#334155"
    tick_color: str = "#cbd5e1"
    label_color: str = "#e2e8f0"
    title_color: str = "#f8fafc"
    legend_facecolor: str = "#020617"
    legend_edgecolor: str = "#334155"
    legend_frame_alpha: float = 0.92
    title_fontsize: float = 13.0
    suptitle_fontsize: float = 17.0
    legend_fontsize: float = 10.5


_SUPPORTED_OVERLAY_TYPES = (BVH, MeshOctree, VoxelOctree, FlatBVH, FlatMeshOctree, FlatVoxelOctree)


def _ensure_grid(grid_or_path: VoxelGrid | str | Path) -> VoxelGrid:
    if isinstance(grid_or_path, VoxelGrid):
        return grid_or_path
    return VoxelGrid.load_npz(grid_or_path) if str(grid_or_path).lower().endswith(".npz") else VoxelGrid.load_pickle(grid_or_path)


def _axis_coordinate(grid: VoxelGrid, axis: str, index: int) -> float:
    axis_to_world = {"x": 0, "y": 1, "z": 2}
    world_axis = axis_to_world[axis]
    return float(grid.bounds_min_xyz[world_axis] + (index + 0.5) * grid.voxel_size_xyz[world_axis])


def central_slice_indices(grid_or_path: VoxelGrid | str | Path) -> dict[str, int]:
    grid = _ensure_grid(grid_or_path)
    z, y, x = grid.shape_zyx
    return {"x": x // 2, "y": y // 2, "z": z // 2}


def extract_slice(
    grid_or_path: VoxelGrid | str | Path,
    axis: str,
    index: int | None = None,
) -> Slice2D:
    grid = _ensure_grid(grid_or_path)
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")

    indices = central_slice_indices(grid)
    slice_index = indices[axis] if index is None else int(index)

    x_min, y_min, z_min = grid.bounds_min_xyz.tolist()
    x_max, y_max, z_max = grid.bounds_max_xyz.tolist()

    if axis == "x":
        _, _, nx = grid.shape_zyx
        if not 0 <= slice_index < nx:
            raise IndexError("x slice index out of range")
        image = np.ascontiguousarray(grid.data[:, :, slice_index], dtype=np.uint8)
        extent = (y_min, y_max, z_min, z_max)
        horizontal_label = "y"
        vertical_label = "z"
    elif axis == "y":
        _, ny, _ = grid.shape_zyx
        if not 0 <= slice_index < ny:
            raise IndexError("y slice index out of range")
        image = np.ascontiguousarray(grid.data[:, slice_index, :], dtype=np.uint8)
        extent = (x_min, x_max, z_min, z_max)
        horizontal_label = "x"
        vertical_label = "z"
    else:
        nz, _, _ = grid.shape_zyx
        if not 0 <= slice_index < nz:
            raise IndexError("z slice index out of range")
        image = np.ascontiguousarray(grid.data[slice_index, :, :], dtype=np.uint8)
        extent = (x_min, x_max, y_min, y_max)
        horizontal_label = "x"
        vertical_label = "y"

    return Slice2D(
        axis=axis,
        index=slice_index,
        coordinate=_axis_coordinate(grid, axis, slice_index),
        image=image,
        extent=extent,
        horizontal_label=horizontal_label,
        vertical_label=vertical_label,
    )


def extract_central_slices(grid_or_path: VoxelGrid | str | Path) -> SliceBundle:
    grid = _ensure_grid(grid_or_path)
    return SliceBundle(
        x=extract_slice(grid, "x"),
        y=extract_slice(grid, "y"),
        z=extract_slice(grid, "z"),
    )


def _overlay_is_octree(
    structure: BVH | MeshOctree | VoxelOctree | FlatBVH | FlatMeshOctree | FlatVoxelOctree,
) -> bool:
    return isinstance(structure, (MeshOctree, VoxelOctree, FlatMeshOctree, FlatVoxelOctree))


def _overlay_label(
    structure: BVH | MeshOctree | VoxelOctree | FlatBVH | FlatMeshOctree | FlatVoxelOctree,
) -> str:
    if isinstance(structure, (VoxelOctree, FlatVoxelOctree)):
        return "Voxel octree"
    if isinstance(structure, (MeshOctree, FlatMeshOctree)):
        return "Mesh octree"
    return "BVH"


def _resolve_leaves_only(
    structure: BVH | MeshOctree | VoxelOctree | FlatBVH | FlatMeshOctree | FlatVoxelOctree,
    leaves_only: bool | None,
) -> bool:
    if leaves_only is not None:
        return bool(leaves_only)
    _ = structure
    return False


def slice_overlay_rectangles(
    structure: BVH | MeshOctree | VoxelOctree | FlatBVH | FlatMeshOctree | FlatVoxelOctree,
    *,
    axis: str,
    coordinate: float,
    max_depth: int | None = None,
    leaves_only: bool | None = None,
    occupied_only: bool = True,
) -> list[SliceOverlayRectangle]:
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")
    if not isinstance(structure, _SUPPORTED_OVERLAY_TYPES):
        raise TypeError("slice overlays currently support BVH and octree structures")

    leaves_only_to_use = _resolve_leaves_only(structure, leaves_only)

    if isinstance(structure, (VoxelOctree, FlatVoxelOctree)):
        boxes = structure.collect_boxes(
            max_depth=max_depth,
            leaves_only=leaves_only_to_use,
            occupied_only=occupied_only,
        )
    else:
        boxes = [
            (mn, mx, depth, None)
            for mn, mx, depth in structure.collect_boxes(
                max_depth=max_depth,
                leaves_only=leaves_only_to_use,
            )
        ]
    rectangles: list[SliceOverlayRectangle] = []

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    for entry in boxes:
        bounds_min_xyz = np.asarray(entry[0], dtype=np.float32)
        bounds_max_xyz = np.asarray(entry[1], dtype=np.float32)
        depth = int(entry[2])
        value = None if len(entry) < 4 else entry[3]

        if not (float(bounds_min_xyz[axis_index]) <= coordinate <= float(bounds_max_xyz[axis_index])):
            continue

        if axis == "x":
            x0, x1 = float(bounds_min_xyz[1]), float(bounds_max_xyz[1])
            y0, y1 = float(bounds_min_xyz[2]), float(bounds_max_xyz[2])
        elif axis == "y":
            x0, x1 = float(bounds_min_xyz[0]), float(bounds_max_xyz[0])
            y0, y1 = float(bounds_min_xyz[2]), float(bounds_max_xyz[2])
        else:
            x0, x1 = float(bounds_min_xyz[0]), float(bounds_max_xyz[0])
            y0, y1 = float(bounds_min_xyz[1]), float(bounds_max_xyz[1])

        rectangles.append(
            SliceOverlayRectangle(
                x_min=x0,
                x_max=x1,
                y_min=y0,
                y_max=y1,
                depth=depth,
                value=value,
            )
        )

    return rectangles


def _draw_slice_axis(
    axis_obj,
    slice_view: Slice2D,
    *,
    overlay: BVH | MeshOctree | VoxelOctree | FlatBVH | FlatMeshOctree | FlatVoxelOctree | None,
    max_depth: int | None,
    leaves_only: bool | None,
    occupied_only: bool,
    style: SlicePlotStyle,
) -> None:
    from matplotlib import patheffects as pe
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    axis_obj.set_facecolor(style.axes_facecolor)
    axis_obj.imshow(
        slice_view.image,
        origin="lower",
        extent=slice_view.extent,
        interpolation="nearest",
        cmap=ListedColormap([style.empty_voxel_color, style.filled_voxel_color]),
        vmin=0,
        vmax=1,
        zorder=1,
    )
    axis_obj.set_title(
        f"{slice_view.axis.upper()} slice @ index {slice_view.index} ({slice_view.coordinate:.4g})",
        fontsize=style.title_fontsize,
        fontweight="semibold",
        color=style.title_color,
        pad=8,
    )
    axis_obj.set_xlabel(slice_view.horizontal_label, color=style.label_color, fontsize=11)
    axis_obj.set_ylabel(slice_view.vertical_label, color=style.label_color, fontsize=11)
    axis_obj.set_aspect("equal", adjustable="box")
    axis_obj.tick_params(axis="both", colors=style.tick_color, labelsize=10)
    axis_obj.grid(False)
    axis_obj.set_xlim(slice_view.extent[0], slice_view.extent[1])
    axis_obj.set_ylim(slice_view.extent[2], slice_view.extent[3])
    for spine in axis_obj.spines.values():
        spine.set_color(style.spine_color)
        spine.set_linewidth(1.0)

    if overlay is None:
        return

    overlays = slice_overlay_rectangles(
        overlay,
        axis=slice_view.axis,
        coordinate=slice_view.coordinate,
        max_depth=max_depth,
        leaves_only=leaves_only,
        occupied_only=occupied_only,
    )
    for rect in overlays:
        linewidth = max(0.8, style.overlay_linewidth_base - style.overlay_linewidth_decay * rect.depth)
        patch = Rectangle(
            (rect.x_min, rect.y_min),
            rect.x_max - rect.x_min,
            rect.y_max - rect.y_min,
            fill=False,
            edgecolor=style.overlay_color,
            linewidth=linewidth,
            alpha=style.overlay_alpha,
            joinstyle="round",
            capstyle="round",
            zorder=3,
        )
        patch.set_path_effects([
            pe.Stroke(linewidth=linewidth + 1.1, foreground=style.overlay_halo_color, alpha=0.98),
            pe.Normal(),
        ])
        axis_obj.add_patch(patch)


def _style_legend(legend, style: SlicePlotStyle) -> None:
    frame = legend.get_frame()
    frame.set_facecolor(style.legend_facecolor)
    frame.set_edgecolor(style.legend_edgecolor)
    frame.set_linewidth(0.9)
    frame.set_alpha(style.legend_frame_alpha)
    for text_item in legend.get_texts():
        text_item.set_color(style.label_color)


def plot_slice(
    grid_or_path: VoxelGrid | str | Path,
    axis: str,
    index: int | None = None,
    *,
    overlay: BVH | MeshOctree | VoxelOctree | FlatBVH | FlatMeshOctree | FlatVoxelOctree | None = None,
    max_depth: int | None = None,
    leaves_only: bool | None = None,
    occupied_only: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    style: SlicePlotStyle | None = None,
    return_figure: bool | None = None,
):
    """Plot one voxel slice with an optional BVH or octree overlay.

    By default, the figure object is returned only when ``show=False``. This keeps
    interactive environments from displaying the same figure twice while still
    supporting programmatic access when requested.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise ImportError(
            "matplotlib is required for slice plotting. Install it with `pip install matplotlib` "
            "or install the optional extra `voxelerate[viz]`."
        ) from exc

    slice_view = extract_slice(grid_or_path, axis=axis, index=index)
    style_to_use = SlicePlotStyle() if style is None else replace(style)

    fig, ax = plt.subplots(1, 1, figsize=(6.3, 6.2), constrained_layout=False)
    fig.patch.set_facecolor(style_to_use.figure_facecolor)
    fig.subplots_adjust(top=0.88, left=0.12, right=0.96, bottom=0.11)
    _draw_slice_axis(
        ax,
        slice_view,
        overlay=overlay,
        max_depth=max_depth,
        leaves_only=leaves_only,
        occupied_only=occupied_only,
        style=style_to_use,
    )

    legend_handles = [
        Patch(facecolor=style_to_use.filled_voxel_color, edgecolor=style_to_use.filled_voxel_color, label="Occupied voxels"),
    ]
    if overlay is not None:
        legend_handles.append(
            Line2D([0], [0], color=style_to_use.overlay_color, linewidth=style_to_use.overlay_linewidth_base, label=_overlay_label(overlay))
        )
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=style_to_use.legend_fontsize,
        frameon=True,
        fancybox=True,
        borderpad=0.55,
        handlelength=1.8,
    )
    _style_legend(legend, style_to_use)

    if title is not None:
        fig.suptitle(
            title,
            fontsize=style_to_use.suptitle_fontsize,
            fontweight="semibold",
            color=style_to_use.title_color,
            y=0.97,
        )

    should_return_figure = (not show) if return_figure is None else bool(return_figure)

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=180, facecolor=style_to_use.figure_facecolor, bbox_inches="tight")
    if show:
        plt.show()
    if not should_return_figure:
        plt.close(fig)
        return None
    return fig


def plot_central_slices(
    grid_or_path: VoxelGrid | str | Path,
    *,
    octree: BVH | MeshOctree | VoxelOctree | FlatBVH | FlatMeshOctree | FlatVoxelOctree | None = None,
    max_depth: int | None = None,
    leaves_only: bool | None = None,
    occupied_only: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    style: SlicePlotStyle | None = None,
    return_figure: bool | None = None,
):
    """Plot the central X/Y/Z slices of a voxel grid with optional hierarchy overlays.

    By default, the figure object is returned only when ``show=False``. This keeps
    interactive environments from displaying the same figure twice while still
    supporting programmatic access when requested.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise ImportError(
            "matplotlib is required for slice plotting. Install it with `pip install matplotlib` "
            "or install the optional extra `voxelerate[viz]`."
        ) from exc

    grid = _ensure_grid(grid_or_path)
    slices = extract_central_slices(grid)
    style_to_use = SlicePlotStyle() if style is None else replace(style)
    if octree is not None and max_depth is None:
        max_depth_to_use = None
    else:
        max_depth_to_use = max_depth

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.9), constrained_layout=False)
    fig.patch.set_facecolor(style_to_use.figure_facecolor)
    fig.subplots_adjust(top=0.80, left=0.045, right=0.99, bottom=0.10, wspace=0.22)
    for subplot, slice_view in zip(axes, [slices.x, slices.y, slices.z]):
        _draw_slice_axis(
            subplot,
            slice_view,
            overlay=octree,
            max_depth=max_depth_to_use,
            leaves_only=leaves_only,
            occupied_only=occupied_only,
            style=style_to_use,
        )

    legend_handles = [
        Patch(facecolor=style_to_use.filled_voxel_color, edgecolor=style_to_use.filled_voxel_color, label="Occupied voxels"),
    ]
    if octree is not None:
        legend_handles.append(
            Line2D([0], [0], color=style_to_use.overlay_color, linewidth=style_to_use.overlay_linewidth_base, label=_overlay_label(octree))
        )
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(legend_handles),
        frameon=True,
        fontsize=style_to_use.legend_fontsize,
        labelcolor=style_to_use.label_color,
        fancybox=True,
        borderpad=0.6,
        handlelength=1.8,
    )
    _style_legend(legend, style_to_use)

    if title is not None:
        fig.suptitle(
            title,
            fontsize=style_to_use.suptitle_fontsize,
            fontweight="semibold",
            color=style_to_use.title_color,
            y=0.975,
        )

    should_return_figure = (not show) if return_figure is None else bool(return_figure)

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=180, facecolor=style_to_use.figure_facecolor, bbox_inches="tight")
    if show:
        plt.show()
    if not should_return_figure:
        plt.close(fig)
        return None
    return fig


__all__ = [
    "Slice2D",
    "SliceBundle",
    "SliceOverlayRectangle",
    "SlicePlotStyle",
    "central_slice_indices",
    "extract_central_slices",
    "extract_slice",
    "plot_central_slices",
    "plot_slice",
    "slice_overlay_rectangles",
]
