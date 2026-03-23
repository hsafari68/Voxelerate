from __future__ import annotations

from pathlib import Path
from typing import overload

import numpy as np

from .bvh import BVH
from .grid import VoxelizationGrid
from .math3d import resolve_transform
from .mesh import Mesh, load_mesh
from .octree import MeshOctree, VoxelOctree
from .slices import SlicePlotStyle, extract_central_slices, plot_central_slices, plot_slice
from .spatial_io import FlatBVH, FlatMeshOctree, FlatVoxelOctree, load_bvh, load_mesh_octree, load_voxel_octree
from .voxel_grid import VoxelGrid


@overload
def voxelize(
    mesh_or_path: Mesh,
    voxel_size: float | tuple[float, float, float],
    *,
    mode: str = "solid",
    padding: float | tuple[float, float, float] = 0.0,
    transform: np.ndarray | None = None,
    translation_xyz: np.ndarray | None = None,
    rotation_degrees_xyz: np.ndarray | None = None,
    scale_xyz: float | np.ndarray | None = None,
    pivot_xyz: np.ndarray | None = None,
    visible_context: bool = False,
    pixel_size: float | None = None,
) -> VoxelGrid:
    ...


@overload
def voxelize(
    mesh_or_path: str | Path,
    voxel_size: float | tuple[float, float, float],
    *,
    center: bool = False,
    mode: str = "solid",
    padding: float | tuple[float, float, float] = 0.0,
    transform: np.ndarray | None = None,
    translation_xyz: np.ndarray | None = None,
    rotation_degrees_xyz: np.ndarray | None = None,
    scale_xyz: float | np.ndarray | None = None,
    pivot_xyz: np.ndarray | None = None,
    visible_context: bool = False,
    pixel_size: float | None = None,
) -> VoxelGrid:
    ...


def _ensure_mesh(mesh_or_path: Mesh | str | Path, *, center: bool = False) -> Mesh:
    return load_mesh(mesh_or_path, center=center) if isinstance(mesh_or_path, (str, Path)) else mesh_or_path


def build_voxelization_grid(
    mesh_or_path: Mesh | str | Path,
    *,
    center: bool = False,
    voxel_size: float | tuple[float, float, float] | None = None,
    bounds_min_xyz: np.ndarray | None = None,
    bounds_max_xyz: np.ndarray | None = None,
    grid_size_xyz: tuple[int, int, int] | np.ndarray | None = None,
    padding: float | tuple[float, float, float] = 0.0,
    transform: np.ndarray | None = None,
    translation_xyz: np.ndarray | None = None,
    rotation_degrees_xyz: np.ndarray | None = None,
    scale_xyz: float | np.ndarray | None = None,
    pivot_xyz: np.ndarray | None = None,
    pixel_size: float | None = None,
) -> VoxelizationGrid:
    mesh = _ensure_mesh(mesh_or_path, center=center)
    final_transform = resolve_transform(
        transform=transform,
        translation_xyz=translation_xyz,
        rotation_degrees_xyz=rotation_degrees_xyz,
        scale_xyz=scale_xyz,
        pivot_xyz=pivot_xyz,
    )

    if bounds_min_xyz is None or bounds_max_xyz is None:
        transformed = mesh.transformed_vertices(final_transform)
        mesh_min = transformed.min(axis=0).astype(np.float32)
        mesh_max = transformed.max(axis=0).astype(np.float32)
        bounds_min_xyz = mesh_min if bounds_min_xyz is None else np.asarray(bounds_min_xyz, dtype=np.float32)
        bounds_max_xyz = mesh_max if bounds_max_xyz is None else np.asarray(bounds_max_xyz, dtype=np.float32)

    return VoxelizationGrid.from_bounds(
        bounds_min_xyz=np.asarray(bounds_min_xyz, dtype=np.float32),
        bounds_max_xyz=np.asarray(bounds_max_xyz, dtype=np.float32),
        voxel_size=voxel_size,
        grid_size_xyz=grid_size_xyz,
        padding=padding,
        pixel_size=pixel_size,
    )


def voxelize(
    mesh_or_path: Mesh | str | Path,
    voxel_size: float | tuple[float, float, float],
    *,
    center: bool = False,
    mode: str = "solid",
    padding: float | tuple[float, float, float] = 0.0,
    transform: np.ndarray | None = None,
    translation_xyz: np.ndarray | None = None,
    rotation_degrees_xyz: np.ndarray | None = None,
    scale_xyz: float | np.ndarray | None = None,
    pivot_xyz: np.ndarray | None = None,
    visible_context: bool = False,
    pixel_size: float | None = None,
) -> VoxelGrid:
    """Voxelize a mesh on the GPU using a grid derived from the mesh bounds."""

    mesh = _ensure_mesh(mesh_or_path, center=center)
    final_transform = resolve_transform(
        transform=transform,
        translation_xyz=translation_xyz,
        rotation_degrees_xyz=rotation_degrees_xyz,
        scale_xyz=scale_xyz,
        pivot_xyz=pivot_xyz,
    )

    from .voxelizer import OpenGLVoxelizer

    with OpenGLVoxelizer(visible_context=visible_context) as voxelizer:
        return voxelizer.voxelize(
            mesh,
            voxel_size=voxel_size,
            mode=mode,
            padding=padding,
            transform=final_transform,
            pixel_size=pixel_size,
        )


def voxelize_on_grid(
    mesh_or_path: Mesh | str | Path,
    *,
    grid: VoxelizationGrid | None = None,
    voxel_size: float | tuple[float, float, float] | None = None,
    bounds_min_xyz: np.ndarray | None = None,
    bounds_max_xyz: np.ndarray | None = None,
    grid_size_xyz: tuple[int, int, int] | np.ndarray | None = None,
    center: bool = False,
    mode: str = "solid",
    padding: float | tuple[float, float, float] = 0.0,
    transform: np.ndarray | None = None,
    translation_xyz: np.ndarray | None = None,
    rotation_degrees_xyz: np.ndarray | None = None,
    scale_xyz: float | np.ndarray | None = None,
    pivot_xyz: np.ndarray | None = None,
    visible_context: bool = False,
    pixel_size: float | None = None,
) -> VoxelGrid:
    """
    Voxelize a mesh on an explicit grid.

    This is the API equivalent of the explicit workflow where `grid_size_xyz`,
    `bounds_min_xyz`, and `bounds_max_xyz` come from an external setup.
    """
    mesh = _ensure_mesh(mesh_or_path, center=center)
    final_transform = resolve_transform(
        transform=transform,
        translation_xyz=translation_xyz,
        rotation_degrees_xyz=rotation_degrees_xyz,
        scale_xyz=scale_xyz,
        pivot_xyz=pivot_xyz,
    )

    if grid is None:
        grid = build_voxelization_grid(
            mesh,
            voxel_size=voxel_size,
            bounds_min_xyz=bounds_min_xyz,
            bounds_max_xyz=bounds_max_xyz,
            grid_size_xyz=grid_size_xyz,
            padding=padding,
            transform=final_transform,
            pixel_size=pixel_size,
        )

    from .voxelizer import OpenGLVoxelizer

    with OpenGLVoxelizer(visible_context=visible_context) as voxelizer:
        return voxelizer.voxelize_on_grid(
            mesh,
            grid=grid,
            mode=mode,
            transform=final_transform,
        )


def show(
    mesh_or_path: Mesh | str | Path | None = None,
    *,
    voxel_grid: VoxelGrid | str | Path | None = None,
    center: bool = False,
    transform: np.ndarray | None = None,
    translation_xyz: np.ndarray | None = None,
    rotation_degrees_xyz: np.ndarray | None = None,
    scale_xyz: float | np.ndarray | None = None,
    pivot_xyz: np.ndarray | None = None,
    title: str = "Voxelerate Viewer",
    bvh: BVH | FlatBVH | None = None,
    octree: MeshOctree | VoxelOctree | FlatMeshOctree | FlatVoxelOctree | None = None,
    return_visual_transform: bool = False,
    editable: bool | None = None,
) -> np.ndarray | None:
    """Launch the interactive viewer.

    Viewer mouse and keyboard transforms are preview-only and do not affect any
    later voxelization. To transform geometry for voxelization, pass the desired
    matrix or translation/rotation/scale inputs directly to `voxelize(...)` or
    `voxelize_on_grid(...)`.

    `editable` is accepted for backward compatibility but no longer changes the
    control scheme because preview controls are always enabled.
    """

    mesh = None if mesh_or_path is None else _ensure_mesh(mesh_or_path, center=center)
    if isinstance(voxel_grid, (str, Path)):
        voxel_grid = load_voxel_grid(voxel_grid)

    base_transform = resolve_transform(
        transform=transform,
        translation_xyz=translation_xyz,
        rotation_degrees_xyz=rotation_degrees_xyz,
        scale_xyz=scale_xyz,
        pivot_xyz=pivot_xyz,
    )

    from .viewer import MeshViewer

    viewer = MeshViewer(title=title)
    _ = editable  # backward-compatible no-op
    return viewer.show(
        mesh,
        voxel_grid=voxel_grid,
        bvh=bvh,
        octree=octree,
        transform=base_transform,
        return_visual_transform=return_visual_transform,
    )


def build_bvh(
    mesh_or_path: Mesh | str | Path,
    *,
    center: bool = False,
    strategy: str = "sah",
    max_leaf_size: int = 4,
    bins: int = 16,
) -> BVH:
    mesh = _ensure_mesh(mesh_or_path, center=center)
    return BVH.from_mesh(mesh, strategy=strategy, max_leaf_size=max_leaf_size, bins=bins)


def build_mesh_octree(
    mesh_or_path: Mesh | str | Path,
    *,
    center: bool = False,
    max_depth: int = 8,
    max_triangles: int = 32,
) -> MeshOctree:
    mesh = _ensure_mesh(mesh_or_path, center=center)
    return MeshOctree.from_mesh(mesh, max_depth=max_depth, max_triangles=max_triangles)


def build_voxel_octree(
    grid_or_path: VoxelGrid | str | Path,
    *,
    min_dim: int = 1,
    max_depth: int | None = None,
) -> VoxelOctree:
    grid = load_voxel_grid(grid_or_path) if isinstance(grid_or_path, (str, Path)) else grid_or_path
    return VoxelOctree.from_grid(grid, min_dim=min_dim, max_depth=max_depth)


def load_voxel_grid(path: str | Path) -> VoxelGrid:
    input_path = Path(path)
    if input_path.suffix.lower() == ".npz":
        return VoxelGrid.load_npz(input_path)
    return VoxelGrid.load_pickle(input_path)


__all__ = [
    "BVH",
    "SlicePlotStyle",
    "FlatBVH",
    "FlatMeshOctree",
    "FlatVoxelOctree",
    "Mesh",
    "MeshOctree",
    "VoxelGrid",
    "VoxelOctree",
    "VoxelizationGrid",
    "build_bvh",
    "build_mesh_octree",
    "build_voxel_octree",
    "build_voxelization_grid",
    "extract_central_slices",
    "load_bvh",
    "load_mesh",
    "load_mesh_octree",
    "load_voxel_grid",
    "load_voxel_octree",
    "plot_central_slices",
    "plot_slice",
    "show",
    "voxelize",
    "voxelize_on_grid",
]
