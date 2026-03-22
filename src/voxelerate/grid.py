from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]


def _expand_xyz_scalar(value: float | tuple[float, float, float] | NDArray[np.floating]) -> Float32Array:
    if np.isscalar(value):
        scalar = float(value)
        return np.array([scalar, scalar, scalar], dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(3)
    return np.ascontiguousarray(array, dtype=np.float32)


def _expand_padding(padding: float | tuple[float, float, float] | NDArray[np.floating]) -> Float32Array:
    return _expand_xyz_scalar(padding)


@dataclass(slots=True)
class VoxelizationGrid:
    """
    Explicit voxelization grid specification.

    - bounds are always in `(x, y, z)` world order
    - dense volume arrays are always `(z, y, x)`
    - `grid_size_xyz` is the OpenGL texture size `(x, y, z)`
    """

    bounds_min_xyz: Float32Array
    bounds_max_xyz: Float32Array
    grid_size_xyz: Int32Array
    voxel_size_xyz: Float32Array
    pixel_size: float | None = None

    def __post_init__(self) -> None:
        self.bounds_min_xyz = np.ascontiguousarray(self.bounds_min_xyz, dtype=np.float32).reshape(3)
        self.bounds_max_xyz = np.ascontiguousarray(self.bounds_max_xyz, dtype=np.float32).reshape(3)
        self.grid_size_xyz = np.ascontiguousarray(self.grid_size_xyz, dtype=np.int32).reshape(3)
        self.voxel_size_xyz = np.ascontiguousarray(self.voxel_size_xyz, dtype=np.float32).reshape(3)
        self.pixel_size = None if self.pixel_size is None else float(self.pixel_size)

        if np.any(self.grid_size_xyz <= 0):
            raise ValueError("grid_size_xyz must be positive in every axis")
        if np.any(self.voxel_size_xyz <= 0):
            raise ValueError("voxel_size_xyz must be positive in every axis")
        if np.any(self.bounds_max_xyz <= self.bounds_min_xyz):
            raise ValueError("bounds_max_xyz must be strictly larger than bounds_min_xyz")

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
        *,
        voxel_size: float | tuple[float, float, float],
        padding: float | tuple[float, float, float] = 0.0,
        transform: NDArray[np.floating] | None = None,
        pixel_size: float | None = None,
    ) -> "VoxelizationGrid":
        transformed = mesh.transformed_vertices(transform)
        bounds_min_xyz = transformed.min(axis=0).astype(np.float32)
        bounds_max_xyz = transformed.max(axis=0).astype(np.float32)
        return cls.from_bounds(
            bounds_min_xyz,
            bounds_max_xyz,
            voxel_size=voxel_size,
            padding=padding,
            pixel_size=pixel_size,
        )

    @classmethod
    def from_bounds(
        cls,
        bounds_min_xyz: NDArray[np.floating],
        bounds_max_xyz: NDArray[np.floating],
        *,
        voxel_size: float | tuple[float, float, float] | None = None,
        grid_size_xyz: tuple[int, int, int] | NDArray[np.integer] | None = None,
        padding: float | tuple[float, float, float] = 0.0,
        pixel_size: float | None = None,
        align_bounds_max: bool = True,
    ) -> "VoxelizationGrid":
        bounds_min = np.asarray(bounds_min_xyz, dtype=np.float32).reshape(3)
        bounds_max = np.asarray(bounds_max_xyz, dtype=np.float32).reshape(3)
        pad = _expand_padding(padding)

        padded_min = bounds_min - pad
        padded_requested_max = bounds_max + pad
        extent = padded_requested_max - padded_min

        if voxel_size is None and grid_size_xyz is None:
            raise ValueError("Either voxel_size or grid_size_xyz must be provided.")

        if voxel_size is not None and grid_size_xyz is None:
            voxel_size_xyz = _expand_xyz_scalar(voxel_size)
            safe_extent = np.maximum(extent, voxel_size_xyz)
            grid_size = np.maximum(np.ceil(safe_extent / voxel_size_xyz).astype(np.int32), 1)
            aligned_max = padded_min + grid_size.astype(np.float32) * voxel_size_xyz
        elif voxel_size is None and grid_size_xyz is not None:
            grid_size = np.asarray(grid_size_xyz, dtype=np.int32).reshape(3)
            if np.any(grid_size <= 0):
                raise ValueError("grid_size_xyz must be positive.")
            voxel_size_xyz = extent / grid_size.astype(np.float32)
            aligned_max = padded_requested_max
        else:
            grid_size = np.asarray(grid_size_xyz, dtype=np.int32).reshape(3)
            if np.any(grid_size <= 0):
                raise ValueError("grid_size_xyz must be positive.")
            voxel_size_xyz = _expand_xyz_scalar(voxel_size)
            computed_max = padded_min + grid_size.astype(np.float32) * voxel_size_xyz
            aligned_max = computed_max if align_bounds_max else padded_requested_max

        return cls(
            bounds_min_xyz=padded_min.astype(np.float32),
            bounds_max_xyz=aligned_max.astype(np.float32),
            grid_size_xyz=grid_size.astype(np.int32),
            voxel_size_xyz=voxel_size_xyz.astype(np.float32),
            pixel_size=pixel_size,
        )

    @classmethod
    def from_legacy(
        cls,
        *,
        min_point_xyz: NDArray[np.floating],
        max_point_xyz: NDArray[np.floating],
        num_voxels_zyx: NDArray[np.integer] | tuple[int, int, int],
        pixel_size: float | None = None,
        voxel_size: float | tuple[float, float, float] | NDArray[np.floating] | None = None,
    ) -> "VoxelizationGrid":
        legacy_num_voxels = np.asarray(num_voxels_zyx, dtype=np.int32).reshape(3)
        grid_size_xyz = np.array(
            [legacy_num_voxels[2], legacy_num_voxels[1], legacy_num_voxels[0]],
            dtype=np.int32,
        )

        if voxel_size is None:
            return cls.from_bounds(
                min_point_xyz,
                max_point_xyz,
                grid_size_xyz=grid_size_xyz,
                pixel_size=pixel_size,
            )

        return cls.from_bounds(
            min_point_xyz,
            max_point_xyz,
            voxel_size=voxel_size,
            grid_size_xyz=grid_size_xyz,
            pixel_size=pixel_size,
            align_bounds_max=True,
        )

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        x, y, z = [int(v) for v in self.grid_size_xyz.tolist()]
        return z, y, x

    @property
    def num_voxels_zyx(self) -> Int32Array:
        z, y, x = self.shape_zyx
        return np.array([z, y, x], dtype=np.int32)

    @property
    def extent_xyz(self) -> Float32Array:
        return (self.bounds_max_xyz - self.bounds_min_xyz).astype(np.float32)

    @property
    def voxel_size(self) -> float | Float32Array:
        if np.allclose(self.voxel_size_xyz, self.voxel_size_xyz[0]):
            return float(self.voxel_size_xyz[0])
        return self.voxel_size_xyz.copy()

    def summary(self) -> str:
        return (
            f"VoxelizationGrid(bounds_min_xyz={self.bounds_min_xyz.tolist()}, "
            f"bounds_max_xyz={self.bounds_max_xyz.tolist()}, "
            f"grid_size_xyz={self.grid_size_xyz.tolist()}, "
            f"voxel_size_xyz={self.voxel_size_xyz.tolist()}, pixel_size={self.pixel_size})"
        )
