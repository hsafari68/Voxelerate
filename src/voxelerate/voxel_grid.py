from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import pickle

import numpy as np
from numpy.typing import NDArray

from .grid import VoxelizationGrid
from .version import __version__

UInt8Array = NDArray[np.uint8]
Float32Array = NDArray[np.float32]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return str(value)


def _expand_voxel_size(value: float | NDArray[np.floating]) -> Float32Array:
    if np.isscalar(value):
        scalar = float(value)
        return np.array([scalar, scalar, scalar], dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(3).astype(np.float32, copy=False)


@dataclass(slots=True)
class VoxelGrid:
    """Voxelized volume data stored in explicit `(z, y, x)` axis order."""

    data: UInt8Array
    bounds_min_xyz: Float32Array
    bounds_max_xyz: Float32Array
    voxel_size_xyz: Float32Array | float
    mode: str = "solid"
    pixel_size: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.ascontiguousarray(self.data, dtype=np.uint8)
        self.bounds_min_xyz = np.ascontiguousarray(self.bounds_min_xyz, dtype=np.float32).reshape(3)
        self.bounds_max_xyz = np.ascontiguousarray(self.bounds_max_xyz, dtype=np.float32).reshape(3)
        self.voxel_size_xyz = np.ascontiguousarray(_expand_voxel_size(self.voxel_size_xyz), dtype=np.float32)
        self.mode = str(self.mode)
        self.pixel_size = None if self.pixel_size is None else float(self.pixel_size)
        self.metadata = dict(self.metadata)

        if self.data.ndim != 3:
            raise ValueError("data must have shape (z, y, x)")
        if np.any(self.bounds_max_xyz <= self.bounds_min_xyz):
            raise ValueError("bounds_max_xyz must be strictly larger than bounds_min_xyz")
        if np.any(self.voxel_size_xyz <= 0):
            raise ValueError("voxel_size_xyz must be positive")

    @property
    def axis_order(self) -> str:
        return "zyx"

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        z, y, x = self.data.shape
        return int(z), int(y), int(x)

    @property
    def grid_size_xyz(self) -> tuple[int, int, int]:
        z, y, x = self.data.shape
        return int(x), int(y), int(z)

    @property
    def num_voxels_zyx(self) -> tuple[int, int, int]:
        return self.shape_zyx

    @property
    def voxel_size(self) -> float | Float32Array:
        if np.allclose(self.voxel_size_xyz, self.voxel_size_xyz[0]):
            return float(self.voxel_size_xyz[0])
        return self.voxel_size_xyz.copy()

    @property
    def grid(self) -> VoxelizationGrid:
        return VoxelizationGrid(
            bounds_min_xyz=self.bounds_min_xyz.copy(),
            bounds_max_xyz=self.bounds_max_xyz.copy(),
            grid_size_xyz=np.array(self.grid_size_xyz, dtype=np.int32),
            voxel_size_xyz=self.voxel_size_xyz.copy(),
            pixel_size=self.pixel_size,
        )

    @property
    def occupied_voxels(self) -> int:
        return int(np.count_nonzero(self.data))

    @property
    def occupied_fraction(self) -> float:
        if self.data.size == 0:
            return 0.0
        return float(self.occupied_voxels / self.data.size)

    def occupied_indices(self) -> NDArray[np.int32]:
        return np.argwhere(self.data > 0).astype(np.int32, copy=False)

    def occupied_centers(self) -> Float32Array:
        occupied = self.occupied_indices()
        if occupied.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        centers = np.empty((occupied.shape[0], 3), dtype=np.float32)
        centers[:, 0] = self.bounds_min_xyz[0] + (occupied[:, 2] + 0.5) * self.voxel_size_xyz[0]
        centers[:, 1] = self.bounds_min_xyz[1] + (occupied[:, 1] + 0.5) * self.voxel_size_xyz[1]
        centers[:, 2] = self.bounds_min_xyz[2] + (occupied[:, 0] + 0.5) * self.voxel_size_xyz[2]
        return centers

    def to_pickle_dict(self, *, format: str = "hybrid") -> dict[str, Any]:
        legacy_voxel_size = float(self.voxel_size_xyz[0]) if np.allclose(
            self.voxel_size_xyz, self.voxel_size_xyz[0]
        ) else self.voxel_size_xyz.copy()

        rich = {
            "format": "voxelerate.voxelgrid",
            "format_version": 2,
            "voxelerate_version": __version__,
            "axis_order": self.axis_order,
            "array_zyx": self.data,
            "shape_zyx": np.array(self.shape_zyx, dtype=np.int32),
            "grid_size_xyz": np.array(self.grid_size_xyz, dtype=np.int32),
            "bounds_min_xyz": self.bounds_min_xyz.copy(),
            "bounds_max_xyz": self.bounds_max_xyz.copy(),
            "voxel_size_xyz": self.voxel_size_xyz.copy(),
            "voxel_size": legacy_voxel_size,
            "pixel_size": self.pixel_size,
            "mode": self.mode,
            "metadata": self.metadata,
        }

        legacy = {
            "data": np.ascontiguousarray(self.data.reshape(-1), dtype=np.uint8),
            "min_point": self.bounds_min_xyz.copy(),
            "max_point": self.bounds_max_xyz.copy(),
            "num_voxels": np.array(self.shape_zyx, dtype=np.int32),
            "pixel_size": self.pixel_size,
            "voxel_size": legacy_voxel_size,
        }

        if format == "legacy":
            return legacy
        if format == "rich":
            return rich
        if format == "hybrid":
            payload = dict(rich)
            payload.update(legacy)
            return payload

        raise ValueError("format must be one of: 'legacy', 'rich', 'hybrid'")

    def save_pickle(self, path: str | Path, *, format: str = "hybrid") -> Path:
        output_path = Path(path)
        payload = self.to_pickle_dict(format=format)
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output_path

    @classmethod
    def load_pickle(cls, path: str | Path) -> "VoxelGrid":
        input_path = Path(path)
        with input_path.open("rb") as handle:
            payload = pickle.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("Invalid voxel-grid pickle payload.")

        if "array_zyx" in payload:
            data = np.asarray(payload["array_zyx"], dtype=np.uint8)
        elif "data" in payload:
            raw = np.asarray(payload["data"], dtype=np.uint8)
            if raw.ndim == 3:
                data = raw
            else:
                if "shape_zyx" in payload:
                    shape = tuple(int(v) for v in np.asarray(payload["shape_zyx"]).reshape(3))
                elif "num_voxels" in payload:
                    shape = tuple(int(v) for v in np.asarray(payload["num_voxels"]).reshape(3))
                else:
                    raise ValueError("Cannot infer voxel-grid shape from pickle payload.")
                data = raw.reshape(shape)
        else:
            raise ValueError("Pickle payload does not contain voxel data.")

        bounds_min_xyz = np.asarray(
            payload.get("bounds_min_xyz", payload.get("min_point")),
            dtype=np.float32,
        )
        bounds_max_xyz = np.asarray(
            payload.get("bounds_max_xyz", payload.get("max_point")),
            dtype=np.float32,
        )

        if "voxel_size_xyz" in payload:
            voxel_size_xyz = np.asarray(payload["voxel_size_xyz"], dtype=np.float32)
        else:
            legacy_voxel_size = payload.get("voxel_size")
            if legacy_voxel_size is None:
                extent = bounds_max_xyz - bounds_min_xyz
                z, y, x = data.shape
                voxel_size_xyz = extent / np.array([x, y, z], dtype=np.float32)
            else:
                voxel_size_xyz = _expand_voxel_size(legacy_voxel_size)

        return cls(
            data=data.astype(np.uint8, copy=False),
            bounds_min_xyz=bounds_min_xyz,
            bounds_max_xyz=bounds_max_xyz,
            voxel_size_xyz=voxel_size_xyz,
            mode=str(payload.get("mode", "solid")),
            pixel_size=payload.get("pixel_size"),
            metadata=dict(payload.get("metadata", {})),
        )

    def save_npz(self, path: str | Path) -> Path:
        output_path = Path(path)
        np.savez_compressed(
            output_path,
            data=self.data,
            bounds_min_xyz=self.bounds_min_xyz,
            bounds_max_xyz=self.bounds_max_xyz,
            voxel_size_xyz=self.voxel_size_xyz,
            pixel_size=np.array(np.nan if self.pixel_size is None else self.pixel_size, dtype=np.float32),
            mode=np.array(self.mode),
            metadata_json=np.array(json.dumps(_json_ready(self.metadata))),
        )
        return output_path

    @classmethod
    def load_npz(cls, path: str | Path) -> "VoxelGrid":
        payload = np.load(Path(path), allow_pickle=False)
        pixel_raw = float(payload["pixel_size"])
        return cls(
            data=np.asarray(payload["data"], dtype=np.uint8),
            bounds_min_xyz=np.asarray(payload["bounds_min_xyz"], dtype=np.float32),
            bounds_max_xyz=np.asarray(payload["bounds_max_xyz"], dtype=np.float32),
            voxel_size_xyz=np.asarray(payload["voxel_size_xyz"], dtype=np.float32),
            mode=str(payload["mode"].item()),
            pixel_size=None if np.isnan(pixel_raw) else pixel_raw,
            metadata=json.loads(str(payload["metadata_json"].item())),
        )

    def summary(self) -> str:
        return (
            f"VoxelGrid(shape_zyx={self.shape_zyx}, grid_size_xyz={self.grid_size_xyz}, "
            f"voxel_size_xyz={self.voxel_size_xyz.tolist()}, occupied_voxels={self.occupied_voxels}, "
            f"occupied_fraction={self.occupied_fraction:.4f}, mode={self.mode!r}, "
            f"pixel_size={self.pixel_size})"
        )
