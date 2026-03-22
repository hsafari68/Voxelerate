from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]


class SpatialArchiveError(ValueError):
    """Raised when a serialized spatial structure cannot be loaded."""


@dataclass(slots=True)
class FlatBVH:
    nodes: np.ndarray
    triangle_indices: Int32Array
    metadata: dict[str, Any] = field(default_factory=dict)
    space: str = "mesh"

    @property
    def bounds(self) -> tuple[Float32Array, Float32Array]:
        if len(self.nodes) == 0:
            raise SpatialArchiveError("FlatBVH contains no nodes.")
        return (
            np.asarray(self.nodes[0]["bbox_min_xyz"], dtype=np.float32),
            np.asarray(self.nodes[0]["bbox_max_xyz"], dtype=np.float32),
        )

    def collect_boxes(
        self,
        *,
        max_depth: int | None = None,
        leaves_only: bool = False,
    ) -> list[tuple[Float32Array, Float32Array, int]]:
        if len(self.nodes) == 0:
            return []

        boxes: list[tuple[Float32Array, Float32Array, int]] = []
        stack = [0]
        while stack:
            index = stack.pop()
            node = self.nodes[index]
            depth = int(node["depth"])
            if max_depth is not None and depth > max_depth:
                continue

            left = int(node["left"])
            right = int(node["right"])
            is_leaf = left < 0 and right < 0
            if not leaves_only or is_leaf:
                boxes.append(
                    (
                        np.asarray(node["bbox_min_xyz"], dtype=np.float32),
                        np.asarray(node["bbox_max_xyz"], dtype=np.float32),
                        depth,
                    )
                )
            if right >= 0:
                stack.append(right)
            if left >= 0:
                stack.append(left)
        return boxes

    def save_npz(self, path: str | Path) -> Path:
        output_path = Path(path)
        np.savez_compressed(
            output_path,
            format=np.array("voxelerate.flat_bvh"),
            format_version=np.array(1, dtype=np.int32),
            nodes=self.nodes,
            triangle_indices=self.triangle_indices,
            metadata_json=np.array(json.dumps(self.metadata)),
        )
        return output_path

    @classmethod
    def load_npz(cls, path: str | Path) -> "FlatBVH":
        payload = np.load(Path(path), allow_pickle=False)
        fmt = str(payload["format"].item())
        if fmt != "voxelerate.flat_bvh":
            raise SpatialArchiveError(f"Unsupported BVH archive format: {fmt!r}")
        metadata_raw = str(payload["metadata_json"].item()) if "metadata_json" in payload else "{}"
        return cls(
            nodes=np.asarray(payload["nodes"]),
            triangle_indices=np.asarray(payload["triangle_indices"], dtype=np.int32),
            metadata=json.loads(metadata_raw),
        )

    def summary(self) -> str:
        if len(self.nodes) == 0:
            return "FlatBVH(nodes=0, triangles=0)"
        boxes = self.collect_boxes()
        leaves = self.collect_boxes(leaves_only=True)
        max_depth = max(depth for _, _, depth in boxes) if boxes else 0
        triangle_count = int(self.metadata.get("triangle_count", len(self.triangle_indices)))
        return (
            f"FlatBVH(nodes={len(boxes)}, leaves={len(leaves)}, triangles={triangle_count}, "
            f"max_depth={max_depth}, strategy={self.metadata.get('strategy')!r})"
        )


@dataclass(slots=True)
class FlatMeshOctree:
    nodes: np.ndarray
    triangle_indices: Int32Array
    metadata: dict[str, Any] = field(default_factory=dict)
    space: str = "mesh"

    @property
    def bounds(self) -> tuple[Float32Array, Float32Array]:
        if len(self.nodes) == 0:
            raise SpatialArchiveError("FlatMeshOctree contains no nodes.")
        return (
            np.asarray(self.nodes[0]["bbox_min_xyz"], dtype=np.float32),
            np.asarray(self.nodes[0]["bbox_max_xyz"], dtype=np.float32),
        )

    def collect_boxes(
        self,
        *,
        max_depth: int | None = None,
        leaves_only: bool = False,
    ) -> list[tuple[Float32Array, Float32Array, int]]:
        if len(self.nodes) == 0:
            return []

        boxes: list[tuple[Float32Array, Float32Array, int]] = []
        stack = [0]
        while stack:
            index = stack.pop()
            node = self.nodes[index]
            depth = int(node["depth"])
            if max_depth is not None and depth > max_depth:
                continue

            children = np.asarray(node["children"], dtype=np.int32)
            child_indices = [int(child) for child in children.tolist() if int(child) >= 0]
            is_leaf = len(child_indices) == 0
            if not leaves_only or is_leaf:
                boxes.append(
                    (
                        np.asarray(node["bbox_min_xyz"], dtype=np.float32),
                        np.asarray(node["bbox_max_xyz"], dtype=np.float32),
                        depth,
                    )
                )
            stack.extend(reversed(child_indices))
        return boxes

    def save_npz(self, path: str | Path) -> Path:
        output_path = Path(path)
        np.savez_compressed(
            output_path,
            format=np.array("voxelerate.flat_mesh_octree"),
            format_version=np.array(1, dtype=np.int32),
            nodes=self.nodes,
            triangle_indices=self.triangle_indices,
            metadata_json=np.array(json.dumps(self.metadata)),
        )
        return output_path

    @classmethod
    def load_npz(cls, path: str | Path) -> "FlatMeshOctree":
        payload = np.load(Path(path), allow_pickle=False)
        fmt = str(payload["format"].item())
        if fmt != "voxelerate.flat_mesh_octree":
            raise SpatialArchiveError(f"Unsupported mesh-octree archive format: {fmt!r}")
        metadata_raw = str(payload["metadata_json"].item()) if "metadata_json" in payload else "{}"
        return cls(
            nodes=np.asarray(payload["nodes"]),
            triangle_indices=np.asarray(payload["triangle_indices"], dtype=np.int32),
            metadata=json.loads(metadata_raw),
        )

    def summary(self) -> str:
        if len(self.nodes) == 0:
            return "FlatMeshOctree(nodes=0, triangles=0)"
        boxes = self.collect_boxes()
        leaves = self.collect_boxes(leaves_only=True)
        max_depth = max(depth for _, _, depth in boxes) if boxes else 0
        triangle_count = int(self.metadata.get("triangle_count", len(self.triangle_indices)))
        return (
            f"FlatMeshOctree(nodes={len(boxes)}, leaves={len(leaves)}, triangles={triangle_count}, "
            f"max_depth={max_depth}, max_triangles={self.metadata.get('max_triangles')})"
        )


@dataclass(slots=True)
class FlatVoxelOctree:
    nodes: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    space: str = "voxel"

    @property
    def bounds(self) -> tuple[Float32Array, Float32Array]:
        if len(self.nodes) == 0:
            raise SpatialArchiveError("FlatVoxelOctree contains no nodes.")
        return (
            np.asarray(self.nodes[0]["bbox_min_xyz"], dtype=np.float32),
            np.asarray(self.nodes[0]["bbox_max_xyz"], dtype=np.float32),
        )

    def collect_boxes(
        self,
        *,
        max_depth: int | None = None,
        leaves_only: bool = False,
        occupied_only: bool = False,
    ) -> list[tuple[Float32Array, Float32Array, int, int | None]]:
        if len(self.nodes) == 0:
            return []

        boxes: list[tuple[Float32Array, Float32Array, int, int | None]] = []
        stack = [0]
        while stack:
            index = stack.pop()
            node = self.nodes[index]
            depth = int(node["depth"])
            if max_depth is not None and depth > max_depth:
                continue

            children = np.asarray(node["children"], dtype=np.int32)
            child_indices = [int(child) for child in children.tolist() if int(child) >= 0]
            is_leaf = len(child_indices) == 0
            value = int(node["value"])
            if leaves_only and not is_leaf:
                stack.extend(reversed(child_indices))
                continue
            if occupied_only and value == 0 and is_leaf:
                continue
            boxes.append(
                (
                    np.asarray(node["bbox_min_xyz"], dtype=np.float32),
                    np.asarray(node["bbox_max_xyz"], dtype=np.float32),
                    depth,
                    None if value < 0 else value,
                )
            )
            stack.extend(reversed(child_indices))
        return boxes

    def save_npz(self, path: str | Path) -> Path:
        output_path = Path(path)
        np.savez_compressed(
            output_path,
            format=np.array("voxelerate.flat_voxel_octree"),
            format_version=np.array(1, dtype=np.int32),
            nodes=self.nodes,
            metadata_json=np.array(json.dumps(self.metadata)),
        )
        return output_path

    @classmethod
    def load_npz(cls, path: str | Path) -> "FlatVoxelOctree":
        payload = np.load(Path(path), allow_pickle=False)
        fmt = str(payload["format"].item())
        if fmt != "voxelerate.flat_voxel_octree":
            raise SpatialArchiveError(f"Unsupported voxel-octree archive format: {fmt!r}")
        metadata_raw = str(payload["metadata_json"].item()) if "metadata_json" in payload else "{}"
        return cls(
            nodes=np.asarray(payload["nodes"]),
            metadata=json.loads(metadata_raw),
        )

    def summary(self) -> str:
        if len(self.nodes) == 0:
            return "FlatVoxelOctree(nodes=0)"
        boxes = self.collect_boxes()
        leaves = self.collect_boxes(leaves_only=True)
        occupied_leaves = self.collect_boxes(leaves_only=True, occupied_only=True)
        max_depth = max(depth for _, _, depth, _ in boxes) if boxes else 0
        return (
            f"FlatVoxelOctree(nodes={len(boxes)}, leaves={len(leaves)}, occupied_leaves={len(occupied_leaves)}, "
            f"max_depth={max_depth}, min_dim={self.metadata.get('min_dim')})"
        )


def load_bvh(path: str | Path) -> FlatBVH:
    return FlatBVH.load_npz(path)


def load_mesh_octree(path: str | Path) -> FlatMeshOctree:
    return FlatMeshOctree.load_npz(path)


def load_voxel_octree(path: str | Path) -> FlatVoxelOctree:
    return FlatVoxelOctree.load_npz(path)
