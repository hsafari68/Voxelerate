from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh
from .spatial_io import FlatBVH

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]
SplitStrategy = Literal["median", "sah"]


def _surface_area(bounds_min_xyz: Float32Array, bounds_max_xyz: Float32Array) -> float:
    extent = np.maximum(bounds_max_xyz - bounds_min_xyz, 0.0)
    return float(2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0]))


def _empty_bounds() -> tuple[Float32Array, Float32Array]:
    return (
        np.array([np.inf, np.inf, np.inf], dtype=np.float32),
        np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32),
    )


@dataclass(slots=True)
class BVHNode:
    bounds_min_xyz: Float32Array
    bounds_max_xyz: Float32Array
    triangle_indices: Int32Array
    depth: int
    split_axis: int = -1
    left: "BVHNode | None" = None
    right: "BVHNode | None" = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangle_indices))


@dataclass(slots=True)
class BVH:
    mesh: Mesh
    max_leaf_size: int = 4
    strategy: SplitStrategy = "sah"
    bins: int = 16
    root: BVHNode | None = None
    _tri_bounds_min: Float32Array = field(init=False, repr=False)
    _tri_bounds_max: Float32Array = field(init=False, repr=False)
    _tri_centroids: Float32Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        triangles = self.mesh.vertices[self.mesh.faces]
        self._tri_bounds_min = triangles.min(axis=1).astype(np.float32)
        self._tri_bounds_max = triangles.max(axis=1).astype(np.float32)
        self._tri_centroids = (0.5 * (self._tri_bounds_min + self._tri_bounds_max)).astype(np.float32)

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
        *,
        max_leaf_size: int = 4,
        strategy: SplitStrategy = "sah",
        bins: int = 16,
    ) -> "BVH":
        bvh = cls(mesh=mesh, max_leaf_size=max_leaf_size, strategy=strategy, bins=bins)
        bvh.build()
        return bvh

    def build(self) -> BVHNode:
        indices = np.arange(self.mesh.triangle_count, dtype=np.int32)
        self.root = self._build_recursive(indices, depth=0)
        return self.root

    def _build_recursive(self, indices: Int32Array, depth: int) -> BVHNode:
        bounds_min = self._tri_bounds_min[indices].min(axis=0).astype(np.float32)
        bounds_max = self._tri_bounds_max[indices].max(axis=0).astype(np.float32)
        node = BVHNode(
            bounds_min_xyz=bounds_min,
            bounds_max_xyz=bounds_max,
            triangle_indices=np.ascontiguousarray(indices, dtype=np.int32),
            depth=depth,
        )

        if len(indices) <= self.max_leaf_size:
            return node

        if self.strategy == "sah":
            axis, left_indices, right_indices = self._split_sah(indices, bounds_min, bounds_max)
            if len(left_indices) == 0 or len(right_indices) == 0:
                axis, left_indices, right_indices = self._split_median(indices, bounds_min, bounds_max)
        else:
            axis, left_indices, right_indices = self._split_median(indices, bounds_min, bounds_max)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return node

        node.split_axis = axis
        node.left = self._build_recursive(left_indices, depth + 1)
        node.right = self._build_recursive(right_indices, depth + 1)
        return node

    def _split_median(
        self,
        indices: Int32Array,
        bounds_min: Float32Array,
        bounds_max: Float32Array,
    ) -> tuple[int, Int32Array, Int32Array]:
        extent = bounds_max - bounds_min
        axis = int(np.argmax(extent))
        order = np.argsort(self._tri_centroids[indices, axis], kind="mergesort")
        sorted_indices = indices[order]
        mid = len(sorted_indices) // 2
        return axis, sorted_indices[:mid], sorted_indices[mid:]

    def _split_sah(
        self,
        indices: Int32Array,
        bounds_min: Float32Array,
        bounds_max: Float32Array,
    ) -> tuple[int, Int32Array, Int32Array]:
        best_axis = -1
        best_split_bin = -1
        best_cost = float("inf")
        axis_extent = bounds_max - bounds_min

        if np.all(axis_extent < 1e-12):
            return -1, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        for axis in range(3):
            extent = float(axis_extent[axis])
            if extent < 1e-12:
                continue

            bin_counts = np.zeros(self.bins, dtype=np.int32)
            bin_min = np.full((self.bins, 3), np.inf, dtype=np.float32)
            bin_max = np.full((self.bins, 3), -np.inf, dtype=np.float32)

            centers = self._tri_centroids[indices, axis]
            rel = (centers - bounds_min[axis]) / max(extent, 1e-12)
            bin_ids = np.clip((rel * self.bins).astype(np.int32), 0, self.bins - 1)

            tri_min = self._tri_bounds_min[indices]
            tri_max = self._tri_bounds_max[indices]

            for local_idx, bin_id in enumerate(bin_ids.tolist()):
                bin_counts[bin_id] += 1
                bin_min[bin_id] = np.minimum(bin_min[bin_id], tri_min[local_idx])
                bin_max[bin_id] = np.maximum(bin_max[bin_id], tri_max[local_idx])

            left_counts = np.zeros(self.bins - 1, dtype=np.int32)
            right_counts = np.zeros(self.bins - 1, dtype=np.int32)
            left_area = np.zeros(self.bins - 1, dtype=np.float32)
            right_area = np.zeros(self.bins - 1, dtype=np.float32)

            run_min, run_max = _empty_bounds()
            count = 0
            for i in range(self.bins - 1):
                count += int(bin_counts[i])
                if bin_counts[i] > 0:
                    run_min = np.minimum(run_min, bin_min[i])
                    run_max = np.maximum(run_max, bin_max[i])
                left_counts[i] = count
                if count > 0:
                    left_area[i] = _surface_area(run_min, run_max)

            run_min, run_max = _empty_bounds()
            count = 0
            for i in range(self.bins - 1, 0, -1):
                count += int(bin_counts[i])
                if bin_counts[i] > 0:
                    run_min = np.minimum(run_min, bin_min[i])
                    run_max = np.maximum(run_max, bin_max[i])
                right_counts[i - 1] = count
                if count > 0:
                    right_area[i - 1] = _surface_area(run_min, run_max)

            sah_cost = left_counts * left_area + right_counts * right_area
            if sah_cost.size == 0:
                continue
            local_best = int(np.argmin(sah_cost))
            if left_counts[local_best] == 0 or right_counts[local_best] == 0:
                continue
            cost = float(sah_cost[local_best])
            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                best_split_bin = local_best

        if best_axis == -1:
            return -1, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        extent = float(axis_extent[best_axis])
        centers = self._tri_centroids[indices, best_axis]
        rel = (centers - bounds_min[best_axis]) / max(extent, 1e-12)
        bin_ids = np.clip((rel * self.bins).astype(np.int32), 0, self.bins - 1)

        left_mask = bin_ids <= best_split_bin
        right_mask = ~left_mask
        return best_axis, indices[left_mask], indices[right_mask]

    def collect_boxes(
        self,
        *,
        max_depth: int | None = None,
        leaves_only: bool = False,
    ) -> list[tuple[Float32Array, Float32Array, int]]:
        if self.root is None:
            self.build()

        assert self.root is not None
        boxes: list[tuple[Float32Array, Float32Array, int]] = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            if max_depth is not None and node.depth > max_depth:
                continue
            if not leaves_only or node.is_leaf():
                boxes.append((node.bounds_min_xyz, node.bounds_max_xyz, node.depth))
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

        return boxes

    def flatten(self) -> tuple[np.ndarray, Int32Array]:
        if self.root is None:
            self.build()
        assert self.root is not None

        node_dtype = np.dtype(
            [
                ("bbox_min_xyz", np.float32, 3),
                ("bbox_max_xyz", np.float32, 3),
                ("left", np.int32),
                ("right", np.int32),
                ("first_triangle", np.int32),
                ("triangle_count", np.int32),
                ("depth", np.int32),
                ("split_axis", np.int32),
            ]
        )

        nodes: list[np.ndarray] = []
        triangle_stream: list[int] = []

        def recurse(node: BVHNode) -> int:
            index = len(nodes)
            entry = np.zeros((), dtype=node_dtype)
            nodes.append(entry)

            entry["bbox_min_xyz"] = node.bounds_min_xyz
            entry["bbox_max_xyz"] = node.bounds_max_xyz
            entry["depth"] = node.depth
            entry["split_axis"] = node.split_axis

            if node.is_leaf():
                entry["left"] = -1
                entry["right"] = -1
                entry["first_triangle"] = len(triangle_stream)
                entry["triangle_count"] = node.triangle_count
                triangle_stream.extend(int(i) for i in node.triangle_indices.tolist())
            else:
                entry["first_triangle"] = -1
                entry["triangle_count"] = 0
                entry["left"] = recurse(node.left) if node.left is not None else -1
                entry["right"] = recurse(node.right) if node.right is not None else -1

            return index

        recurse(self.root)
        return np.asarray(nodes, dtype=node_dtype), np.asarray(triangle_stream, dtype=np.int32)

    def to_flat(self) -> FlatBVH:
        nodes, triangle_indices = self.flatten()
        metadata = {
            "strategy": self.strategy,
            "max_leaf_size": int(self.max_leaf_size),
            "bins": int(self.bins),
            "mesh_name": self.mesh.name,
            "source_mesh": str(self.mesh.source_path) if self.mesh.source_path is not None else None,
            "triangle_count": int(self.mesh.triangle_count),
        }
        return FlatBVH(nodes=nodes, triangle_indices=triangle_indices, metadata=metadata)

    def save_npz(self, path: str | Path) -> Path:
        return self.to_flat().save_npz(path)

    def summary(self) -> str:
        if self.root is None:
            self.build()
        assert self.root is not None
        boxes = self.collect_boxes()
        leaf_boxes = self.collect_boxes(leaves_only=True)
        max_depth = max(depth for _, _, depth in boxes) if boxes else 0
        return (
            f"BVH(strategy={self.strategy!r}, triangles={self.mesh.triangle_count}, nodes={len(boxes)}, "
            f"leaves={len(leaf_boxes)}, max_depth={max_depth}, max_leaf_size={self.max_leaf_size})"
        )
