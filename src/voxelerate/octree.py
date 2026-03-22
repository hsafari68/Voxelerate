from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh
from .spatial_io import FlatMeshOctree, FlatVoxelOctree
from .voxel_grid import VoxelGrid

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]


@dataclass(slots=True)
class OctreeNode:
    bounds_min_xyz: Float32Array
    bounds_max_xyz: Float32Array
    depth: int
    children: list["OctreeNode"] = field(default_factory=list)
    triangle_indices: Int32Array | None = None
    value: int | None = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass(slots=True)
class MeshOctree:
    mesh: Mesh
    max_depth: int = 8
    max_triangles: int = 32
    root: OctreeNode | None = None
    space: str = "mesh"
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
        max_depth: int = 8,
        max_triangles: int = 32,
    ) -> "MeshOctree":
        tree = cls(mesh=mesh, max_depth=max_depth, max_triangles=max_triangles)
        tree.build()
        return tree

    def build(self) -> OctreeNode:
        bounds_min_xyz, bounds_max_xyz = self.mesh.bounds
        indices = np.arange(self.mesh.triangle_count, dtype=np.int32)
        self.root = self._build_recursive(bounds_min_xyz, bounds_max_xyz, indices, depth=0)
        return self.root

    def _build_recursive(
        self,
        bounds_min_xyz: Float32Array,
        bounds_max_xyz: Float32Array,
        indices: Int32Array,
        depth: int,
    ) -> OctreeNode:
        node = OctreeNode(
            bounds_min_xyz=bounds_min_xyz.astype(np.float32),
            bounds_max_xyz=bounds_max_xyz.astype(np.float32),
            depth=depth,
            triangle_indices=np.ascontiguousarray(indices, dtype=np.int32),
        )

        if len(indices) <= self.max_triangles or depth >= self.max_depth:
            return node

        center = (bounds_min_xyz + bounds_max_xyz) * 0.5
        child_groups: list[list[int]] = [[] for _ in range(8)]

        tri_centers = self._tri_centroids[indices]
        child_bits = (tri_centers >= center).astype(np.int32)
        child_ids = child_bits[:, 0] | (child_bits[:, 1] << 1) | (child_bits[:, 2] << 2)

        for tri_index, child_id in zip(indices.tolist(), child_ids.tolist()):
            child_groups[int(child_id)].append(int(tri_index))

        if max(len(group) for group in child_groups) == len(indices):
            return node

        children: list[OctreeNode] = []
        for child_id, tri_list in enumerate(child_groups):
            if not tri_list:
                continue

            child_min = bounds_min_xyz.copy()
            child_max = bounds_max_xyz.copy()

            if child_id & 1:
                child_min[0] = center[0]
            else:
                child_max[0] = center[0]

            if child_id & 2:
                child_min[1] = center[1]
            else:
                child_max[1] = center[1]

            if child_id & 4:
                child_min[2] = center[2]
            else:
                child_max[2] = center[2]

            children.append(
                self._build_recursive(
                    child_min.astype(np.float32),
                    child_max.astype(np.float32),
                    np.asarray(tri_list, dtype=np.int32),
                    depth + 1,
                )
            )

        if not children:
            return node

        node.children = children
        return node

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
            stack.extend(reversed(node.children))

        return boxes

    def flatten(self) -> tuple[np.ndarray, Int32Array]:
        if self.root is None:
            self.build()
        assert self.root is not None

        node_dtype = np.dtype(
            [
                ("bbox_min_xyz", np.float32, 3),
                ("bbox_max_xyz", np.float32, 3),
                ("children", np.int32, 8),
                ("first_triangle", np.int32),
                ("triangle_count", np.int32),
                ("depth", np.int32),
            ]
        )

        nodes: list[np.ndarray] = []
        triangle_stream: list[int] = []

        def recurse(node: OctreeNode) -> int:
            index = len(nodes)
            entry = np.zeros((), dtype=node_dtype)
            nodes.append(entry)

            entry["bbox_min_xyz"] = node.bounds_min_xyz
            entry["bbox_max_xyz"] = node.bounds_max_xyz
            entry["depth"] = node.depth
            entry["children"][:] = -1

            if node.is_leaf():
                triangles = np.empty(0, dtype=np.int32) if node.triangle_indices is None else node.triangle_indices
                entry["first_triangle"] = len(triangle_stream)
                entry["triangle_count"] = int(len(triangles))
                triangle_stream.extend(int(index_value) for index_value in triangles.tolist())
            else:
                entry["first_triangle"] = -1
                entry["triangle_count"] = 0
                for child_idx, child in enumerate(node.children[:8]):
                    entry["children"][child_idx] = recurse(child)

            return index

        recurse(self.root)
        return np.asarray(nodes, dtype=node_dtype), np.asarray(triangle_stream, dtype=np.int32)

    def to_flat(self) -> FlatMeshOctree:
        nodes, triangle_indices = self.flatten()
        metadata = {
            "mesh_name": self.mesh.name,
            "source_mesh": str(self.mesh.source_path) if self.mesh.source_path is not None else None,
            "triangle_count": int(self.mesh.triangle_count),
            "max_depth": int(self.max_depth),
            "max_triangles": int(self.max_triangles),
        }
        return FlatMeshOctree(nodes=nodes, triangle_indices=triangle_indices, metadata=metadata)

    def save_npz(self, path: str | Path) -> Path:
        return self.to_flat().save_npz(path)

    def summary(self) -> str:
        if self.root is None:
            self.build()
        boxes = self.collect_boxes()
        leaf_boxes = self.collect_boxes(leaves_only=True)
        max_depth = max(depth for _, _, depth in boxes) if boxes else 0
        return (
            f"MeshOctree(triangles={self.mesh.triangle_count}, nodes={len(boxes)}, "
            f"leaves={len(leaf_boxes)}, max_depth={max_depth}, max_triangles={self.max_triangles})"
        )


@dataclass(slots=True)
class VoxelOctree:
    grid: VoxelGrid
    min_dim: int = 1
    max_depth: int | None = None
    root: OctreeNode | None = None
    space: str = "voxel"

    @classmethod
    def from_grid(
        cls,
        grid: VoxelGrid,
        *,
        min_dim: int = 1,
        max_depth: int | None = None,
    ) -> "VoxelOctree":
        tree = cls(grid=grid, min_dim=min_dim, max_depth=max_depth)
        tree.build()
        return tree

    def build(self) -> OctreeNode:
        z, y, x = self.grid.shape_zyx
        self.root = self._build_recursive(
            z0=0,
            y0=0,
            x0=0,
            dz=z,
            dy=y,
            dx=x,
            bounds_min_xyz=self.grid.bounds_min_xyz,
            bounds_max_xyz=self.grid.bounds_max_xyz,
            depth=0,
        )
        return self.root

    def _build_recursive(
        self,
        *,
        z0: int,
        y0: int,
        x0: int,
        dz: int,
        dy: int,
        dx: int,
        bounds_min_xyz: Float32Array,
        bounds_max_xyz: Float32Array,
        depth: int,
    ) -> OctreeNode:
        node = OctreeNode(
            bounds_min_xyz=np.asarray(bounds_min_xyz, dtype=np.float32),
            bounds_max_xyz=np.asarray(bounds_max_xyz, dtype=np.float32),
            depth=depth,
        )

        sub_volume = self.grid.data[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx]
        min_value = int(sub_volume.min())
        max_value = int(sub_volume.max())

        if min_value == max_value:
            node.value = min_value
            return node

        if min(dz, dy, dx) <= self.min_dim:
            node.value = int(round(float(sub_volume.mean())))
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            node.value = int(round(float(sub_volume.mean())))
            return node

        midz = dz // 2
        midy = dy // 2
        midx = dx // 2

        if min(midz, midy, midx) == 0:
            node.value = int(round(float(sub_volume.mean())))
            return node

        child_dims = [
            (0,      0,      0,      midz,      midy,      midx),
            (0,      0,      midx,   midz,      midy,      dx - midx),
            (0,      midy,   0,      midz,      dy - midy, midx),
            (0,      midy,   midx,   midz,      dy - midy, dx - midx),
            (midz,   0,      0,      dz - midz, midy,      midx),
            (midz,   0,      midx,   dz - midz, midy,      dx - midx),
            (midz,   midy,   0,      dz - midz, dy - midy, midx),
            (midz,   midy,   midx,   dz - midz, dy - midy, dx - midx),
        ]

        center = (bounds_min_xyz + bounds_max_xyz) * 0.5
        bx0, by0, bz0 = bounds_min_xyz
        bx1, by1, bz1 = bounds_max_xyz
        cx, cy, cz = center

        child_bounds = [
            (np.array([bx0, by0, bz0], dtype=np.float32), np.array([cx,  cy,  cz], dtype=np.float32)),
            (np.array([cx,  by0, bz0], dtype=np.float32), np.array([bx1, cy,  cz], dtype=np.float32)),
            (np.array([bx0, cy,  bz0], dtype=np.float32), np.array([cx,  by1, cz], dtype=np.float32)),
            (np.array([cx,  cy,  bz0], dtype=np.float32), np.array([bx1, by1, cz], dtype=np.float32)),
            (np.array([bx0, by0, cz], dtype=np.float32), np.array([cx,  cy,  bz1], dtype=np.float32)),
            (np.array([cx,  by0, cz], dtype=np.float32), np.array([bx1, cy,  bz1], dtype=np.float32)),
            (np.array([bx0, cy,  cz], dtype=np.float32), np.array([cx,  by1, bz1], dtype=np.float32)),
            (np.array([cx,  cy,  cz], dtype=np.float32), np.array([bx1, by1, bz1], dtype=np.float32)),
        ]

        children: list[OctreeNode] = []
        for (oz, oy, ox, cdz, cdy, cdx), (child_min, child_max) in zip(child_dims, child_bounds):
            if cdz <= 0 or cdy <= 0 or cdx <= 0:
                continue
            children.append(
                self._build_recursive(
                    z0=z0 + oz,
                    y0=y0 + oy,
                    x0=x0 + ox,
                    dz=cdz,
                    dy=cdy,
                    dx=cdx,
                    bounds_min_xyz=child_min,
                    bounds_max_xyz=child_max,
                    depth=depth + 1,
                )
            )

        if not children:
            node.value = int(round(float(sub_volume.mean())))
            return node

        node.children = children
        return node

    def collect_boxes(
        self,
        *,
        max_depth: int | None = None,
        leaves_only: bool = False,
        occupied_only: bool = False,
    ) -> list[tuple[Float32Array, Float32Array, int, int | None]]:
        if self.root is None:
            self.build()
        assert self.root is not None

        boxes: list[tuple[Float32Array, Float32Array, int, int | None]] = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            if max_depth is not None and node.depth > max_depth:
                continue
            if leaves_only and not node.is_leaf():
                stack.extend(reversed(node.children))
                continue
            if occupied_only and node.value == 0:
                stack.extend(reversed(node.children))
                continue
            boxes.append((node.bounds_min_xyz, node.bounds_max_xyz, node.depth, node.value))
            stack.extend(reversed(node.children))

        return boxes

    def flatten(self) -> np.ndarray:
        if self.root is None:
            self.build()
        assert self.root is not None

        node_dtype = np.dtype(
            [
                ("bbox_min_xyz", np.float32, 3),
                ("bbox_mid_xyz", np.float32, 3),
                ("bbox_max_xyz", np.float32, 3),
                ("children", np.int32, 8),
                ("value", np.int32),
                ("parent", np.int32),
                ("depth", np.int32),
            ]
        )

        nodes: list[np.ndarray] = []

        def recurse(node: OctreeNode, parent_index: int) -> int:
            index = len(nodes)
            entry = np.zeros((), dtype=node_dtype)
            nodes.append(entry)

            entry["bbox_min_xyz"] = node.bounds_min_xyz
            entry["bbox_mid_xyz"] = (node.bounds_min_xyz + node.bounds_max_xyz) * 0.5
            entry["bbox_max_xyz"] = node.bounds_max_xyz
            entry["value"] = -1 if node.value is None else int(node.value)
            entry["parent"] = parent_index
            entry["depth"] = node.depth
            entry["children"][:] = -1

            for child_idx, child in enumerate(node.children[:8]):
                entry["children"][child_idx] = recurse(child, index)

            return index

        recurse(self.root, -1)
        return np.asarray(nodes, dtype=node_dtype)

    def to_flat(self) -> FlatVoxelOctree:
        nodes = self.flatten()
        metadata = {
            "grid_shape_zyx": list(self.grid.shape_zyx),
            "bounds_min_xyz": self.grid.bounds_min_xyz.tolist(),
            "bounds_max_xyz": self.grid.bounds_max_xyz.tolist(),
            "voxel_size_xyz": self.grid.voxel_size_xyz.tolist(),
            "min_dim": int(self.min_dim),
            "max_depth": None if self.max_depth is None else int(self.max_depth),
        }
        return FlatVoxelOctree(nodes=nodes, metadata=metadata)

    def save_npz(self, path: str | Path) -> Path:
        return self.to_flat().save_npz(path)

    def summary(self) -> str:
        if self.root is None:
            self.build()
        boxes = self.collect_boxes()
        leaves = self.collect_boxes(leaves_only=True)
        occupied_leaves = self.collect_boxes(leaves_only=True, occupied_only=True)
        max_depth = max(depth for _, _, depth, _ in boxes) if boxes else 0
        return (
            f"VoxelOctree(grid_shape_zyx={self.grid.shape_zyx}, nodes={len(boxes)}, leaves={len(leaves)}, "
            f"occupied_leaves={len(occupied_leaves)}, max_depth={max_depth}, min_dim={self.min_dim})"
        )
