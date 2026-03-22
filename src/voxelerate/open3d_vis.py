from __future__ import annotations

import numpy as np

from .bvh import BVH
from .mesh import Mesh
from .octree import MeshOctree, VoxelOctree
from .voxel_grid import VoxelGrid


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D is not installed. Install it with `pip install -e .[open3d]`."
        ) from exc
    return o3d


def mesh_to_open3d(mesh: Mesh):
    o3d = _require_open3d()
    geometry = o3d.geometry.TriangleMesh()
    geometry.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64))
    geometry.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
    geometry.compute_vertex_normals()
    return geometry


def boxes_to_open3d_lineset(
    boxes: list[tuple[np.ndarray, np.ndarray, int]],
    *,
    color: tuple[float, float, float] = (0.9, 0.2, 0.1),
):
    o3d = _require_open3d()

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    points = []
    lines = []
    colors = []
    offset = 0

    for bounds_min_xyz, bounds_max_xyz, _ in boxes:
        x0, y0, z0 = bounds_min_xyz.tolist()
        x1, y1, z1 = bounds_max_xyz.tolist()
        corners = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=np.float64,
        )
        points.append(corners)
        for start, end in edges:
            lines.append([offset + start, offset + end])
            colors.append(color)
        offset += 8

    points_array = np.vstack(points) if points else np.zeros((0, 3), dtype=np.float64)
    lines_array = np.asarray(lines, dtype=np.int32)
    colors_array = np.asarray(colors, dtype=np.float64)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_array)
    line_set.lines = o3d.utility.Vector2iVector(lines_array)
    line_set.colors = o3d.utility.Vector3dVector(colors_array)
    return line_set


def voxel_grid_to_open3d_point_cloud(
    grid: VoxelGrid,
    *,
    color: tuple[float, float, float] = (0.2, 0.9, 0.4),
):
    o3d = _require_open3d()
    cloud = o3d.geometry.PointCloud()
    centers = grid.occupied_centers().astype(np.float64)
    cloud.points = o3d.utility.Vector3dVector(centers)
    if len(centers) > 0:
        colors = np.tile(np.asarray(color, dtype=np.float64), (len(centers), 1))
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def show_bvh(mesh: Mesh, bvh: BVH) -> None:
    o3d = _require_open3d()
    boxes = bvh.collect_boxes()
    o3d.visualization.draw_geometries([mesh_to_open3d(mesh), boxes_to_open3d_lineset(boxes)])


def show_mesh_octree(mesh: Mesh, octree: MeshOctree) -> None:
    o3d = _require_open3d()
    boxes = octree.collect_boxes()
    o3d.visualization.draw_geometries([mesh_to_open3d(mesh), boxes_to_open3d_lineset(boxes)])


def show_voxel_octree(grid: VoxelGrid, octree: VoxelOctree) -> None:
    o3d = _require_open3d()
    boxes = [(mn, mx, depth) for mn, mx, depth, value in octree.collect_boxes(occupied_only=True) if value != 0]
    o3d.visualization.draw_geometries(
        [voxel_grid_to_open3d_point_cloud(grid), boxes_to_open3d_lineset(boxes)]
    )
