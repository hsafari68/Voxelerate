"""Single-mesh pipeline: solid voxelization, BVH and octree construction, interactive visualization, and central slice inspection."""

from voxelerate import (
    build_bvh,
    build_mesh_octree,
    build_voxel_octree,
    load_mesh,
    load_voxel_grid,
    plot_central_slices,
    show,
    voxelize,
)


path = "models/Eiffel_Tower.stl"
voxel_grid_path = "single_mesh_solid.pkl"

mesh = load_mesh(path)
grid = voxelize(mesh, voxel_size=0.25, mode="solid", pixel_size=0.015)

print(mesh.summary())
print(grid.summary())

bvh = build_bvh(mesh, strategy="sah")
mesh_octree = build_mesh_octree(mesh, max_depth=6, max_triangles=24)

grid.save_pickle(voxel_grid_path, format="hybrid")
print(f"Saved voxel grid to: {voxel_grid_path}")

show(
    mesh,
    voxel_grid=grid,
    bvh=bvh,
    octree=mesh_octree,
    title="Voxelerate Viewer | single mesh pipeline",
)

reloaded_grid = load_voxel_grid(voxel_grid_path)
voxel_octree = build_voxel_octree(reloaded_grid, max_depth=8)
plot_central_slices(
    reloaded_grid,
    octree=voxel_octree,
    max_depth=5,
    title="Single mesh | central solid voxel slices",
)
