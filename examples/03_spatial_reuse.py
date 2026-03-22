"""Spatial-structure reuse workflow: build BVH and octree archives, save them, reload them later, and visualize the reloaded structures together with voxel slices."""

from voxelerate import (
    build_bvh,
    build_mesh_octree,
    build_voxel_octree,
    load_bvh,
    load_mesh,
    load_mesh_octree,
    load_voxel_grid,
    load_voxel_octree,
    plot_central_slices,
    show,
    voxelize,
)


path = "models/David.stl"
bvh_path = "david.bvh.npz"
mesh_octree_path = "david_mesh_octree.npz"
voxel_grid_path = "david_surface.pkl"
voxel_octree_path = "david_voxel_octree.npz"

mesh = load_mesh(path)
print(mesh.summary())

print("Building reusable BVH and mesh octree archives ...")
bvh = build_bvh(mesh, strategy="median", max_leaf_size=8)
mesh_octree = build_mesh_octree(mesh, max_depth=6, max_triangles=64)

bvh.save_npz(bvh_path)
mesh_octree.save_npz(mesh_octree_path)
print(f"Saved BVH archive to: {bvh_path}")
print(f"Saved mesh octree archive to: {mesh_octree_path}")

grid = voxelize(mesh, voxel_size=0.5, mode="surface", pixel_size=0.015)
grid.save_pickle(voxel_grid_path, format="hybrid")
print(f"Saved voxel grid to: {voxel_grid_path}")

voxel_octree = build_voxel_octree(grid, max_depth=7)
voxel_octree.save_npz(voxel_octree_path)
print(f"Saved voxel octree archive to: {voxel_octree_path}")

flat_bvh = load_bvh(bvh_path)
flat_mesh_octree = load_mesh_octree(mesh_octree_path)
reloaded_grid = load_voxel_grid(voxel_grid_path)
flat_voxel_octree = load_voxel_octree(voxel_octree_path)

print(flat_bvh.summary())
print(flat_mesh_octree.summary())
print(flat_voxel_octree.summary())
print(reloaded_grid.summary())

show(
    mesh,
    voxel_grid=reloaded_grid,
    bvh=flat_bvh,
    octree=flat_mesh_octree,
    title="Voxelerate Viewer | reloaded spatial structures",
)
plot_central_slices(
    reloaded_grid,
    octree=flat_voxel_octree,
    max_depth=5,
    title="Spatial reuse | reloaded voxel-octree slices",
)
