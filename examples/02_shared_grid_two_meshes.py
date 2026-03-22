"""Shared-grid workflow for two meshes: build one explicit grid from the outer mesh, voxelize both meshes on that grid, visualize the results, and compare central slices."""

import numpy as np

from voxelerate import build_voxelization_grid, load_mesh, plot_central_slices, show, voxelize_on_grid


outer_path = "models/C12.STL"
inner_path = "models/2,5 mm cut.STL"
outer_voxel_path = "c12_shared_grid.pkl"
inner_voxel_path = "cut_shared_grid.pkl"

pixel_size = 0.015
voxel_factor = 3.0
voxel_size = voxel_factor * pixel_size

outer_mesh = load_mesh(outer_path)
inner_mesh = load_mesh(inner_path)

shared_grid = build_voxelization_grid(
    outer_mesh,
    voxel_size=voxel_size,
    pixel_size=pixel_size,
)
translation_xyz = (outer_mesh.center - inner_mesh.center).astype(np.float32)

print("Outer mesh:", outer_mesh.summary())
print("Inner mesh:", inner_mesh.summary())
print(shared_grid.summary())
print("Translation applied to inner mesh for voxelization:", translation_xyz.tolist())

outer_grid = voxelize_on_grid(outer_mesh, grid=shared_grid, mode="solid")
inner_grid = voxelize_on_grid(
    inner_mesh,
    grid=shared_grid,
    mode="solid",
    translation_xyz=translation_xyz,
)

outer_grid.save_pickle(outer_voxel_path, format="hybrid")
inner_grid.save_pickle(inner_voxel_path, format="hybrid")
print(f"Saved outer grid to: {outer_voxel_path}")
print(f"Saved inner grid to: {inner_voxel_path}")

show(
    outer_mesh,
    voxel_grid=outer_grid,
    title="Voxelerate Viewer | outer mesh on shared grid",
)
show(
    inner_mesh,
    voxel_grid=inner_grid,
    translation_xyz=translation_xyz,
    title="Voxelerate Viewer | centered inner mesh on shared grid",
)

plot_central_slices(
    outer_grid,
    title="Shared grid | outer mesh slices",
)
plot_central_slices(
    inner_grid,
    title="Shared grid | centered inner mesh slices",
)
