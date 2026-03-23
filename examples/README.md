# Voxelerate examples

The curated example set focuses on three core workflows:

- `01_single_mesh_pipeline.py` — load `models/Eiffel_Tower.stl`, build a solid voxel grid, construct a BVH and mesh octree, visualize the mesh / voxel grid / hierarchies, save the voxel grid, reload it, and inspect central slices with a voxel-octree overlay.
- `02_shared_grid_two_meshes.py` — voxelize `models/C12.STL` and `models/2,5 mm cut.STL` on the same explicit grid defined by the larger outer mesh, visualize both results, and compare their central slices.
- `03_spatial_reuse.py` — build BVH and octree archives for `models/David.stl`, save them, reload them later, then visualize the reloaded structures and slice overlays.

Typical usage from the repository root:

```bash
python examples/01_single_mesh_pipeline.py
python examples/02_shared_grid_two_meshes.py
python examples/03_spatial_reuse.py
```

Update the path variables at the top of each example if you want to use your own meshes.
