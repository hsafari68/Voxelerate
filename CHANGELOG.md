# Changelog

## 0.5.6

- Fixed duplicate slice figures when plotting helpers both displayed and returned Matplotlib figures; figures are now returned only when requested or when `show=False`.
- Simplified the example script descriptions and removed workflow commentary from example source files.

## 0.5.5

- Simplified the curated examples into standalone scripts with direct path variables and no extra wrapper logic.
- Renamed the first example to `01_single_mesh_pipeline.py`.
- Updated the examples guide and README to match the simplified workflow.
- Added a regression test for the curated example scripts.

## 0.5.4

- curated the release example set down to three primary workflows
- added a full Eiffel Tower pipeline example with solid voxelization, BVH, octrees, viewer, and slice inspection
- added a shared-grid two-mesh example using `C12.STL` and `2,5 mm cut.STL`
- added a spatial-reuse example that saves and reloads BVH and octree archives for `David.stl`
- updated the bundled `models/` directory to keep only `Eiffel_Tower.stl`, `David.stl`, `C12.STL`, and `2,5 mm cut.STL`
- refreshed the README and examples guide to match the curated release workflow

## 0.5.3

- restored viewer control instructions to the window title
- improved solid voxel preview behavior with x-ray support and mesh toggling
- clarified viewer control help in the Python API and examples