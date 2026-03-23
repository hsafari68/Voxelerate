# Architecture

## Goals

Voxelerate is split into clear layers so that mesh loading, visualization, voxelization, slice inspection, spatial structures, and serialization are reusable independently.

## Main layers

### `mesh.py`
Owns mesh ingestion and lightweight mesh representation.

Responsibilities:
- load OBJ / STL
- triangulate polygons
- compute smooth vertex normals
- expose mesh bounds and flattened triangle buffers

### `grid.py`
Defines the explicit voxelization domain.

Responsibilities:
- build a voxel grid from mesh bounds
- build a voxel grid from explicit bounds + grid size
- keep axis conventions explicit
- preserve optional `pixel_size` metadata

### `math3d.py`
Owns transform helpers.

Responsibilities:
- create translation / rotation / scale matrices
- compose transform matrices from explicit TRS inputs
- resolve a final transform from an existing matrix plus optional TRS overrides

### `voxelizer.py`
Owns headless GPU voxelization.

Responsibilities:
- create an offscreen OpenGL context
- compile compute shaders
- dispatch surface or solid voxelization
- return a `VoxelGrid`
- record the applied transform in voxel-grid metadata

### `voxel_grid.py`
Owns saved dense voxel data.

Responsibilities:
- store voxel arrays in `(z, y, x)`
- expose world bounds in `(x, y, z)`
- save and load hybrid / legacy pickle payloads
- save and load compressed NPZ payloads
- keep metadata JSON-safe for NPZ export

### `viewer.py`
Owns interactive rendering only.

Responsibilities:
- visualize meshes
- visualize voxelized geometry
- overlay BVH / octree boxes
- keep preview transforms separate from voxelization state
- provide always-available preview controls for mouse, arrows, and plus/minus scaling
- expose the control summary in the window title and through `viewer_controls_help()`
- use an x-ray point pass for solid voxel previews so dense interiors are not visually hidden by the shell

### `bvh.py`
Owns triangle hierarchy construction.

Responsibilities:
- build median-split or SAH BVH
- expose box collections for visualization
- flatten the tree for downstream GPU usage
- export a reusable flat archive

### `octree.py`
Owns mesh and voxel octrees.

Responsibilities:
- build mesh-space octrees from triangle centroids
- build voxel-space octrees from dense occupancy grids
- flatten octrees for downstream traversal
- export reusable flat archives

### `spatial_io.py`
Owns compact persistence for expensive acceleration structures.

Responsibilities:
- save and load flattened BVH archives
- save and load flattened mesh-octree archives
- save and load flattened voxel-octree archives
- keep the loaded structures lightweight but still viewable through `collect_boxes()`

### `slices.py`
Owns 2D inspection utilities for dense voxel volumes.

Responsibilities:
- extract central X / Y / Z slices
- convert voxel volumes to display-ready 2D arrays
- overlay BVH / octree boxes on slice planes
- use a styled dark matplotlib presentation with light occupied voxels and red hierarchy overlays by default
- optionally save figures through matplotlib

## Coordinate contract

Voxelerate uses one explicit contract everywhere:

- world-space vectors: `(x, y, z)`
- OpenGL 3D texture dimensions: `(x, y, z)`
- dense NumPy volume arrays: `(z, y, x)`

This avoids the hidden axis swaps present in many research prototypes.

## Serialization strategy

There are now two categories of saved data.

### 1. Dense voxel data
The default **hybrid pickle** stores both:
- legacy keys like `data`, `num_voxels`, `min_point`, `max_point`
- richer keys like `array_zyx`, `grid_size_xyz`, `voxel_size_xyz`, `mode`, and `metadata`

This preserves compatibility with older research scripts.

### 2. Spatial hierarchies
BVH and octrees are saved as **flattened NPZ archives** rather than Python object pickles.

That is deliberate because flattened node arrays are:
- smaller
- faster to reload
- easier to move toward GPU traversal later
- safer than serializing recursive Python objects directly

## Why the viewer is separate from the voxelizer

The voxelizer is useful headlessly. The viewer is optional.

That separation makes it easier to:
- run voxelization in batch jobs
- load saved volumes later without rendering
- inspect saved spatial structures without rebuilding them
- build project-specific layers, such as volumetric printing pipelines, on top of the core repository

## Why `show()` stays central

The repository keeps a single high-level `show()` entry point because it matches the way the prototype was already used.

What changed is the transform contract:
- `show()` is for **inspection and preview**
- viewer mouse and keyboard transforms are **display-only**
- voxelization transforms come only from explicit `transform`, `translation_xyz`, `rotation_degrees_xyz`, `scale_xyz`, and `pivot_xyz` inputs passed into `voxelize(...)` or `voxelize_on_grid(...)`

This keeps the workflow simple while separating render state from voxelization state.

## Repository assets

A top-level `models/` folder is included for local reference meshes and examples. It now contains the curated demo set used throughout the repository: `David.stl`, `Eiffel_Tower.stl`, `C12.STL`, and `2,5 mm cut.STL`.
