# Voxelerate

**Voxelerate** is a GPU-first toolkit for loading, visualizing, voxelizing, and serializing 3D triangle meshes.
It is designed for workflows where a mesh must be inspected in 3D, converted into a dense voxel volume, optionally
accelerated with BVH or octree structures, and saved for reuse in later pipelines such as volumetric 3D printing,
simulation, geometry processing, or analysis.

> Project figure placeholder: add a hero image later at `docs/assets/hero.png` and reference it near the top of this README.

## Overview

Voxelerate is built around a practical geometry workflow:

- load **OBJ** and **STL** meshes
- inspect meshes in a real-time GPU viewer
- voxelize with **surface** or **solid** modes
- work from either **automatic bounds** or an **explicit external grid**
- build and reuse **BVH** and **octree** hierarchies
- inspect voxel data with clean **2D slice visualizations**
- save voxel grids and spatial structures for later use

A core design rule in Voxelerate is that viewer interaction is **preview-only**. Camera and preview transforms in the
window never change the geometry that gets voxelized. The only transforms that affect voxelization are the ones you
pass explicitly as inputs.

## Highlights

- **Mesh I/O** for OBJ and STL
- **GPU viewer** for meshes, voxel grids, BVHs, and octrees
- **GPU voxelization** using OpenGL compute shaders
- **Surface** and **solid** voxelization modes
- **Automatic** or **explicit** voxelization grids
- **BVH** with median and SAH splitting
- **Mesh octree** and **voxel octree** generation
- **Pickle** and **NPZ** serialization
- **2D slice plots** with optional hierarchy overlays
- **Reusable saved hierarchies** for expensive large-mesh preprocessing

## Installation

Install the core package:

```bash
pip install -e .
```

Install with Open3D adapters:

```bash
pip install -e ".[open3d]"
```

Install with matplotlib slice plotting support:

```bash
pip install -e ".[viz]"
```

Install development tools:

```bash
pip install -e ".[dev]"
```

## Runtime requirements

Voxelerate uses desktop OpenGL through GLFW and PyOpenGL.

- **Viewer:** OpenGL 3.3+
- **Voxelizer:** OpenGL 4.3+ for compute shaders

## Core conventions

These conventions are explicit throughout the repository.

- world-space vectors use **`(x, y, z)`**
- grid sizes use **`(x, y, z)`**
- dense voxel arrays use **`(z, y, x)`**
- compatibility pickle files store `num_voxels` as **`(z, y, x)`**

## Quick start

```python
from voxelerate import build_voxel_octree, load_mesh, plot_central_slices, show, voxelize

mesh = load_mesh("models/Eiffel_Tower.stl")
voxel_grid = voxelize(
    mesh,
    voxel_size=0.25,
    mode="solid",
    pixel_size=0.015,
)

print(mesh.summary())
print(voxel_grid.summary())

show(mesh, voxel_grid=voxel_grid, title="Voxelerate - quick start")

voxel_octree = build_voxel_octree(voxel_grid, max_depth=8)
plot_central_slices(
    voxel_grid,
    octree=voxel_octree,
    max_depth=5,
    title="Eiffel Tower - central slices",
)

voxel_grid.save_pickle("eiffel_tower_solid.pkl", format="hybrid")
```

## Interactive viewer

The viewer is meant for inspection and presentation. The control summary is shown directly in the window title,
and the same reference is also available from Python:

```python
from voxelerate import viewer_controls_help

print(viewer_controls_help())
```

Current viewer controls:

- **Left drag:** orbit camera
- **Right drag:** pan camera
- **Mouse wheel:** zoom camera
- **Arrow keys:** rotate the displayed preview
- **Plus / minus:** scale the displayed preview while held
- **M:** toggle mesh rendering
- **B:** toggle bounding box
- **G:** toggle ground grid
- **O:** toggle BVH / octree overlays
- **V:** toggle voxel rendering
- **W:** toggle wireframe
- **X:** toggle x-ray solid-voxel preview
- **R:** reset the camera
- **T:** reset the preview transform
- **Esc:** close the viewer

Preview controls are **visual only**. They never change the voxelized result. To affect voxelization, pass an
explicit transform matrix or translation / rotation / scale inputs into `voxelize(...)` or `voxelize_on_grid(...)`.

## Why solid voxel previews can look sparser than surface previews

For a closed watertight mesh, the stored solid voxel grid should normally contain at least as many occupied voxels
as the surface grid, and usually many more. The confusing part is **preview occlusion**: when solid voxels are drawn
as opaque depth-tested points, the outer shell hides most interior voxels, especially when the mesh itself is also
rendered.

Voxelerate defaults to an **x-ray point pass** for solid voxel previews so the denser interior is visible. This changes
only the viewer, not the saved voxel grid.

## Output formats

### Voxel grids

Voxelerate can save voxel data in multiple formats.

- **Hybrid pickle**: rich metadata plus compatibility keys for older workflows
- **Legacy pickle**: compatibility-first output matching the previous key layout
- **NPZ**: compressed array-based storage

A typical hybrid pickle contains:

- voxel data as a NumPy array
- `bounds_min_xyz`
- `bounds_max_xyz`
- `grid_size_xyz`
- `voxel_size`
- `pixel_size`
- `axis_order`
- voxelization `mode`
- additional metadata

### Spatial structures

Flattened BVH and octree archives are stored as compressed **NPZ** files so they can be loaded later without rebuilding.
This is especially useful for large meshes where hierarchy construction can be expensive.

## Slice visualization

Voxelerate includes central-slice inspection for dense voxel grids.

- central **X / Y / Z** slice plotting
- optional **BVH** or **octree** overlays
- light occupied voxels with red hierarchy overlays
- save-to-image support through matplotlib
- compatible with both **surface** and **solid** voxelization results

Example:

```python
from voxelerate import build_voxel_octree, load_voxel_grid, plot_central_slices

grid = load_voxel_grid("eiffel_tower_solid.pkl")
octree = build_voxel_octree(grid, max_depth=8)

plot_central_slices(
    grid,
    octree=octree,
    max_depth=5,
    title="Central voxel slices with octree overlay",
)
```

## Included models

The repository intentionally ships with a small curated set of demo assets under `models/`.

- `Eiffel_Tower.stl`
- `David.stl`
- `C12.STL`
- `2,5 mm cut.STL`

## Examples

The `examples/` folder is intentionally small and centered on the main workflows.
Every example includes visualization.

- `01_single_mesh_pipeline.py` — load a single mesh, build a solid voxel grid, create BVH and octrees,
  visualize the mesh / grid / hierarchies, save the voxel grid, then inspect slices.
- `02_shared_grid_two_meshes.py` — voxelize two meshes on the same explicit grid defined by the larger
  outer mesh, visualize both results, and compare their slice plots.
- `03_spatial_reuse.py` — save BVH and octree archives for later reuse, reload them, then visualize
  the reloaded structures and slice overlays.

See `examples/README.md` for the curated example workflows and run instructions.

## Command-line interface

Open a mesh viewer:

```bash
voxelerate view models/Eiffel_Tower.stl
```

Voxelize a mesh using automatic bounds:

```bash
voxelerate voxelize models/Eiffel_Tower.stl \
  --voxel-size 0.25 \
  --mode solid \
  --pixel-size 0.015 \
  --output eiffel_tower_solid.pkl
```

Voxelize with explicit transform vectors:

```bash
voxelerate voxelize models/David.stl \
  --voxel-size 0.5 \
  --translate 0.0 0.0 0.0 \
  --rotate 0.0 25.0 0.0 \
  --scale 1.0 \
  --mode surface \
  --output david_surface_transformed.pkl
```

Voxelize on an explicit grid:

```bash
voxelerate voxelize models/C12.STL \
  --grid-size 267 223 267 \
  --bounds-min 0 0 0 \
  --bounds-max 12 10 12 \
  --mode solid \
  --output c12_shared_grid.pkl
```

Inspect slices from a saved voxel grid:

```bash
voxelerate slices eiffel_tower_solid.pkl --show-octree
```

Build and save a BVH:

```bash
voxelerate bvh models/David.stl --strategy median --max-leaf-size 8 --save david.bvh.npz
```

Build and save a voxel octree:

```bash
voxelerate octree eiffel_tower_solid.pkl --max-depth 8 --save eiffel_tower_octree.npz
```

## Notes for large meshes

For meshes with hundreds of thousands or millions of triangles:

- BVH construction can take noticeable time in pure Python and NumPy
- `strategy="median"` is usually faster than `strategy="sah"`
- saved flattened BVH or octree archives can avoid rebuilding every run
- viewer preview transforms remain lightweight because they are display-only

## Repository layout

```text
Voxelerate/
├── docs/
├── examples/
├── models/
├── src/voxelerate/
│   ├── shaders/
│   ├── viewer.py
│   ├── voxelizer.py
│   ├── voxel_grid.py
│   ├── bvh.py
│   ├── octree.py
│   └── ...
├── tests/
├── pyproject.toml
├── LICENSE
└── README.md
```

## Citation

If you use Voxelerate in academic work, cite the software repository and the specific release or commit you used.
Once the repository is published, replace the placeholder fields below with the final author names, repository URL,
and release tag.

```bibtex
@software{voxelerate,
  title   = {Voxelerate: GPU Mesh Visualization, Voxelization, and Spatial Hierarchies for 3D Geometry},
  author  = {Author or Research Group Name},
  year    = {2026},
  version = {0.5.6},
  url     = {https://github.com/<user-or-org>/voxelerate},
  note    = {Software repository. Cite the release or commit used in your work.}
}
```

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for the full text.
