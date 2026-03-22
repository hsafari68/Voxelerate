from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .api import (
    build_bvh,
    build_mesh_octree,
    build_voxel_octree,
    load_voxel_grid,
    load_voxel_octree,
    plot_central_slices,
    show,
    voxelize,
    voxelize_on_grid,
)
from .mesh import load_mesh


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def _parse_optional_scale(values: list[float] | None) -> float | tuple[float, float, float] | None:
    if values is None:
        return None
    if len(values) == 1:
        return float(values[0])
    if len(values) == 3:
        return (float(values[0]), float(values[1]), float(values[2]))
    raise SystemExit("--scale expects either 1 value or 3 values.")


def _add_transform_arguments(parser: argparse.ArgumentParser, *, for_voxelization: bool) -> None:
    prefix = "Apply this transform during voxelization." if for_voxelization else "Set the initial preview transform."
    parser.add_argument(
        "--translate",
        nargs=3,
        type=float,
        metavar=("TX", "TY", "TZ"),
        default=None,
        help=f"{prefix} Translation in world-space (x, y, z).",
    )
    parser.add_argument(
        "--rotate",
        nargs=3,
        type=float,
        metavar=("RX", "RY", "RZ"),
        default=None,
        help=f"{prefix} Rotation in degrees around x, y, z.",
    )
    parser.add_argument(
        "--scale",
        nargs="+",
        type=float,
        metavar=("S"),
        default=None,
        help=f"{prefix} Uniform scale with 1 value, or non-uniform scale with 3 values.",
    )
    parser.add_argument(
        "--pivot",
        nargs=3,
        type=float,
        metavar=("PX", "PY", "PZ"),
        default=None,
        help=f"{prefix} Optional pivot for scale and rotation in world-space.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxelerate",
        description="GPU mesh viewer, voxelizer, and spatial-structure builder.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    view_parser = subparsers.add_parser("view", help="Open the interactive mesh viewer.")
    view_parser.add_argument("mesh", type=Path, help="Path to an OBJ or STL mesh.")
    view_parser.add_argument("--center", action="store_true", help="Center the mesh before viewing.")
    view_parser.add_argument("--bvh", action="store_true", help="Build and overlay a BVH.")
    view_parser.add_argument("--octree", action="store_true", help="Build and overlay a mesh octree.")
    _add_transform_arguments(view_parser, for_voxelization=False)
    view_parser.add_argument("--editable", action="store_true", help=argparse.SUPPRESS)
    view_parser.add_argument(
        "--print-visual-transform",
        action="store_true",
        help="Print the final preview transform matrix on exit.",
    )
    view_parser.set_defaults(func=cmd_view)

    view_grid_parser = subparsers.add_parser("view-grid", help="Open the viewer for a saved voxel grid.")
    view_grid_parser.add_argument("grid", type=Path, help="Path to a .pkl or .npz voxel grid.")
    view_grid_parser.add_argument("--show-octree", action="store_true", help="Build and overlay a voxel octree.")
    _add_transform_arguments(view_grid_parser, for_voxelization=False)
    view_grid_parser.set_defaults(func=cmd_view_grid)

    voxelize_parser = subparsers.add_parser(
        "voxelize",
        help="Voxelize a mesh and save it to pickle or npz.",
    )
    voxelize_parser.add_argument("mesh", type=Path, help="Path to an OBJ or STL mesh.")
    voxelize_parser.add_argument("--center", action="store_true", help="Center the mesh before voxelization.")
    voxelize_parser.add_argument("--mode", choices=("surface", "solid"), default="solid", help="Voxelization mode.")
    voxelize_parser.add_argument("--padding", type=float, default=0.0, help="Extra padding added to all sides.")
    voxelize_parser.add_argument("--voxel-size", type=positive_float, default=None, help="Voxel edge length in world units.")
    voxelize_parser.add_argument(
        "--grid-size",
        nargs=3,
        type=positive_int,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Explicit grid size in (x, y, z) order.",
    )
    voxelize_parser.add_argument(
        "--bounds-min",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Explicit minimum bounds in world-space (x, y, z).",
    )
    voxelize_parser.add_argument(
        "--bounds-max",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Explicit maximum bounds in world-space (x, y, z).",
    )
    voxelize_parser.add_argument("--pixel-size", type=positive_float, default=None, help="Optional source pixel size.")
    _add_transform_arguments(voxelize_parser, for_voxelization=True)
    voxelize_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to <mesh_stem>_voxels.pkl.",
    )
    voxelize_parser.add_argument(
        "--format",
        choices=("legacy", "rich", "hybrid", "npz"),
        default="hybrid",
        help="Serialization format for the saved voxel grid.",
    )
    voxelize_parser.add_argument("--show", action="store_true", help="Open the viewer after voxelization.")
    voxelize_parser.set_defaults(func=cmd_voxelize)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a saved voxel-grid file.")
    inspect_parser.add_argument("grid", type=Path, help="Path to a .pkl or .npz voxel grid.")
    inspect_parser.add_argument("--show", action="store_true", help="Open the viewer for the saved voxel grid.")
    inspect_parser.add_argument("--show-octree", action="store_true", help="Build and overlay a voxel octree.")
    inspect_parser.add_argument("--slices", action="store_true", help="Plot central slices for the saved voxel grid.")
    inspect_parser.set_defaults(func=cmd_inspect)

    slices_parser = subparsers.add_parser("slices", help="Plot the central slices of a saved voxel grid.")
    slices_parser.add_argument("grid", type=Path, help="Path to a .pkl or .npz voxel grid.")
    slices_parser.add_argument("--show-octree", action="store_true", help="Build a voxel octree and overlay it.")
    slices_parser.add_argument("--octree", type=Path, default=None, help="Optional path to a saved voxel-octree archive (.npz).")
    slices_parser.add_argument("--max-depth", type=positive_int, default=None, help="Maximum displayed octree depth.")
    slices_parser.add_argument("--leaves-only", action="store_true", help="Only draw leaf nodes in the octree overlay.")
    slices_parser.add_argument("--include-empty", action="store_true", help="Also draw empty octree nodes.")
    slices_parser.add_argument("--save", type=Path, default=None, help="Optional output image path.")
    slices_parser.set_defaults(func=cmd_slices)

    bvh_parser = subparsers.add_parser("bvh", help="Build a BVH and print a summary.")
    bvh_parser.add_argument("mesh", type=Path, help="Path to an OBJ or STL mesh.")
    bvh_parser.add_argument("--center", action="store_true", help="Center the mesh first.")
    bvh_parser.add_argument("--strategy", choices=("median", "sah"), default="sah")
    bvh_parser.add_argument("--max-leaf-size", type=positive_int, default=4)
    bvh_parser.add_argument("--bins", type=positive_int, default=16)
    bvh_parser.add_argument("--save", type=Path, default=None, help="Save the flattened BVH to an .npz archive.")
    bvh_parser.add_argument("--show", action="store_true", help="Open the viewer with the BVH overlay.")
    bvh_parser.set_defaults(func=cmd_bvh)

    octree_parser = subparsers.add_parser("octree", help="Build an octree from a mesh or voxel grid.")
    octree_parser.add_argument("source", type=Path, help="Path to an OBJ/STL mesh or a saved .pkl/.npz voxel grid.")
    octree_parser.add_argument("--center", action="store_true", help="Center a mesh source first.")
    octree_parser.add_argument("--max-depth", type=positive_int, default=8)
    octree_parser.add_argument("--max-triangles", type=positive_int, default=32)
    octree_parser.add_argument("--min-dim", type=positive_int, default=1)
    octree_parser.add_argument("--save", type=Path, default=None, help="Save the flattened octree to an .npz archive.")
    octree_parser.add_argument("--show", action="store_true", help="Open the viewer with the octree overlay.")
    octree_parser.set_defaults(func=cmd_octree)

    return parser


def _transform_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "translation_xyz": None if args.translate is None else tuple(float(v) for v in args.translate),
        "rotation_degrees_xyz": None if args.rotate is None else tuple(float(v) for v in args.rotate),
        "scale_xyz": _parse_optional_scale(args.scale),
        "pivot_xyz": None if args.pivot is None else tuple(float(v) for v in args.pivot),
    }


def cmd_view(args: argparse.Namespace) -> int:
    mesh = load_mesh(args.mesh, center=args.center)
    bvh = build_bvh(mesh) if args.bvh else None
    octree = build_mesh_octree(mesh) if args.octree else None
    final_transform = show(
        mesh,
        title=f"Voxelerate Viewer - {args.mesh.name}",
        bvh=bvh,
        octree=octree,
        return_visual_transform=args.print_visual_transform,
        **_transform_kwargs_from_args(args),
    )
    if args.print_visual_transform and final_transform is not None:
        print("Final preview transform matrix:")
        print(np.array2string(np.asarray(final_transform, dtype=np.float32), precision=5, suppress_small=True))
    return 0


def cmd_view_grid(args: argparse.Namespace) -> int:
    grid = load_voxel_grid(args.grid)
    octree = build_voxel_octree(grid) if args.show_octree else None
    show(
        None,
        voxel_grid=grid,
        octree=octree,
        title=f"Voxelerate Viewer - {args.grid.name}",
        **_transform_kwargs_from_args(args),
    )
    return 0


def cmd_voxelize(args: argparse.Namespace) -> int:
    if args.voxel_size is None and args.grid_size is None:
        raise SystemExit("Either --voxel-size or --grid-size must be provided.")

    mesh = load_mesh(args.mesh, center=args.center)
    transform_kwargs = _transform_kwargs_from_args(args)

    if args.grid_size is None and args.bounds_min is None and args.bounds_max is None:
        grid = voxelize(
            mesh,
            voxel_size=args.voxel_size,
            mode=args.mode,
            padding=args.padding,
            pixel_size=args.pixel_size,
            **transform_kwargs,
        )
    else:
        grid = voxelize_on_grid(
            mesh,
            voxel_size=args.voxel_size,
            grid_size_xyz=args.grid_size,
            bounds_min_xyz=args.bounds_min,
            bounds_max_xyz=args.bounds_max,
            mode=args.mode,
            padding=args.padding,
            pixel_size=args.pixel_size,
            **transform_kwargs,
        )

    output_path = args.output
    if output_path is None:
        output_path = args.mesh.with_name(f"{args.mesh.stem}_voxels.{'npz' if args.format == 'npz' else 'pkl'}")

    if args.format == "npz":
        grid.save_npz(output_path)
    else:
        grid.save_pickle(output_path, format=args.format)

    print(mesh.summary())
    print(grid.summary())
    print(f"Saved voxel grid to: {output_path}")

    if args.show:
        show(mesh, voxel_grid=grid, title=f"Voxelerate Viewer - {args.mesh.name}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    grid = load_voxel_grid(args.grid)
    print(grid.summary())
    print(f"bounds_min_xyz={grid.bounds_min_xyz.tolist()}")
    print(f"bounds_max_xyz={grid.bounds_max_xyz.tolist()}")
    print(f"voxel_size_xyz={grid.voxel_size_xyz.tolist()}")
    print(f"pixel_size={grid.pixel_size}")
    print(f"metadata={grid.metadata}")
    if args.slices:
        octree = build_voxel_octree(grid) if args.show_octree else None
        plot_central_slices(grid, octree=octree, title=f"Voxelerate slices - {args.grid.name}")
    if args.show:
        octree = build_voxel_octree(grid) if args.show_octree else None
        show(None, voxel_grid=grid, octree=octree, title=f"Voxelerate Viewer - {args.grid.name}")
    return 0


def cmd_slices(args: argparse.Namespace) -> int:
    grid = load_voxel_grid(args.grid)
    octree = None
    if args.octree is not None:
        octree = load_voxel_octree(args.octree)
    elif args.show_octree:
        octree = build_voxel_octree(grid)

    plot_central_slices(
        grid,
        octree=octree,
        max_depth=args.max_depth,
        leaves_only=args.leaves_only,
        occupied_only=not args.include_empty,
        title=f"Voxelerate slices - {args.grid.name}",
        save_path=args.save,
    )
    if args.save is not None:
        print(f"Saved slice figure to: {args.save}")
    return 0


def cmd_bvh(args: argparse.Namespace) -> int:
    mesh = load_mesh(args.mesh, center=args.center)
    bvh = build_bvh(
        mesh,
        strategy=args.strategy,
        max_leaf_size=args.max_leaf_size,
        bins=args.bins,
    )
    print(mesh.summary())
    print(bvh.summary())
    if args.save is not None:
        bvh.save_npz(args.save)
        print(f"Saved BVH archive to: {args.save}")
    if args.show:
        show(mesh, bvh=bvh, title=f"Voxelerate Viewer - {args.mesh.name}")
    return 0


def cmd_octree(args: argparse.Namespace) -> int:
    source = args.source
    if source.suffix.lower() in {".pkl", ".npz"}:
        grid = load_voxel_grid(source)
        octree = build_voxel_octree(grid, min_dim=args.min_dim, max_depth=args.max_depth)
        print(grid.summary())
        print(octree.summary())
        if args.save is not None:
            octree.save_npz(args.save)
            print(f"Saved voxel-octree archive to: {args.save}")
        if args.show:
            show(None, voxel_grid=grid, octree=octree, title=f"Voxelerate Viewer - {source.name}")
    else:
        mesh = load_mesh(source, center=args.center)
        octree = build_mesh_octree(mesh, max_depth=args.max_depth, max_triangles=args.max_triangles)
        print(mesh.summary())
        print(octree.summary())
        if args.save is not None:
            octree.save_npz(args.save)
            print(f"Saved mesh-octree archive to: {args.save}")
        if args.show:
            show(mesh, octree=octree, title=f"Voxelerate Viewer - {source.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
