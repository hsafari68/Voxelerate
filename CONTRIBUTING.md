# Contributing

Thanks for contributing to Voxelerate.

## Development setup

```bash
pip install -e ".[dev]"
```

To include the optional Open3D adapters:

```bash
pip install -e ".[dev,open3d]"
```

## Running tests

```bash
python -m pytest
```

## Code style

- keep axis order explicit
- keep world coordinates in `(x, y, z)`
- keep dense voxel arrays in `(z, y, x)`
- avoid mutating mesh geometry during rendering or voxelization
