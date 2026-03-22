from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from numpy.typing import NDArray

from .exceptions import UnsupportedMeshError

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]


def _normalize_rows(vectors: Float32Array) -> Float32Array:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    return (vectors / lengths).astype(np.float32, copy=False)


def _compute_vertex_normals(vertices: Float32Array, faces: Int32Array) -> Float32Array:
    normals = np.zeros_like(vertices, dtype=np.float32)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).astype(np.float32)

    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)

    return _normalize_rows(normals)


def _parse_obj(path: Path) -> tuple[Float32Array, Int32Array]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) < 3:
                    continue

                polygon: list[int] = []
                for token in tokens:
                    vertex_token = token.split("/")[0]
                    if not vertex_token:
                        raise UnsupportedMeshError(f"Malformed OBJ face entry in {path.name!r}: {line!r}")
                    index = int(vertex_token)
                    if index < 0:
                        index = len(vertices) + index
                    else:
                        index -= 1
                    polygon.append(index)

                for idx in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[idx], polygon[idx + 1]])

    if not vertices or not faces:
        raise UnsupportedMeshError(f"OBJ file {path} does not contain vertices and triangular faces.")

    return (
        np.ascontiguousarray(np.asarray(vertices, dtype=np.float32)),
        np.ascontiguousarray(np.asarray(faces, dtype=np.int32)),
    )


def _deduplicate_vertices(vertices: Float32Array, faces: Int32Array) -> tuple[Float32Array, Int32Array]:
    if len(vertices) == 0:
        return vertices, faces
    unique_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    remapped_faces = inverse[faces.reshape(-1)].reshape(-1, 3).astype(np.int32, copy=False)
    return (
        np.ascontiguousarray(unique_vertices.astype(np.float32, copy=False)),
        np.ascontiguousarray(remapped_faces, dtype=np.int32),
    )


def _is_binary_stl(path: Path) -> bool:
    file_size = path.stat().st_size
    if file_size < 84:
        return False

    with path.open("rb") as handle:
        header = handle.read(80)
        count_bytes = handle.read(4)
        if len(count_bytes) != 4:
            return False
        triangle_count = struct.unpack("<I", count_bytes)[0]
        expected_size = 84 + triangle_count * 50

    if file_size == expected_size:
        return True

    try:
        header.decode("ascii")
    except UnicodeDecodeError:
        return True
    return False


def _parse_binary_stl(path: Path) -> tuple[Float32Array, Int32Array]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with path.open("rb") as handle:
        handle.read(80)
        triangle_count = struct.unpack("<I", handle.read(4))[0]

        for _ in range(triangle_count):
            record = handle.read(50)
            if len(record) != 50:
                raise UnsupportedMeshError(f"Unexpected end of binary STL file: {path}")
            unpacked = struct.unpack("<12fH", record)
            v0 = list(unpacked[3:6])
            v1 = list(unpacked[6:9])
            v2 = list(unpacked[9:12])

            base = len(vertices)
            vertices.extend([v0, v1, v2])
            faces.append([base, base + 1, base + 2])

    if not vertices:
        raise UnsupportedMeshError(f"Binary STL file {path} does not contain triangles.")

    return _deduplicate_vertices(
        np.ascontiguousarray(np.asarray(vertices, dtype=np.float32)),
        np.ascontiguousarray(np.asarray(faces, dtype=np.int32)),
    )


def _parse_ascii_stl(path: Path) -> tuple[Float32Array, Int32Array]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    tri_vertices: list[list[float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("vertex "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                tri_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(tri_vertices) == 3:
                    base = len(vertices)
                    vertices.extend(tri_vertices)
                    faces.append([base, base + 1, base + 2])
                    tri_vertices = []

    if not vertices:
        raise UnsupportedMeshError(f"ASCII STL file {path} does not contain triangles.")

    return _deduplicate_vertices(
        np.ascontiguousarray(np.asarray(vertices, dtype=np.float32)),
        np.ascontiguousarray(np.asarray(faces, dtype=np.int32)),
    )


def _parse_stl(path: Path) -> tuple[Float32Array, Int32Array]:
    if _is_binary_stl(path):
        return _parse_binary_stl(path)
    return _parse_ascii_stl(path)


@dataclass(slots=True)
class Mesh:
    """Triangle mesh data used by the viewer, voxelizer, BVH, and octree builders."""

    vertices: Float32Array
    faces: Int32Array
    vertex_normals: Float32Array
    name: str = "mesh"
    source_path: Path | None = None

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        self.vertex_normals = np.ascontiguousarray(self.vertex_normals, dtype=np.float32)

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        if self.vertex_normals.shape != self.vertices.shape:
            raise ValueError("vertex_normals must have the same shape as vertices")
        if len(self.vertices) == 0 or len(self.faces) == 0:
            raise ValueError("mesh must not be empty")

    @classmethod
    def from_arrays(
        cls,
        vertices: NDArray[np.floating],
        faces: NDArray[np.integer],
        *,
        name: str = "mesh",
        source_path: Path | None = None,
        center: bool = False,
    ) -> "Mesh":
        vertices_np = np.asarray(vertices, dtype=np.float32)
        faces_np = np.asarray(faces, dtype=np.int32)

        if center:
            bounds_min = vertices_np.min(axis=0)
            bounds_max = vertices_np.max(axis=0)
            vertices_np = vertices_np - (bounds_min + bounds_max) * 0.5

        normals = _compute_vertex_normals(vertices_np, faces_np)
        return cls(
            vertices=np.ascontiguousarray(vertices_np, dtype=np.float32),
            faces=np.ascontiguousarray(faces_np, dtype=np.int32),
            vertex_normals=np.ascontiguousarray(normals, dtype=np.float32),
            name=name,
            source_path=source_path,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        center: bool = False,
    ) -> "Mesh":
        source_path = Path(path)
        suffix = source_path.suffix.lower()

        if suffix == ".obj":
            vertices, faces = _parse_obj(source_path)
        elif suffix == ".stl":
            vertices, faces = _parse_stl(source_path)
        else:
            raise UnsupportedMeshError(
                f"Unsupported mesh format {suffix!r}. Voxelerate currently supports OBJ and STL."
            )

        return cls.from_arrays(
            vertices,
            faces,
            name=source_path.stem,
            source_path=source_path,
            center=center,
        )

    @property
    def bounds(self) -> tuple[Float32Array, Float32Array]:
        return (
            self.vertices.min(axis=0).astype(np.float32),
            self.vertices.max(axis=0).astype(np.float32),
        )

    @property
    def bounding_box(self) -> list[list[float]]:
        bounds_min, bounds_max = self.bounds
        return [bounds_min.tolist(), bounds_max.tolist()]

    @property
    def center(self) -> Float32Array:
        bounds_min, bounds_max = self.bounds
        return ((bounds_min + bounds_max) * 0.5).astype(np.float32)

    @property
    def extent(self) -> Float32Array:
        bounds_min, bounds_max = self.bounds
        return (bounds_max - bounds_min).astype(np.float32)

    @property
    def triangle_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertices_np(self) -> Float32Array:
        return self.vertices

    @property
    def indices_np(self) -> Int32Array:
        return self.faces

    @property
    def normals_np(self) -> Float32Array:
        return self.vertex_normals

    @property
    def triangles_np(self) -> Float32Array:
        return self.flattened_triangles()

    @property
    def pos_norm_np(self) -> Float32Array:
        triangles = self.flattened_triangles()
        tri_normals = self.transformed_normals()[self.faces.reshape(-1)].reshape(-1, 9)
        return np.ascontiguousarray(np.concatenate([triangles, tri_normals], axis=1), dtype=np.float32)

    def centered(self) -> "Mesh":
        return Mesh.from_arrays(
            self.vertices - self.center,
            self.faces,
            name=self.name,
            source_path=self.source_path,
            center=False,
        )

    def transformed_vertices(self, matrix: NDArray[np.floating] | None = None) -> Float32Array:
        if matrix is None:
            return self.vertices.copy()

        transform = np.asarray(matrix, dtype=np.float32)
        if transform.shape != (4, 4):
            raise ValueError("transform matrix must have shape (4, 4)")

        hom = np.concatenate(
            [self.vertices, np.ones((self.vertices.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        transformed = (transform @ hom.T).T[:, :3]
        return np.ascontiguousarray(transformed, dtype=np.float32)

    def transformed_normals(self, matrix: NDArray[np.floating] | None = None) -> Float32Array:
        if matrix is None:
            return self.vertex_normals.copy()

        transform = np.asarray(matrix, dtype=np.float32)
        if transform.shape != (4, 4):
            raise ValueError("transform matrix must have shape (4, 4)")

        normal_matrix = np.linalg.inv(transform[:3, :3]).T.astype(np.float32)
        normals = (normal_matrix @ self.vertex_normals.T).T.astype(np.float32)
        return np.ascontiguousarray(_normalize_rows(normals), dtype=np.float32)

    def flattened_triangles(self, transform: NDArray[np.floating] | None = None) -> Float32Array:
        vertices = self.transformed_vertices(transform)
        triangles = vertices[self.faces.reshape(-1)].reshape(-1, 9)
        return np.ascontiguousarray(triangles, dtype=np.float32)

    def interleaved_positions_normals(
        self,
        transform: NDArray[np.floating] | None = None,
    ) -> Float32Array:
        positions = self.transformed_vertices(transform)
        normals = self.transformed_normals(transform)
        return np.ascontiguousarray(np.concatenate([positions, normals], axis=1), dtype=np.float32)

    def summary(self) -> str:
        bounds_min, bounds_max = self.bounds
        return (
            f"Mesh(name={self.name!r}, vertices={len(self.vertices)}, triangles={self.triangle_count}, "
            f"bounds_min={bounds_min.tolist()}, bounds_max={bounds_max.tolist()})"
        )


def load_mesh(path: str | Path, *, center: bool = False) -> Mesh:
    """Load an OBJ or STL mesh into a Voxelerate Mesh."""
    return Mesh.from_file(path, center=center)
