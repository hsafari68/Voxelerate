from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]


def normalize(vector: NDArray[np.floating]) -> Float32Array:
    vec = np.asarray(vector, dtype=np.float32)
    length = np.linalg.norm(vec)
    if length < 1e-12:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / length).astype(np.float32)


def translation_matrix(offset_xyz: NDArray[np.floating]) -> Float32Array:
    offset = np.asarray(offset_xyz, dtype=np.float32).reshape(3)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = offset
    return matrix


def scale_matrix(scale_xyz: float | NDArray[np.floating]) -> Float32Array:
    if np.isscalar(scale_xyz):
        sx = sy = sz = float(scale_xyz)
    else:
        sx, sy, sz = np.asarray(scale_xyz, dtype=np.float32).reshape(3)

    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = sx
    matrix[1, 1] = sy
    matrix[2, 2] = sz
    return matrix


def rotation_matrix_x(angle_degrees: float) -> Float32Array:
    angle = math.radians(float(angle_degrees))
    c = math.cos(angle)
    s = math.sin(angle)
    matrix = np.eye(4, dtype=np.float32)
    matrix[1, 1] = c
    matrix[1, 2] = -s
    matrix[2, 1] = s
    matrix[2, 2] = c
    return matrix


def rotation_matrix_y(angle_degrees: float) -> Float32Array:
    angle = math.radians(float(angle_degrees))
    c = math.cos(angle)
    s = math.sin(angle)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = c
    matrix[0, 2] = s
    matrix[2, 0] = -s
    matrix[2, 2] = c
    return matrix


def rotation_matrix_z(angle_degrees: float) -> Float32Array:
    angle = math.radians(float(angle_degrees))
    c = math.cos(angle)
    s = math.sin(angle)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = c
    matrix[0, 1] = -s
    matrix[1, 0] = s
    matrix[1, 1] = c
    return matrix


def compose_transform(
    *,
    translation_xyz: NDArray[np.floating] | None = None,
    rotation_degrees_xyz: NDArray[np.floating] | None = None,
    scale_xyz: float | NDArray[np.floating] | None = None,
    pivot_xyz: NDArray[np.floating] | None = None,
) -> Float32Array:
    matrix = np.eye(4, dtype=np.float32)
    pivot_translate = np.eye(4, dtype=np.float32)
    pivot_untranslate = np.eye(4, dtype=np.float32)

    if pivot_xyz is not None:
        pivot = np.asarray(pivot_xyz, dtype=np.float32).reshape(3)
        pivot_translate = translation_matrix(pivot)
        pivot_untranslate = translation_matrix(-pivot)

    local = np.eye(4, dtype=np.float32)
    if scale_xyz is not None:
        local = scale_matrix(scale_xyz) @ local
    if rotation_degrees_xyz is not None:
        rx, ry, rz = np.asarray(rotation_degrees_xyz, dtype=np.float32).reshape(3)
        local = rotation_matrix_z(float(rz)) @ rotation_matrix_y(float(ry)) @ rotation_matrix_x(float(rx)) @ local

    matrix = pivot_translate @ local @ pivot_untranslate @ matrix
    if translation_xyz is not None:
        matrix = translation_matrix(translation_xyz) @ matrix
    return matrix.astype(np.float32)



def resolve_transform(
    transform: NDArray[np.floating] | None = None,
    *,
    translation_xyz: NDArray[np.floating] | None = None,
    rotation_degrees_xyz: NDArray[np.floating] | None = None,
    scale_xyz: float | NDArray[np.floating] | None = None,
    pivot_xyz: NDArray[np.floating] | None = None,
) -> Float32Array | None:
    """Resolve a transform matrix from an explicit matrix and/or TRS components.

    When both `transform` and component inputs are provided, the TRS component
    transform is applied *after* the base matrix, so the returned matrix is
    `component_matrix @ transform`.
    """
    has_components = any(
        value is not None
        for value in (translation_xyz, rotation_degrees_xyz, scale_xyz, pivot_xyz)
    )

    base: Float32Array | None = None
    if transform is not None:
        base = np.asarray(transform, dtype=np.float32)
        if base.shape != (4, 4):
            raise ValueError("transform matrix must have shape (4, 4)")

    if not has_components:
        return None if base is None else base.astype(np.float32, copy=False)

    component = compose_transform(
        translation_xyz=translation_xyz,
        rotation_degrees_xyz=rotation_degrees_xyz,
        scale_xyz=scale_xyz,
        pivot_xyz=pivot_xyz,
    )
    if base is None:
        return component
    return (component @ base).astype(np.float32)

def perspective(
    fov_y_degrees: float,
    aspect_ratio: float,
    near: float,
    far: float,
) -> Float32Array:
    f = 1.0 / math.tan(math.radians(fov_y_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / max(aspect_ratio, 1e-6)
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(
    eye_xyz: NDArray[np.floating],
    target_xyz: NDArray[np.floating],
    up_xyz: NDArray[np.floating],
) -> Float32Array:
    eye = np.asarray(eye_xyz, dtype=np.float32).reshape(3)
    target = np.asarray(target_xyz, dtype=np.float32).reshape(3)
    up = normalize(up_xyz)

    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    camera_up = normalize(np.cross(right, forward))

    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = camera_up
    matrix[2, :3] = -forward
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix
