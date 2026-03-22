from __future__ import annotations

from dataclasses import dataclass, field
import ctypes
import math

import numpy as np
from numpy.typing import NDArray

from .bvh import BVH
from .math3d import (
    look_at,
    normalize,
    perspective,
    rotation_matrix_x,
    rotation_matrix_y,
    scale_matrix,
    translation_matrix,
)
from .mesh import Mesh
from .octree import MeshOctree, VoxelOctree
from .spatial_io import FlatBVH, FlatMeshOctree, FlatVoxelOctree
from .opengl import (
    GL,
    OpenGLContext,
    compile_shader,
    delete_program,
    glfw,
    link_program,
    read_shader,
    require_opengl,
    set_uniform_float,
    set_uniform_int,
    set_uniform_mat3,
    set_uniform_mat4,
    set_uniform_vec3,
    shader_root,
)
from .voxel_grid import VoxelGrid

Float32Array = NDArray[np.float32]


_BITMAP_FONT_5X7: dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "+": ("00100", "00100", "00100", "11111", "00100", "00100", "00100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00001", "00001", "00001", "00001", "10001", "10001", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}

_HELP_PANEL_TITLE = "VOXELERATE CONTROLS"
_HELP_PANEL_LINES: tuple[str, ...] = (
    "LEFT DRAG   ORBIT CAMERA",
    "RIGHT DRAG  PAN CAMERA",
    "WHEEL       ZOOM CAMERA",
    "ARROWS      ROTATE PREVIEW",
    "PLUS MINUS  SCALE PREVIEW",
    "B           TOGGLE BOUNDING BOX",
    "G           TOGGLE GROUND GRID",
    "O           TOGGLE HIERARCHY",
    "V           TOGGLE VOXELS",
    "W           TOGGLE WIREFRAME",
    "R           RESET CAMERA",
    "T           RESET PREVIEW",
    "H           TOGGLE THIS HELP",
    "ESC         CLOSE VIEWER",
    "",
    "PREVIEW CONTROLS ARE VISUAL ONLY",
    "VOXELIZATION USES INPUT TRANSFORMS ONLY",
)

VIEWER_TITLE_CONTROLS = (
    "Mouse: L-orbit R-pan Wheel-zoom | Arrows rotate preview | +/- scale preview | "
    "M mesh B box G grid O hierarchy V voxels W wireframe X xray | R camera T preview | Esc close"
)

_VIEWER_CONTROLS_HELP = """Voxelerate viewer controls

Mouse
  Left drag   Orbit camera
  Right drag  Pan camera
  Wheel       Zoom camera

Preview
  Arrow keys  Rotate the displayed geometry
  + / -       Scale the displayed geometry while held
  T           Reset the preview transform

Visibility
  M           Toggle mesh rendering
  V           Toggle voxel rendering
  B           Toggle the bounding box
  G           Toggle the ground grid
  O           Toggle BVH / octree overlays
  W           Toggle wireframe
  X           Toggle x-ray solid voxel preview

Other
  R           Reset the camera
  Esc         Close the viewer

Preview controls never affect voxelization.
To transform geometry for voxelization, pass transform / translation / rotation / scale inputs into voxelize(...).
"""


def viewer_controls_help() -> str:
    """Return the viewer control reference as plain text."""
    return _VIEWER_CONTROLS_HELP


def _window_title_text(base_title: str, *, voxel_grid: VoxelGrid | None = None) -> str:
    title_parts = [base_title, "preview only"]
    if voxel_grid is not None:
        title_parts.append(f"{voxel_grid.mode}:{voxel_grid.occupied_voxels:,} voxels")
    title_parts.append(VIEWER_TITLE_CONTROLS)
    return " | ".join(title_parts)



def _fill_rgba_rect(image: NDArray[np.uint8], x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int, int]) -> None:
    height, width, _ = image.shape
    x0 = max(0, min(width, int(x0)))
    x1 = max(0, min(width, int(x1)))
    y0 = max(0, min(height, int(y0)))
    y1 = max(0, min(height, int(y1)))
    if x0 >= x1 or y0 >= y1:
        return
    image[y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)


def _stroke_rgba_rect(
    image: NDArray[np.uint8],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int, int],
    *,
    thickness: int = 1,
) -> None:
    _fill_rgba_rect(image, x0, y0, x1, y0 + thickness, color)
    _fill_rgba_rect(image, x0, y1 - thickness, x1, y1, color)
    _fill_rgba_rect(image, x0, y0, x0 + thickness, y1, color)
    _fill_rgba_rect(image, x1 - thickness, y0, x1, y1, color)


def _draw_bitmap_text(
    image: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    *,
    scale: int = 4,
    color: tuple[int, int, int, int] = (255, 255, 255, 255),
    letter_spacing: int = 1,
) -> int:
    cursor_x = int(x)
    glyph_width = 5 * scale
    advance = glyph_width + letter_spacing * scale
    for char in text.upper():
        glyph = _BITMAP_FONT_5X7.get(char, _BITMAP_FONT_5X7[" "])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == "1":
                    _fill_rgba_rect(
                        image,
                        cursor_x + col * scale,
                        y + row * scale,
                        cursor_x + (col + 1) * scale,
                        y + (row + 1) * scale,
                        color,
                    )
        cursor_x += advance
    return cursor_x


def _help_panel_dimensions(scale: int = 4, title_scale: int = 5) -> tuple[int, int]:
    letter_advance = (5 + 1) * scale
    title_advance = (5 + 1) * title_scale
    longest_line = max(len(_HELP_PANEL_TITLE) * title_advance, *(len(line) * letter_advance for line in _HELP_PANEL_LINES))
    width = longest_line + 96
    body_line_height = 7 * scale + 2 * scale
    title_height = 7 * title_scale
    height = 44 + title_height + 28 + len(_HELP_PANEL_LINES) * body_line_height + 34
    return int(width), int(height)


def build_help_panel_image(scale: int = 4, title_scale: int = 5) -> NDArray[np.uint8]:
    width, height = _help_panel_dimensions(scale=scale, title_scale=title_scale)
    image = np.zeros((height, width, 4), dtype=np.uint8)

    background = (5, 8, 17, 218)
    border = (71, 85, 105, 240)
    accent = (239, 68, 68, 235)
    title_color = (248, 250, 252, 255)
    body_color = (226, 232, 240, 255)
    note_color = (148, 163, 184, 255)

    _fill_rgba_rect(image, 0, 0, width, height, background)
    _stroke_rgba_rect(image, 0, 0, width, height, border, thickness=2)
    _fill_rgba_rect(image, 0, 0, width, 8, accent)

    title_y = 28
    _draw_bitmap_text(image, _HELP_PANEL_TITLE, 26, title_y, scale=title_scale, color=title_color)

    separator_y = title_y + 7 * title_scale + 18
    _fill_rgba_rect(image, 24, separator_y, width - 24, separator_y + 2, (51, 65, 85, 255))

    start_y = separator_y + 18
    line_height = 7 * scale + 2 * scale
    for index, line in enumerate(_HELP_PANEL_LINES):
        y = start_y + index * line_height
        current_color = note_color if index >= len(_HELP_PANEL_LINES) - 2 else body_color
        if not line:
            continue
        _draw_bitmap_text(image, line, 28, y, scale=scale, color=current_color)

    return image


def _bbox_lines(bounds_min_xyz: Float32Array, bounds_max_xyz: Float32Array) -> tuple[Float32Array, NDArray[np.uint32]]:
    x0, y0, z0 = bounds_min_xyz.tolist()
    x1, y1, z1 = bounds_max_xyz.tolist()

    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0, 1, 1, 2, 2, 3, 3, 0,
            4, 5, 5, 6, 6, 7, 7, 4,
            0, 4, 1, 5, 2, 6, 3, 7,
        ],
        dtype=np.uint32,
    )
    return vertices, indices


def _ground_grid_lines(
    bounds_min_xyz: Float32Array,
    bounds_max_xyz: Float32Array,
    *,
    divisions: int = 16,
) -> tuple[Float32Array, NDArray[np.uint32]]:
    extent = bounds_max_xyz - bounds_min_xyz
    center = (bounds_min_xyz + bounds_max_xyz) * 0.5
    size = max(float(extent[0]), float(extent[2]), 1e-3) * 1.35
    half = size * 0.5

    if bounds_min_xyz[1] <= 0.0 <= bounds_max_xyz[1]:
        y = 0.0
    else:
        y = float(bounds_min_xyz[1] - max(extent[1] * 0.05, size * 0.01))

    xs = np.linspace(center[0] - half, center[0] + half, divisions + 1, dtype=np.float32)
    zs = np.linspace(center[2] - half, center[2] + half, divisions + 1, dtype=np.float32)

    vertices: list[list[float]] = []
    indices: list[int] = []

    for x in xs:
        base = len(vertices)
        vertices.append([float(x), y, float(center[2] - half)])
        vertices.append([float(x), y, float(center[2] + half)])
        indices.extend([base, base + 1])

    for z in zs:
        base = len(vertices)
        vertices.append([float(center[0] - half), y, float(z)])
        vertices.append([float(center[0] + half), y, float(z)])
        indices.extend([base, base + 1])

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def _combine_line_sets(
    box_groups: list[tuple[list[tuple[np.ndarray, np.ndarray, int]], bool]],
) -> list[tuple[Float32Array, NDArray[np.uint32], bool]]:
    line_sets: list[tuple[Float32Array, NDArray[np.uint32], bool]] = []
    for boxes, apply_transform in box_groups:
        if not boxes:
            continue
        vertices_all = []
        indices_all = []
        offset = 0
        for bounds_min_xyz, bounds_max_xyz, _ in boxes:
            vertices, indices = _bbox_lines(
                np.asarray(bounds_min_xyz, dtype=np.float32),
                np.asarray(bounds_max_xyz, dtype=np.float32),
            )
            vertices_all.append(vertices)
            indices_all.append(indices + offset)
            offset += len(vertices)
        line_sets.append(
            (
                np.vstack(vertices_all).astype(np.float32),
                np.concatenate(indices_all).astype(np.uint32),
                apply_transform,
            )
        )
    return line_sets


def _merge_bounds(*items: tuple[np.ndarray, np.ndarray] | None) -> tuple[Float32Array, Float32Array]:
    mins = []
    maxs = []
    for item in items:
        if item is None:
            continue
        bounds_min_xyz, bounds_max_xyz = item
        mins.append(np.asarray(bounds_min_xyz, dtype=np.float32))
        maxs.append(np.asarray(bounds_max_xyz, dtype=np.float32))
    if not mins:
        raise ValueError("At least one bounds item is required.")
    return np.min(mins, axis=0).astype(np.float32), np.max(maxs, axis=0).astype(np.float32)


def _transform_bounds(
    bounds_min_xyz: NDArray[np.floating],
    bounds_max_xyz: NDArray[np.floating],
    transform: NDArray[np.floating] | None,
) -> tuple[Float32Array, Float32Array]:
    bounds_min = np.asarray(bounds_min_xyz, dtype=np.float32).reshape(3)
    bounds_max = np.asarray(bounds_max_xyz, dtype=np.float32).reshape(3)
    if transform is None:
        return bounds_min.astype(np.float32), bounds_max.astype(np.float32)

    corners = np.array(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_max[1], bounds_max[2]],
            [bounds_min[0], bounds_max[1], bounds_max[2]],
        ],
        dtype=np.float32,
    )
    hom = np.concatenate([corners, np.ones((8, 1), dtype=np.float32)], axis=1)
    matrix = np.asarray(transform, dtype=np.float32)
    transformed = (matrix @ hom.T).T[:, :3]
    return transformed.min(axis=0).astype(np.float32), transformed.max(axis=0).astype(np.float32)


def _transformed_bounds(mesh: Mesh, transform: NDArray[np.floating] | None) -> tuple[Float32Array, Float32Array]:
    return _transform_bounds(*mesh.bounds, transform)


def _transformed_center(mesh: Mesh, transform: NDArray[np.floating] | None) -> Float32Array:
    center = np.concatenate([mesh.center.astype(np.float32), np.array([1.0], dtype=np.float32)])
    matrix = np.eye(4, dtype=np.float32) if transform is None else np.asarray(transform, dtype=np.float32)
    return (matrix @ center)[:3].astype(np.float32)


@dataclass(slots=True)
class OrbitCamera:
    target: Float32Array
    distance: float
    yaw_degrees: float = 35.0
    pitch_degrees: float = 25.0
    fov_y_degrees: float = 45.0
    world_up: Float32Array = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))

    @classmethod
    def from_bounds(cls, bounds_min_xyz: Float32Array, bounds_max_xyz: Float32Array) -> "OrbitCamera":
        target = ((bounds_min_xyz + bounds_max_xyz) * 0.5).astype(np.float32)
        radius = max(np.linalg.norm(bounds_max_xyz - bounds_min_xyz) * 0.5, 1e-3)
        distance = max(radius / math.tan(math.radians(45.0) * 0.5), radius * 2.25)
        return cls(target=target, distance=float(distance))

    @property
    def position(self) -> Float32Array:
        yaw = math.radians(self.yaw_degrees)
        pitch = math.radians(self.pitch_degrees)
        direction = np.array(
            [
                math.cos(pitch) * math.cos(yaw),
                math.sin(pitch),
                math.cos(pitch) * math.sin(yaw),
            ],
            dtype=np.float32,
        )
        return (self.target + direction * self.distance).astype(np.float32)

    def view_matrix(self) -> Float32Array:
        return look_at(self.position, self.target, self.world_up)

    def projection_matrix(self, width: int, height: int) -> Float32Array:
        aspect = max(width / max(height, 1), 1e-6)
        near = max(self.distance * 0.01, 1e-3)
        far = max(self.distance * 10.0, near + 1.0)
        return perspective(self.fov_y_degrees, aspect, near, far)

    def orbit(self, delta_x: float, delta_y: float) -> None:
        self.yaw_degrees += delta_x * 0.35
        self.pitch_degrees = float(np.clip(self.pitch_degrees - delta_y * 0.35, -89.0, 89.0))

    def zoom(self, scroll_steps: float) -> None:
        scale = 0.92 ** scroll_steps
        self.distance = max(self.distance * scale, 1e-3)

    def pan(self, delta_x: float, delta_y: float, width: int, height: int) -> None:
        forward = normalize(self.target - self.position)
        right = normalize(np.cross(forward, self.world_up))
        up = normalize(np.cross(right, forward))

        scale = self.distance * math.tan(math.radians(self.fov_y_degrees) * 0.5)
        dx = -delta_x / max(width, 1) * scale * 2.0
        dy = delta_y / max(height, 1) * scale * 2.0
        self.target = (self.target + right * dx + up * dy).astype(np.float32)


class MeshViewer:
    """Interactive OpenGL viewer for meshes, voxel grids, and spatial hierarchies."""

    def __init__(self, *, width: int = 1440, height: int = 900, title: str = "Voxelerate Viewer") -> None:
        self.width = int(width)
        self.height = int(height)
        self.title = title

        self._context: OpenGLContext | None = None
        self._mesh_program: int | None = None
        self._line_program: int | None = None
        self._point_program: int | None = None
        self._overlay_program: int | None = None

        self._mesh_vao: int | None = None
        self._mesh_vbo: int | None = None
        self._mesh_ebo: int | None = None
        self._mesh_index_count = 0

        self._line_objects: list[dict[str, object]] = []
        self._voxel_vao: int | None = None
        self._voxel_vbo: int | None = None
        self._voxel_count = 0
        self._voxel_size_reference = 1.0

        self._help_vao: int | None = None
        self._help_vbo: int | None = None
        self._help_texture: int | None = None
        self._help_texture_shape: tuple[int, int] = (0, 0)

        self._camera: OrbitCamera | None = None
        self._scene_bounds_min: Float32Array | None = None
        self._scene_bounds_max: Float32Array | None = None
        self._left_mouse_down = False
        self._right_mouse_down = False
        self._last_cursor: tuple[float, float] | None = None
        self._show_bbox = True
        self._show_ground = True
        self._show_voxels = True
        self._show_hierarchy = True
        self._show_mesh = True
        self._show_help = False
        self._wireframe = False
        self._voxel_mode: str | None = None
        self._voxel_xray = False
        self._key_latch: dict[int, bool] = {}
        self._interaction_mode: str | None = None
        self._initial_model_transform = np.eye(4, dtype=np.float32)
        self._model_transform = np.eye(4, dtype=np.float32)
        self._transform_pivot = np.zeros(3, dtype=np.float32)
        self._mesh_for_reset: Mesh | None = None
        self._voxel_grid_for_reset: VoxelGrid | None = None
        self._display_transform_enabled = False

    def show(
        self,
        mesh: Mesh | None = None,
        *,
        voxel_grid: VoxelGrid | None = None,
        bvh: BVH | FlatBVH | None = None,
        octree: MeshOctree | VoxelOctree | FlatMeshOctree | FlatVoxelOctree | None = None,
        transform: NDArray[np.floating] | None = None,
        return_visual_transform: bool = False,
    ) -> NDArray[np.float32] | None:
        require_opengl()
        assert glfw is not None and GL is not None

        if mesh is None and voxel_grid is None:
            raise ValueError("At least one of `mesh` or `voxel_grid` must be provided.")

        self._display_transform_enabled = bool(mesh is not None or voxel_grid is not None)
        self._show_mesh = True
        self._show_voxels = voxel_grid is not None
        self._voxel_mode = None
        self._voxel_xray = False
        self._initial_model_transform = (
            np.eye(4, dtype=np.float32) if transform is None else np.asarray(transform, dtype=np.float32)
        )
        self._model_transform = self._initial_model_transform.copy()
        self._mesh_for_reset = mesh
        self._voxel_grid_for_reset = voxel_grid

        self._context = OpenGLContext(
            width=self.width,
            height=self.height,
            title=self.title,
            visible=True,
            samples=4,
            version_attempts=((4, 5), (4, 3), (3, 3)),
        ).create()

        try:
            window = self._context.window
            assert window is not None

            mesh_bounds = _transformed_bounds(mesh, self._model_transform) if mesh is not None else None
            volume_bounds = (
                _transform_bounds(voxel_grid.bounds_min_xyz, voxel_grid.bounds_max_xyz, self._model_transform)
                if voxel_grid is not None
                else None
            )
            self._scene_bounds_min, self._scene_bounds_max = _merge_bounds(mesh_bounds, volume_bounds)
            self._camera = OrbitCamera.from_bounds(self._scene_bounds_min, self._scene_bounds_max)
            self._transform_pivot = (
                _transformed_center(mesh, self._model_transform)
                if mesh is not None
                else ((self._scene_bounds_min + self._scene_bounds_max) * 0.5)
            ).astype(np.float32)

            self._build_programs()
            if mesh is not None:
                self._build_mesh_buffers(mesh)
            self._build_line_overlays(mesh, voxel_grid=voxel_grid, bvh=bvh, octree=octree)
            if voxel_grid is not None:
                self._build_voxel_buffers(voxel_grid)

            glfw.set_window_user_pointer(window, self)
            glfw.set_cursor_pos_callback(window, self._cursor_position_callback)
            glfw.set_mouse_button_callback(window, self._mouse_button_callback)
            glfw.set_scroll_callback(window, self._scroll_callback)
            glfw.set_framebuffer_size_callback(window, self._framebuffer_size_callback)

            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_MULTISAMPLE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
            GL.glClearColor(0.08, 0.09, 0.11, 1.0)

            while not glfw.window_should_close(window):
                glfw.poll_events()
                self._process_keyboard(window)
                self._render(mesh=mesh)
                glfw.swap_buffers(window)
            return self._model_transform.copy() if return_visual_transform else None
        finally:
            self._cleanup()

    def _build_programs(self) -> None:
        assert GL is not None
        mesh_vs = compile_shader(
            read_shader(shader_root() / "mesh.vert"),
            GL.GL_VERTEX_SHADER,
            label="mesh.vert",
        )
        mesh_fs = compile_shader(
            read_shader(shader_root() / "mesh.frag"),
            GL.GL_FRAGMENT_SHADER,
            label="mesh.frag",
        )
        self._mesh_program = link_program(mesh_vs, mesh_fs, label="mesh-program")

        line_vs = compile_shader(
            read_shader(shader_root() / "flat_color.vert"),
            GL.GL_VERTEX_SHADER,
            label="flat_color.vert",
        )
        line_fs = compile_shader(
            read_shader(shader_root() / "flat_color.frag"),
            GL.GL_FRAGMENT_SHADER,
            label="flat_color.frag",
        )
        self._line_program = link_program(line_vs, line_fs, label="flat-color-program")

        point_vs = compile_shader(
            read_shader(shader_root() / "voxel_points.vert"),
            GL.GL_VERTEX_SHADER,
            label="voxel_points.vert",
        )
        point_fs = compile_shader(
            read_shader(shader_root() / "voxel_points.frag"),
            GL.GL_FRAGMENT_SHADER,
            label="voxel_points.frag",
        )
        self._point_program = link_program(point_vs, point_fs, label="voxel-point-program")

        overlay_vs = compile_shader(
            read_shader(shader_root() / "overlay.vert"),
            GL.GL_VERTEX_SHADER,
            label="overlay.vert",
        )
        overlay_fs = compile_shader(
            read_shader(shader_root() / "overlay.frag"),
            GL.GL_FRAGMENT_SHADER,
            label="overlay.frag",
        )
        self._overlay_program = link_program(overlay_vs, overlay_fs, label="overlay-program")

    def _build_mesh_buffers(self, mesh: Mesh) -> None:
        assert GL is not None
        interleaved = mesh.interleaved_positions_normals(transform=None)
        indices = np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.uint32)

        self._mesh_index_count = int(indices.size)
        self._mesh_vao = GL.glGenVertexArrays(1)
        self._mesh_vbo = GL.glGenBuffers(1)
        self._mesh_ebo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self._mesh_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._mesh_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL.GL_STATIC_DRAW)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._mesh_ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)

        stride = 6 * interleaved.itemsize
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, None)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            1,
            3,
            GL.GL_FLOAT,
            GL.GL_FALSE,
            stride,
            ctypes.c_void_p(3 * interleaved.itemsize),
        )
        GL.glEnableVertexAttribArray(1)
        GL.glBindVertexArray(0)

    def _build_line_buffer(
        self,
        vertices: Float32Array,
        indices: NDArray[np.uint32],
        *,
        color: tuple[float, float, float],
        kind: str,
        apply_transform: bool = False,
    ) -> None:
        assert GL is not None
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        ebo = GL.glGenBuffers(1)

        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)

        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 3 * vertices.itemsize, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

        self._line_objects.append(
            {
                "vao": vao,
                "vbo": vbo,
                "ebo": ebo,
                "count": int(indices.size),
                "color": np.array(color, dtype=np.float32),
                "kind": kind,
                "apply_transform": apply_transform,
            }
        )

    def _build_line_overlays(
        self,
        mesh: Mesh | None,
        *,
        voxel_grid: VoxelGrid | None,
        bvh: BVH | FlatBVH | None,
        octree: MeshOctree | VoxelOctree | FlatMeshOctree | FlatVoxelOctree | None,
    ) -> None:
        if self._scene_bounds_min is None or self._scene_bounds_max is None:
            return

        grid_vertices, grid_indices = _ground_grid_lines(self._scene_bounds_min, self._scene_bounds_max)
        self._build_line_buffer(grid_vertices, grid_indices, color=(0.26, 0.28, 0.31), kind="ground")

        if mesh is not None:
            bounds_min, bounds_max = mesh.bounds
            bbox_vertices, bbox_indices = _bbox_lines(bounds_min, bounds_max)
            self._build_line_buffer(
                bbox_vertices,
                bbox_indices,
                color=(0.25, 0.65, 1.0),
                kind="bbox",
                apply_transform=True,
            )
        elif voxel_grid is not None:
            bbox_vertices, bbox_indices = _bbox_lines(voxel_grid.bounds_min_xyz, voxel_grid.bounds_max_xyz)
            self._build_line_buffer(
                bbox_vertices,
                bbox_indices,
                color=(0.25, 0.65, 1.0),
                kind="bbox",
                apply_transform=True,
            )

        hierarchy_groups: list[tuple[list[tuple[np.ndarray, np.ndarray, int]], bool, tuple[float, float, float], str]] = []

        if bvh is not None:
            hierarchy_groups.append((bvh.collect_boxes(max_depth=4), True, (1.0, 0.65, 0.2), "hierarchy"))

        if octree is not None:
            if isinstance(octree, (MeshOctree, FlatMeshOctree)):
                boxes = octree.collect_boxes(max_depth=4)
                hierarchy_groups.append((boxes, True, (0.92, 0.18, 0.24), "hierarchy"))
            else:
                boxes = [
                    (mn, mx, depth)
                    for mn, mx, depth, value in octree.collect_boxes(max_depth=4, occupied_only=True)
                    if value != 0
                ]
                hierarchy_groups.append((boxes, True, (0.92, 0.18, 0.24), "hierarchy"))

        for boxes, apply_transform, color, kind in hierarchy_groups:
            if not boxes:
                continue
            vertices_all = []
            indices_all = []
            offset = 0
            for bounds_min_xyz, bounds_max_xyz, _ in boxes:
                vertices, indices = _bbox_lines(
                    np.asarray(bounds_min_xyz, dtype=np.float32),
                    np.asarray(bounds_max_xyz, dtype=np.float32),
                )
                vertices_all.append(vertices)
                indices_all.append(indices + offset)
                offset += len(vertices)
            self._build_line_buffer(
                np.vstack(vertices_all).astype(np.float32),
                np.concatenate(indices_all).astype(np.uint32),
                color=color,
                kind=kind,
                apply_transform=apply_transform,
            )

    def _build_voxel_buffers(self, voxel_grid: VoxelGrid) -> None:
        assert GL is not None
        centers = voxel_grid.occupied_centers()
        self._voxel_count = int(len(centers))
        self._voxel_size_reference = float(np.min(voxel_grid.voxel_size_xyz))
        self._voxel_mode = voxel_grid.mode.lower()
        self._voxel_xray = self._voxel_mode == "solid"
        self._voxel_vao = GL.glGenVertexArrays(1)
        self._voxel_vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self._voxel_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._voxel_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, centers.nbytes, centers, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 3 * centers.itemsize, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

    def _build_help_overlay_resources(self) -> None:
        assert GL is not None
        image = build_help_panel_image()
        height, width, _ = image.shape
        self._help_texture_shape = (width, height)

        self._help_vao = GL.glGenVertexArrays(1)
        self._help_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._help_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._help_vbo)
        initial_vertices = np.zeros((4, 4), dtype=np.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, initial_vertices.nbytes, initial_vertices, GL.GL_DYNAMIC_DRAW)
        stride = 4 * np.dtype(np.float32).itemsize
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, None)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(2 * np.dtype(np.float32).itemsize))
        GL.glEnableVertexAttribArray(1)
        GL.glBindVertexArray(0)

        self._help_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._help_texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            image,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _update_help_quad_vertices(self) -> None:
        if self._help_vbo is None or self.width <= 0 or self.height <= 0:
            return
        assert GL is not None

        tex_width, tex_height = self._help_texture_shape
        if tex_width <= 0 or tex_height <= 0:
            return

        margin_px = 18.0
        scale = min(
            1.0,
            max((self.width - 2.0 * margin_px) / tex_width, 0.1),
            max((self.height - 2.0 * margin_px) / tex_height, 0.1),
        )
        draw_width = tex_width * scale
        draw_height = tex_height * scale

        x0_px = margin_px
        x1_px = x0_px + draw_width
        y0_px = margin_px
        y1_px = y0_px + draw_height

        x0 = -1.0 + 2.0 * x0_px / max(self.width, 1)
        x1 = -1.0 + 2.0 * x1_px / max(self.width, 1)
        y_top = 1.0 - 2.0 * y0_px / max(self.height, 1)
        y_bottom = 1.0 - 2.0 * y1_px / max(self.height, 1)

        vertices = np.array(
            [
                [x0, y_top, 0.0, 1.0],
                [x0, y_bottom, 0.0, 0.0],
                [x1, y_top, 1.0, 1.0],
                [x1, y_bottom, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._help_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

    def _render_help_overlay(self) -> None:
        if self._overlay_program is None or self._help_vao is None or self._help_texture is None:
            return
        assert GL is not None

        self._update_help_quad_vertices()
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self._overlay_program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._help_texture)
        set_uniform_int(self._overlay_program, "u_texture", 0)
        GL.glBindVertexArray(self._help_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def _render(self, *, mesh: Mesh | None) -> None:
        if self._camera is None or self._line_program is None or self._point_program is None:
            return
        assert GL is not None

        model = self._model_transform.copy()
        view = self._camera.view_matrix()
        projection = self._camera.projection_matrix(self.width, self.height)
        normal_matrix = np.linalg.inv(model[:3, :3]).T.astype(np.float32)

        GL.glViewport(0, 0, self.width, self.height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        if self._show_mesh and mesh is not None and self._mesh_program is not None and self._mesh_vao is not None:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if self._wireframe else GL.GL_FILL)

            GL.glUseProgram(self._mesh_program)
            set_uniform_mat4(self._mesh_program, "u_model", model)
            set_uniform_mat4(self._mesh_program, "u_view", view)
            set_uniform_mat4(self._mesh_program, "u_projection", projection)
            set_uniform_mat3(self._mesh_program, "u_normal_matrix", normal_matrix)
            set_uniform_vec3(self._mesh_program, "u_camera_pos", self._camera.position)
            set_uniform_vec3(
                self._mesh_program,
                "u_base_color",
                np.array([0.78, 0.80, 0.84], dtype=np.float32),
            )
            set_uniform_vec3(
                self._mesh_program,
                "u_light_pos",
                self._camera.position + np.array([0.0, 2.0, 2.0], dtype=np.float32),
            )

            GL.glBindVertexArray(self._mesh_vao)
            GL.glDrawElements(GL.GL_TRIANGLES, self._mesh_index_count, GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        GL.glUseProgram(self._line_program)
        for item in self._line_objects:
            kind = item["kind"]
            if kind == "ground" and not self._show_ground:
                continue
            if kind == "bbox" and not self._show_bbox:
                continue
            if kind == "hierarchy" and not self._show_hierarchy:
                continue

            current_model = model if bool(item["apply_transform"]) else np.eye(4, dtype=np.float32)
            set_uniform_mat4(self._line_program, "u_model", current_model)
            set_uniform_mat4(self._line_program, "u_view", view)
            set_uniform_mat4(self._line_program, "u_projection", projection)
            set_uniform_vec3(self._line_program, "u_color", item["color"])
            GL.glBindVertexArray(item["vao"])
            GL.glDrawElements(GL.GL_LINES, int(item["count"]), GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

        if self._show_voxels and self._voxel_vao is not None and self._voxel_count > 0:
            solid_xray = self._voxel_mode == "solid" and self._voxel_xray
            if solid_xray:
                GL.glDisable(GL.GL_DEPTH_TEST)
                GL.glDepthMask(GL.GL_FALSE)
            else:
                GL.glEnable(GL.GL_DEPTH_TEST)
                GL.glDepthMask(GL.GL_TRUE)

            GL.glUseProgram(self._point_program)
            set_uniform_mat4(self._point_program, "u_model", model)
            set_uniform_mat4(self._point_program, "u_view", view)
            set_uniform_mat4(self._point_program, "u_projection", projection)
            set_uniform_vec3(
                self._point_program,
                "u_color",
                np.array([0.26, 0.95, 0.46], dtype=np.float32),
            )
            set_uniform_float(
                self._point_program,
                "u_point_scale",
                max(self._voxel_size_reference * self.height * (0.95 if solid_xray else 1.25), 2.0),
            )
            set_uniform_float(
                self._point_program,
                "u_opacity",
                0.18 if solid_xray else 0.92,
            )
            GL.glBindVertexArray(self._voxel_vao)
            GL.glDrawArrays(GL.GL_POINTS, 0, self._voxel_count)
            GL.glBindVertexArray(0)

            if solid_xray:
                GL.glDepthMask(GL.GL_TRUE)
                GL.glEnable(GL.GL_DEPTH_TEST)

        window = self._context.window if self._context is not None else None
        if window is not None and glfw is not None:
            glfw.set_window_title(window, _window_title_text(self.title, voxel_grid=self._voxel_grid_for_reset))

    def _process_keyboard(self, window) -> None:
        assert glfw is not None
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        if self._pressed_once(window, glfw.KEY_B):
            self._show_bbox = not self._show_bbox
        if self._pressed_once(window, glfw.KEY_G):
            self._show_ground = not self._show_ground
        if self._pressed_once(window, glfw.KEY_O):
            self._show_hierarchy = not self._show_hierarchy
        if self._pressed_once(window, glfw.KEY_M):
            self._show_mesh = not self._show_mesh
        if self._pressed_once(window, glfw.KEY_V):
            self._show_voxels = not self._show_voxels
        if self._pressed_once(window, glfw.KEY_W):
            self._wireframe = not self._wireframe
        if self._pressed_once(window, glfw.KEY_X):
            self._voxel_xray = not self._voxel_xray
        if self._pressed_once(window, glfw.KEY_R):
            self._reset_camera_to_current_scene()
        if self._pressed_once(window, glfw.KEY_T):
            self._reset_model_transform()
            self._reset_camera_to_current_scene()

        if self._display_transform_enabled:
            self._process_model_keyboard(window)

    def _pressed_once(self, window, key: int) -> bool:
        assert glfw is not None
        is_pressed = glfw.get_key(window, key) == glfw.PRESS
        was_pressed = self._key_latch.get(key, False)
        self._key_latch[key] = is_pressed
        return is_pressed and not was_pressed

    @staticmethod
    def _cursor_position_callback(window, x_pos: float, y_pos: float) -> None:
        if glfw is None:
            return
        viewer = glfw.get_window_user_pointer(window)
        if viewer is None:
            return
        viewer._on_cursor_position(x_pos, y_pos)

    def _on_cursor_position(self, x_pos: float, y_pos: float) -> None:
        if self._camera is None:
            return

        if self._last_cursor is None:
            self._last_cursor = (x_pos, y_pos)
            return

        delta_x = x_pos - self._last_cursor[0]
        delta_y = y_pos - self._last_cursor[1]
        self._last_cursor = (x_pos, y_pos)

        if self._interaction_mode == "camera_orbit":
            self._camera.orbit(delta_x, delta_y)
        elif self._interaction_mode == "camera_pan":
            self._camera.pan(delta_x, delta_y, self.width, self.height)
        elif self._interaction_mode == "model_rotate":
            self._rotate_model(delta_x, delta_y)
        elif self._interaction_mode == "model_translate":
            self._translate_model(delta_x, delta_y)

    @staticmethod
    def _mouse_button_callback(window, button: int, action: int, mods: int) -> None:
        if glfw is None:
            return
        viewer = glfw.get_window_user_pointer(window)
        if viewer is None:
            return
        viewer._on_mouse_button(button, action, mods)

    def _on_mouse_button(self, button: int, action: int, mods: int) -> None:
        assert glfw is not None
        _ = mods
        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._left_mouse_down = pressed
            if pressed:
                self._interaction_mode = "camera_orbit"
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._right_mouse_down = pressed
            if pressed:
                self._interaction_mode = "camera_pan"
        if not pressed:
            self._interaction_mode = None
            self._last_cursor = None

    @staticmethod
    def _scroll_callback(window, x_offset: float, y_offset: float) -> None:
        if glfw is None:
            return
        viewer = glfw.get_window_user_pointer(window)
        if viewer is None:
            return
        viewer._on_scroll(window, y_offset)

    def _on_scroll(self, window, y_offset: float) -> None:
        _ = window
        if self._camera is not None:
            self._camera.zoom(y_offset)

    @staticmethod
    def _framebuffer_size_callback(window, width: int, height: int) -> None:
        if glfw is None:
            return
        viewer = glfw.get_window_user_pointer(window)
        if viewer is None:
            return
        viewer.width = max(int(width), 1)
        viewer.height = max(int(height), 1)

    def _apply_model_matrix(self, matrix: NDArray[np.floating]) -> None:
        self._model_transform = (np.asarray(matrix, dtype=np.float32) @ self._model_transform).astype(np.float32)

    def _camera_basis(self) -> tuple[Float32Array, Float32Array, Float32Array]:
        if self._camera is None:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            return forward, right, up
        forward = normalize(self._camera.target - self._camera.position)
        right = normalize(np.cross(forward, self._camera.world_up))
        up = normalize(np.cross(right, forward))
        return forward, right, up

    def _reset_model_transform(self) -> None:
        self._model_transform = self._initial_model_transform.copy()
        if self._mesh_for_reset is not None:
            self._transform_pivot = _transformed_center(self._mesh_for_reset, self._initial_model_transform)
        elif self._voxel_grid_for_reset is not None:
            bounds_min, bounds_max = _transform_bounds(
                self._voxel_grid_for_reset.bounds_min_xyz,
                self._voxel_grid_for_reset.bounds_max_xyz,
                self._initial_model_transform,
            )
            self._transform_pivot = ((bounds_min + bounds_max) * 0.5).astype(np.float32)
        elif self._scene_bounds_min is not None and self._scene_bounds_max is not None:
            self._transform_pivot = ((self._scene_bounds_min + self._scene_bounds_max) * 0.5).astype(np.float32)

    def _process_model_keyboard(self, window) -> None:
        assert glfw is not None
        rotation_step_degrees = 1.4
        scale_step = 0.2

        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            self._rotate_model_by_angles(yaw_degrees=-rotation_step_degrees)
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            self._rotate_model_by_angles(yaw_degrees=rotation_step_degrees)
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            self._rotate_model_by_angles(pitch_degrees=-rotation_step_degrees)
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            self._rotate_model_by_angles(pitch_degrees=rotation_step_degrees)

        scale_up = glfw.get_key(window, getattr(glfw, "KEY_KP_ADD", glfw.KEY_UNKNOWN)) == glfw.PRESS
        scale_up = scale_up or glfw.get_key(window, glfw.KEY_EQUAL) == glfw.PRESS
        scale_down = glfw.get_key(window, getattr(glfw, "KEY_KP_SUBTRACT", glfw.KEY_UNKNOWN)) == glfw.PRESS
        scale_down = scale_down or glfw.get_key(window, glfw.KEY_MINUS) == glfw.PRESS

        if scale_up:
            self._scale_model(scale_step)
        if scale_down:
            self._scale_model(-scale_step)

    def _rotate_model_by_angles(
        self,
        *,
        pitch_degrees: float = 0.0,
        yaw_degrees: float = 0.0,
    ) -> None:
        if abs(pitch_degrees) < 1e-9 and abs(yaw_degrees) < 1e-9:
            return
        pivot = self._transform_pivot
        rotation = translation_matrix(pivot)
        if abs(yaw_degrees) >= 1e-9:
            rotation = rotation @ rotation_matrix_y(yaw_degrees)
        if abs(pitch_degrees) >= 1e-9:
            rotation = rotation @ rotation_matrix_x(pitch_degrees)
        rotation = rotation @ translation_matrix(-pivot)
        self._apply_model_matrix(rotation)

    def _rotate_model(self, delta_x: float, delta_y: float) -> None:
        self._rotate_model_by_angles(
            yaw_degrees=delta_x * 0.35,
            pitch_degrees=-delta_y * 0.35,
        )

    def _translate_model(self, delta_x: float, delta_y: float) -> None:
        if self._camera is None:
            return
        _, right, up = self._camera_basis()
        scale = self._camera.distance * math.tan(math.radians(self._camera.fov_y_degrees) * 0.5)
        dx = delta_x / max(self.width, 1) * scale * 2.0
        dy = -delta_y / max(self.height, 1) * scale * 2.0
        offset = (right * dx + up * dy).astype(np.float32)
        self._apply_model_matrix(translation_matrix(offset))
        self._transform_pivot = (self._transform_pivot + offset).astype(np.float32)

    def _scale_model(self, scroll_steps: float) -> None:
        pivot = self._transform_pivot
        factor = 1.08 ** float(scroll_steps)
        scaling = translation_matrix(pivot) @ scale_matrix(factor) @ translation_matrix(-pivot)
        self._apply_model_matrix(scaling)

    def _reset_camera_to_current_scene(self) -> None:
        if self._mesh_for_reset is not None:
            mesh_bounds = _transformed_bounds(self._mesh_for_reset, self._model_transform)
            volume_bounds = None if self._voxel_grid_for_reset is None else _transform_bounds(
                self._voxel_grid_for_reset.bounds_min_xyz,
                self._voxel_grid_for_reset.bounds_max_xyz,
                self._model_transform,
            )
            self._scene_bounds_min, self._scene_bounds_max = _merge_bounds(mesh_bounds, volume_bounds)
        elif self._voxel_grid_for_reset is not None:
            self._scene_bounds_min, self._scene_bounds_max = _transform_bounds(
                self._voxel_grid_for_reset.bounds_min_xyz,
                self._voxel_grid_for_reset.bounds_max_xyz,
                self._model_transform,
            )
        if self._scene_bounds_min is not None and self._scene_bounds_max is not None:
            self._camera = OrbitCamera.from_bounds(self._scene_bounds_min, self._scene_bounds_max)

    def _cleanup(self) -> None:
        if GL is not None:
            if self._mesh_vao is not None:
                GL.glDeleteVertexArrays(1, [self._mesh_vao])
            if self._mesh_vbo is not None:
                GL.glDeleteBuffers(1, [self._mesh_vbo])
            if self._mesh_ebo is not None:
                GL.glDeleteBuffers(1, [self._mesh_ebo])
            if self._voxel_vao is not None:
                GL.glDeleteVertexArrays(1, [self._voxel_vao])
            if self._voxel_vbo is not None:
                GL.glDeleteBuffers(1, [self._voxel_vbo])
            if self._help_vao is not None:
                GL.glDeleteVertexArrays(1, [self._help_vao])
            if self._help_vbo is not None:
                GL.glDeleteBuffers(1, [self._help_vbo])
            if self._help_texture is not None:
                GL.glDeleteTextures(1, [self._help_texture])
            for item in self._line_objects:
                GL.glDeleteVertexArrays(1, [item["vao"]])
                GL.glDeleteBuffers(1, [item["vbo"]])
                GL.glDeleteBuffers(1, [item["ebo"]])

        delete_program(self._mesh_program)
        delete_program(self._line_program)
        delete_program(self._point_program)
        delete_program(self._overlay_program)

        self._mesh_program = None
        self._line_program = None
        self._point_program = None
        self._overlay_program = None
        self._mesh_vao = None
        self._mesh_vbo = None
        self._mesh_ebo = None
        self._voxel_vao = None
        self._voxel_vbo = None
        self._help_vao = None
        self._help_vbo = None
        self._help_texture = None
        self._help_texture_shape = (0, 0)
        self._line_objects = []

        if self._context is not None:
            self._context.destroy()
            self._context = None
