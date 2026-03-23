from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .grid import VoxelizationGrid
from .mesh import Mesh
from .opengl import (
    GL,
    OpenGLContext,
    compile_shader,
    delete_program,
    link_program,
    read_shader,
    require_opengl,
    set_uniform_int,
    set_uniform_ivec3,
    set_uniform_vec3,
    shader_root,
)
from .voxel_grid import VoxelGrid


@dataclass(slots=True)
class OpenGLVoxelizer:
    """GPU voxelizer using OpenGL compute shaders."""

    visible_context: bool = False
    context_size: tuple[int, int] = (64, 64)
    _context: OpenGLContext | None = field(init=False, default=None)
    _surface_program: int | None = field(init=False, default=None)
    _solid_program: int | None = field(init=False, default=None)

    def __enter__(self) -> "OpenGLVoxelizer":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._context is not None:
            return

        require_opengl()
        self._context = OpenGLContext(
            width=self.context_size[0],
            height=self.context_size[1],
            title="Voxelerate Voxelizer",
            visible=self.visible_context,
        ).create()
        self._context.assert_compute_shader_support()
        self._surface_program = self._build_compute_program("surface_voxelize.comp", "surface")
        self._solid_program = self._build_compute_program("solid_voxelize.comp", "solid")

    def close(self) -> None:
        delete_program(self._surface_program)
        delete_program(self._solid_program)
        self._surface_program = None
        self._solid_program = None

        if self._context is not None:
            self._context.destroy()
            self._context = None

    def _build_compute_program(self, filename: str, label: str) -> int:
        assert GL is not None
        source = read_shader(shader_root() / filename)
        shader = compile_shader(source, GL.GL_COMPUTE_SHADER, label=f"{label}-compute")
        return link_program(shader, label=f"{label}-compute-program")

    def voxelize(
        self,
        mesh: Mesh,
        *,
        voxel_size: float | tuple[float, float, float],
        mode: str = "solid",
        padding: float | tuple[float, float, float] = 0.0,
        transform: NDArray[np.floating] | None = None,
        pixel_size: float | None = None,
    ) -> VoxelGrid:
        grid = VoxelizationGrid.from_mesh(
            mesh,
            voxel_size=voxel_size,
            padding=padding,
            transform=transform,
            pixel_size=pixel_size,
        )
        return self.voxelize_on_grid(mesh, grid=grid, mode=mode, transform=transform)

    def voxelize_on_grid(
        self,
        mesh: Mesh,
        *,
        grid: VoxelizationGrid,
        mode: str = "solid",
        transform: NDArray[np.floating] | None = None,
    ) -> VoxelGrid:
        if self._context is None:
            self.open()

        if mode not in {"surface", "solid"}:
            raise ValueError("mode must be either 'surface' or 'solid'")

        transformed_vertices = mesh.transformed_vertices(transform)
        triangle_buffer = transformed_vertices[mesh.faces.reshape(-1)].reshape(-1, 9).astype(np.float32)
        raw_volume = self._dispatch(
            triangles=triangle_buffer,
            grid=grid,
            mode=mode,
        )

        if mode == "solid":
            data = (raw_volume & 1).astype(np.uint8)
        else:
            data = (raw_volume > 0).astype(np.uint8)

        mesh_bounds_min = transformed_vertices.min(axis=0).astype(np.float32)
        mesh_bounds_max = transformed_vertices.max(axis=0).astype(np.float32)

        metadata = {
            "backend": "opengl",
            "source_mesh": str(mesh.source_path) if mesh.source_path is not None else None,
            "mesh_name": mesh.name,
            "mesh_bounds_min_xyz": mesh_bounds_min.tolist(),
            "mesh_bounds_max_xyz": mesh_bounds_max.tolist(),
            "grid_size_xyz": grid.grid_size_xyz.astype(np.int32).tolist(),
            "voxel_size_xyz": grid.voxel_size_xyz.astype(np.float32).tolist(),
            "applied_transform": None if transform is None else np.asarray(transform, dtype=np.float32).tolist(),
        }

        return VoxelGrid(
            data=data,
            bounds_min_xyz=grid.bounds_min_xyz,
            bounds_max_xyz=grid.bounds_max_xyz,
            voxel_size_xyz=grid.voxel_size_xyz,
            mode=mode,
            pixel_size=grid.pixel_size,
            metadata=metadata,
        )

    def _dispatch(
        self,
        *,
        triangles: NDArray[np.float32],
        grid: VoxelizationGrid,
        mode: str,
    ) -> NDArray[np.uint32]:
        assert GL is not None
        nx, ny, nz = map(int, grid.grid_size_xyz.tolist())
        triangle_count = int(triangles.shape[0])

        program = self._solid_program if mode == "solid" else self._surface_program
        if program is None:
            raise RuntimeError("Voxelizer program has not been initialized.")

        triangle_ssbo = GL.glGenBuffers(1)
        voxel_texture = GL.glGenTextures(1)

        try:
            GL.glUseProgram(program)

            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, triangle_ssbo)
            GL.glBufferData(
                GL.GL_SHADER_STORAGE_BUFFER,
                triangles.nbytes,
                np.ascontiguousarray(triangles, dtype=np.float32),
                GL.GL_STATIC_DRAW,
            )
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, triangle_ssbo)

            GL.glBindTexture(GL.GL_TEXTURE_3D, voxel_texture)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
            GL.glTexStorage3D(GL.GL_TEXTURE_3D, 1, GL.GL_R32UI, nx, ny, nz)

            zeros = np.zeros((nz, ny, nx), dtype=np.uint32)
            GL.glTexSubImage3D(
                GL.GL_TEXTURE_3D,
                0,
                0,
                0,
                0,
                nx,
                ny,
                nz,
                GL.GL_RED_INTEGER,
                GL.GL_UNSIGNED_INT,
                zeros,
            )
            GL.glBindImageTexture(1, voxel_texture, 0, GL.GL_TRUE, 0, GL.GL_READ_WRITE, GL.GL_R32UI)

            set_uniform_vec3(program, "unit", grid.voxel_size_xyz)
            set_uniform_vec3(program, "bbox_min", grid.bounds_min_xyz)
            set_uniform_ivec3(program, "grid_size", grid.grid_size_xyz)
            set_uniform_int(program, "n_triangles", triangle_count)

            work_group_size_x = 64
            work_groups_x = max(1, math.ceil(max(triangle_count, 1) / work_group_size_x))
            GL.glDispatchCompute(work_groups_x, 1, 1)
            GL.glMemoryBarrier(
                GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                | GL.GL_TEXTURE_UPDATE_BARRIER_BIT
                | GL.GL_BUFFER_UPDATE_BARRIER_BIT
            )

            volume = np.zeros((nz, ny, nx), dtype=np.uint32)
            GL.glGetTexImage(
                GL.GL_TEXTURE_3D,
                0,
                GL.GL_RED_INTEGER,
                GL.GL_UNSIGNED_INT,
                volume,
            )
            return volume
        finally:
            GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            GL.glDeleteTextures(1, [voxel_texture])
            GL.glDeleteBuffers(1, [triangle_ssbo])
