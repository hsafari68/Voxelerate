class VoxelerateError(Exception):
    """Base exception for Voxelerate."""


class UnsupportedMeshError(VoxelerateError):
    """Raised when a mesh file cannot be converted into a triangle mesh."""


class OpenGLUnavailableError(VoxelerateError):
    """Raised when an OpenGL context cannot be created or required bindings are missing."""


class ShaderCompilationError(VoxelerateError):
    """Raised when shader compilation or linking fails."""
