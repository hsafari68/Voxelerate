from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import platform

from .exceptions import OpenGLUnavailableError, ShaderCompilationError

try:
    import glfw
except ImportError:
    glfw = None  # type: ignore[assignment]

try:
    from OpenGL import GL
except ImportError:
    GL = None  # type: ignore[assignment]

_glfw_context_users = 0


def require_opengl() -> None:
    if glfw is None:
        raise OpenGLUnavailableError(
            "glfw is not installed. Install Voxelerate with its GPU dependencies."
        )
    if GL is None:
        raise OpenGLUnavailableError(
            "PyOpenGL is not installed. Install Voxelerate with its GPU dependencies."
        )


def _increment_glfw_users() -> None:
    global _glfw_context_users
    require_opengl()
    assert glfw is not None
    if _glfw_context_users == 0:
        if not glfw.init():
            raise OpenGLUnavailableError("Failed to initialize GLFW.")
    _glfw_context_users += 1


def _decrement_glfw_users() -> None:
    global _glfw_context_users
    if glfw is None:
        return
    _glfw_context_users = max(0, _glfw_context_users - 1)
    if _glfw_context_users == 0:
        glfw.terminate()


@dataclass(slots=True)
class OpenGLContext:
    width: int = 64
    height: int = 64
    title: str = "Voxelerate"
    visible: bool = False
    samples: int = 4
    version_attempts: tuple[tuple[int, int], ...] = ((4, 6), (4, 5), (4, 3), (3, 3))
    window: object | None = field(init=False, default=None)

    def create(self) -> "OpenGLContext":
        require_opengl()
        assert glfw is not None

        _increment_glfw_users()
        last_error: str | None = None

        for major, minor in self.version_attempts:
            glfw.default_window_hints()
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, major)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, minor)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE if self.visible else glfw.FALSE)
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE if self.visible else glfw.FALSE)
            glfw.window_hint(glfw.SAMPLES, self.samples)
            if platform.system() == "Darwin":
                glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

            self.window = glfw.create_window(self.width, self.height, self.title, None, None)
            if self.window is not None:
                break
            last_error = f"Failed to create OpenGL {major}.{minor} context."

        if self.window is None:
            _decrement_glfw_users()
            raise OpenGLUnavailableError(last_error or "Failed to create an OpenGL context.")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if self.visible else 0)
        return self

    def destroy(self) -> None:
        if glfw is None:
            return
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        _decrement_glfw_users()

    def make_current(self) -> None:
        require_opengl()
        assert glfw is not None
        if self.window is None:
            raise OpenGLUnavailableError("OpenGL context has not been created.")
        glfw.make_context_current(self.window)

    def assert_compute_shader_support(self) -> None:
        require_opengl()
        assert GL is not None
        major = int(GL.glGetIntegerv(GL.GL_MAJOR_VERSION))
        minor = int(GL.glGetIntegerv(GL.GL_MINOR_VERSION))
        if (major, minor) < (4, 3):
            raise OpenGLUnavailableError(
                f"Compute shaders require OpenGL 4.3+, got {major}.{minor}."
            )

    def __enter__(self) -> "OpenGLContext":
        return self.create()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()


def shader_root() -> Path:
    return Path(__file__).resolve().parent / "shaders"


def read_shader(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def compile_shader(source: str, shader_type: int, *, label: str) -> int:
    require_opengl()
    assert GL is not None
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)

    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        info = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        GL.glDeleteShader(shader)
        raise ShaderCompilationError(f"Shader compilation failed for {label}:\n{info}")

    return shader


def link_program(*shader_ids: int, label: str) -> int:
    require_opengl()
    assert GL is not None
    program = GL.glCreateProgram()
    try:
        for shader_id in shader_ids:
            GL.glAttachShader(program, shader_id)
        GL.glLinkProgram(program)

        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            info = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
            raise ShaderCompilationError(f"Program link failed for {label}:\n{info}")
        return program
    finally:
        for shader_id in shader_ids:
            GL.glDetachShader(program, shader_id)
            GL.glDeleteShader(shader_id)


def set_uniform_mat4(program: int, name: str, matrix) -> None:
    require_opengl()
    assert GL is not None
    location = GL.glGetUniformLocation(program, name)
    if location == -1:
        return
    GL.glUniformMatrix4fv(location, 1, True, matrix)


def set_uniform_mat3(program: int, name: str, matrix) -> None:
    require_opengl()
    assert GL is not None
    location = GL.glGetUniformLocation(program, name)
    if location == -1:
        return
    GL.glUniformMatrix3fv(location, 1, True, matrix)


def set_uniform_vec3(program: int, name: str, vector) -> None:
    require_opengl()
    assert GL is not None
    location = GL.glGetUniformLocation(program, name)
    if location == -1:
        return
    GL.glUniform3fv(location, 1, vector)


def set_uniform_ivec3(program: int, name: str, vector) -> None:
    require_opengl()
    assert GL is not None
    location = GL.glGetUniformLocation(program, name)
    if location == -1:
        return
    GL.glUniform3iv(location, 1, vector)


def set_uniform_float(program: int, name: str, value: float) -> None:
    require_opengl()
    assert GL is not None
    location = GL.glGetUniformLocation(program, name)
    if location == -1:
        return
    GL.glUniform1f(location, float(value))


def set_uniform_int(program: int, name: str, value: int) -> None:
    require_opengl()
    assert GL is not None
    location = GL.glGetUniformLocation(program, name)
    if location == -1:
        return
    GL.glUniform1i(location, int(value))


def delete_program(program: int | None) -> None:
    if program and GL is not None:
        GL.glDeleteProgram(program)
