from __future__ import annotations

from pathlib import Path
import py_compile


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
MODELS_DIR = REPO_ROOT / "models"


def test_curated_example_catalog() -> None:
    expected = {
        "01_single_mesh_pipeline.py",
        "02_shared_grid_two_meshes.py",
        "03_spatial_reuse.py",
        "README.md",
    }
    found = {path.name for path in EXAMPLES_DIR.iterdir() if path.is_file()}
    assert found == expected


def test_example_scripts_compile() -> None:
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        py_compile.compile(str(path), doraise=True)


def test_curated_model_inventory() -> None:
    expected = {
        "2,5 mm cut.STL",
        "C12.STL",
        "David.stl",
        "Eiffel_Tower.stl",
        "README.md",
    }
    found = {path.name for path in MODELS_DIR.iterdir() if path.is_file()}
    assert found == expected


def test_examples_avoid_runtime_path_scaffolding() -> None:
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        text = path.read_text()
        assert "__file__" not in text
        assert 'if __name__ == "__main__"' not in text
