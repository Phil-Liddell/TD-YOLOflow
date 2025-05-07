"""generate_requirements.py
Creates a reproducible snapshot of the current Python environment **and** bundles the entire
project into a single zip archive.

Features
--------
1. Recursively scans every *.py file starting at the project root to detect **direct** imports.
2. Writes an explicit `requirements.txt` containing the exact versions currently installed.
3. Captures the interpreter version in `python-version.txt` (e.g. 3.11.2).
4. Builds `dist/project_package.zip` that includes all project files **except** any existing
   virtual‑env folders and the archive itself.

Usage
-----
```powershell
# From the project root
python generate_requirements.py           # ⇢ creates requirements.txt, python-version.txt, dist/*.zip
```
Distribute the resulting **dist/project_package.zip** together with `setup_venv.bat` so others
can recreate the exact runtime with one double‑click.
"""

from __future__ import annotations

import ast
import os
import sys
import zipfile
from pathlib import Path

import pkg_resources

PROJECT_ROOT = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / "dist"
ARCHIVE_PATH = DIST_DIR / "project_package.zip"

# ---------------------------------------------------------------------------
# 1. Collect python sources
# ---------------------------------------------------------------------------

def collect_python_files(root: Path) -> list[Path]:
    """Return every *.py file under *root* (recursively)."""
    return [p for p in root.rglob("*.py") if p.is_file()]


# ---------------------------------------------------------------------------
# 2. Parse imports
# ---------------------------------------------------------------------------

def extract_imports(path: Path) -> set[str]:
    """Return the top‑level imported module names found inside *path*."""
    with path.open(encoding="utf‑8") as fh:
        tree = ast.parse(fh.read(), filename=str(path))

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports


# ---------------------------------------------------------------------------
# 3. Freeze dependencies
# ---------------------------------------------------------------------------

def write_requirements(imports: set[str]) -> None:
    """Generate requirements.txt based on currently installed versions."""
    installed = {dist.key: dist.version for dist in pkg_resources.working_set}

    with (PROJECT_ROOT / "requirements.txt").open("w", encoding="utf‑8") as fh:
        for name in sorted(imports):
            key = name.lower()
            if key in installed:
                fh.write(f"{name}=={installed[key]}\n")
            else:
                print(f"[WARN] {name} imported but not installed in current env.")


# ---------------------------------------------------------------------------
# 4. Snapshot interpreter version
# ---------------------------------------------------------------------------

def write_python_version() -> None:
    (PROJECT_ROOT / "python-version.txt").write_text(sys.version.split()[0] + "\n", encoding="utf‑8")


# ---------------------------------------------------------------------------
# 5. Package project
# ---------------------------------------------------------------------------

def build_zip() -> None:
    DIST_DIR.mkdir(exist_ok=True)
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()

    def _should_include(p: Path) -> bool:
        parts = {part.lower() for part in p.parts}
        return ".venv" not in parts and "dist" not in parts

    with zipfile.ZipFile(ARCHIVE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in PROJECT_ROOT.rglob("*"):
            if file_path.is_file() and _should_include(file_path):
                zf.write(file_path, file_path.relative_to(PROJECT_ROOT))

    print(f"[OK] Project packaged → {ARCHIVE_PATH.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    py_files = collect_python_files(PROJECT_ROOT)
    all_imports: set[str] = set()
    for f in py_files:
        all_imports |= extract_imports(f)

    write_requirements(all_imports)
    write_python_version()
    build_zip()

    print("\nSnapshot complete! ► requirements.txt, python-version.txt, dist/*.zip")


if __name__ == "__main__":
    main()
