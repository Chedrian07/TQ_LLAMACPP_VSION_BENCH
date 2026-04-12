from __future__ import annotations

import os
from pathlib import Path


def bench_dir_from(anchor: str | Path) -> Path:
    path = Path(anchor).resolve()
    if path.is_file():
        path = path.parent
    return path


def project_root_from(anchor: str | Path) -> Path:
    return bench_dir_from(anchor).parent


def prepend_ld_library_path(lib_dir: str | Path) -> None:
    resolved = str(Path(lib_dir).resolve())
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in existing.split(":") if part]
    if resolved not in parts:
        parts.insert(0, resolved)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def default_server_binary(project_root: str | Path, *, build_dir: str = "build") -> Path:
    root = Path(project_root).resolve()
    return root / "llama.cpp" / build_dir / "bin" / "llama-server"
