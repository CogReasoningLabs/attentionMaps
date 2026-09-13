"""Checksums, PDF summary, and deterministic ZIP packaging."""

from __future__ import annotations

import hashlib
import json
import platform
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(
    root: Path, *, excluded: Iterable[Path] = (), workers: int = 4
) -> list[dict[str, Any]]:
    """Hash output files concurrently; disk I/O is the limiting operation."""

    excluded_resolved = {path.resolve() for path in excluded}
    base = root if root.is_dir() else root.parent
    candidates = sorted(root.rglob("*")) if root.is_dir() else [root]
    paths = [
        path
        for path in candidates
        if path.is_file()
        and not path.is_symlink()
        and path.resolve() not in excluded_resolved
    ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        return list(executor.map(lambda path: _inventory_item(base, path), paths))


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": __import__("os").cpu_count(),
    }


def write_pdf_report(
    path: Path,
    *,
    title: str,
    summary: Mapping[str, Any],
    image_paths: Iterable[Path],
) -> Path:
    """Build a portable PDF using only matplotlib's existing dependency."""

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        cover = plt.figure(figsize=(8.27, 11.69))
        cover.text(0.08, 0.94, title, fontsize=20, fontweight="bold", va="top")
        lines = [f"{key}: {value}" for key, value in summary.items()]
        cover.text(0.08, 0.88, "\n".join(lines), fontsize=10, va="top", wrap=True)
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)
        for image_path in image_paths:
            if not image_path.is_file():
                continue
            image = mpimg.imread(image_path)
            figure, axis = plt.subplots(figsize=(11.69, 8.27))
            axis.imshow(image)
            axis.set_title(image_path.stem.replace("_", " ").title())
            axis.set_axis_off()
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    return path


def write_manifest(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def write_checksums(root: Path, inventory: Iterable[Mapping[str, Any]]) -> Path:
    path = root / "checksums.sha256"
    path.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in inventory),
        encoding="utf-8",
    )
    return path


def create_zip(source_dir: Path, destination: Path) -> Path:
    """Create a Zip64 archive with stable member ordering and relative paths."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as bundle:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and not path.is_symlink():
                bundle.write(path, path.relative_to(source_dir))
    temporary.replace(destination)
    return destination


def _inventory_item(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
