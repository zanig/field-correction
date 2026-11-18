"""File and cache helpers shared by training and evaluation commands."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_content_sha256(path: Path) -> str:
    """Hash array names, dtypes, shapes and values, independent of NPZ packaging."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as cache:
        for name in sorted(cache.files):
            value = cache[name]
            value = value.astype(value.dtype.newbyteorder("<"), copy=False)
            header = json.dumps((name, value.dtype.str, value.shape), separators=(",", ":"))
            digest.update(header.encode("utf-8") + b"\0")
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
