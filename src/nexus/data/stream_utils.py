"""
Stream utilities: convert dataset config to mosaicml Stream objects.

Supports multiple paths per stream via merge_index for similar MDS sources.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from streaming import Stream
from streaming.base.util import merge_index


def _to_dict(x: Any) -> dict:
    """Convert SimpleNamespace to dict for config values."""
    if isinstance(x, dict):
        return x
    if isinstance(x, SimpleNamespace):
        return {k: _to_dict(v) if isinstance(v, (dict, SimpleNamespace)) else v for k, v in vars(x).items()}
    raise TypeError(f"Expected dict or SimpleNamespace, got {type(x)}")


def _paths_to_stream_local(paths: list[str]) -> str:
    """Resolve one or more MDS paths to a single local path for a Stream.

    Single path: return as-is.
    Multiple paths: merge index.json files into a cache dir, return that path.
    """
    paths = [str(Path(p).resolve()) for p in paths]
    if len(paths) == 1:
        return paths[0]

    # Multiple paths: merge indices into a persistent cache
    path_hash = hashlib.sha256("|".join(sorted(paths)).encode()).hexdigest()[:16]
    cache_root = Path(paths[0]) / ".nexus_merged"
    cache_dir = cache_root / path_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    index_files = []
    for p in paths:
        idx = Path(p) / "index.json"
        if not idx.exists():
            raise FileNotFoundError(f"MDS index not found: {idx}")
        index_files.append(str(idx))

    merged_index = cache_dir / "index.json"
    if not merged_index.exists():
        merge_index(index_files, out=str(cache_dir), keep_local=True)

    return str(cache_dir)


def config_to_streams(streams_config: list[dict[str, Any] | SimpleNamespace]) -> list[Stream]:
    """Convert streams config (list of {paths, proportion?}) to Stream objects.

    Each entry: paths (list of str), proportion (optional float).
    Default proportion: equal across streams (1/len(streams)).
    """
    if not streams_config:
        raise ValueError("streams must be non-empty")

    streams_config = [_to_dict(s) for s in streams_config]
    n = len(streams_config)
    has_proportion = [s.get("proportion") is not None for s in streams_config]
    if any(has_proportion) and not all(has_proportion):
        raise ValueError(
            "Either all streams must set proportion, or none. "
            f"Got proportion for {sum(has_proportion)}/{n} streams."
        )

    if all(has_proportion):
        weights = [float(s["proportion"]) for s in streams_config]
    else:
        weights = [1.0 / n] * n

    streams: list[Stream] = []
    for i, entry in enumerate(streams_config):
        paths = entry.get("paths")
        if not paths:
            raise ValueError(f"stream[{i}]: paths is required and must be non-empty")
        if isinstance(paths, str):
            paths = [paths]
        local = _paths_to_stream_local(paths)
        streams.append(Stream(local=local, proportion=weights[i]))

    return streams
