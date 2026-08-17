"""Shared runtime paths for the pipeline scripts.

Runtime data locations are configured with neutral environment-variable names.
Defaults never escape the project directory.
"""
from __future__ import annotations

import os

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _PIPELINE_DIR

RESTATE_DIR = os.path.join(_PIPELINE_DIR, "ReState")
PROMPT_DIR = os.path.join(_PIPELINE_DIR, "prompt")


def configured_path(env_name: str, default: str = "") -> str:
    """Resolve an optional path from a neutral environment-variable name."""
    value = os.environ.get(env_name, "").strip()
    if not value:
        return default
    return os.path.abspath(os.path.expanduser(value))


# External/runtime resources. Empty values must be supplied by CLI or environment.
DATASET_DIR = configured_path("DATASET_DIR", RESTATE_DIR)
LOWRES_VIDEO_DIR = configured_path("LOWRES_VIDEO_DIR")
HIGHRES_VIDEO_DIR = configured_path("HIGHRES_VIDEO_DIR")
HIGHRES_VIDEO_LIST = configured_path("HIGHRES_VIDEO_LIST")
SPLITTING_DIR = configured_path(
    "SPLITTING_DIR", os.path.join(PROJECT_ROOT, "splitting")
)


def parquet_triplet(main_kept_path: str) -> tuple[str, str, str]:
    """
    From the kept-path ``.../N_name.parquet`` derive:
    - same stem ``_output.parquet`` (full snapshot before split when applicable)
    - kept path (unchanged)
    - ``_exclude.parquet``
    """
    d = os.path.dirname(main_kept_path) or "."
    base = os.path.basename(main_kept_path)
    stem, ext = os.path.splitext(base)
    if ext.lower() != ".parquet":
        stem, ext = base, ".parquet"
    for suffix in ("_output", "_exclude"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    kept = os.path.join(d, f"{stem}{ext}")
    out_full = os.path.join(d, f"{stem}_output{ext}")
    excl = os.path.join(d, f"{stem}_exclude{ext}")
    return out_full, kept, excl


def parquet_error_path(kept_path: str) -> str:



    d = os.path.dirname(kept_path) or "."
    base = os.path.basename(kept_path)
    stem, ext = os.path.splitext(base)
    if ext.lower() != ".parquet":
        stem, ext = base, ".parquet"
    for suffix in ("_output", "_exclude", "_error"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return os.path.join(d, f"{stem}_error{ext}")
