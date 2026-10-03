"""Canonical filesystem locations, resolved from the repository root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # git-ignored bulk downloads
OUTPUT_DIR = DATA_DIR / "output"  # git-ignored runtime output (signals, sqlite)
MODELS_DIR = ROOT / "models"
MODEL_CACHE_DIR = MODELS_DIR / "cache"  # git-ignored encoder download cache
DOCS_DIR = ROOT / "docs"
STATIC_DIR = Path(__file__).resolve().parent / "api" / "static"
