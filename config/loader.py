# config/loader.py
from __future__ import annotations
import pathlib
from typing import Any
import yaml  # pip install pyyaml
from pydantic import TypeAdapter
from .models import AppConfig

_DEF_PATH = pathlib.Path(__file__).with_name("config.yaml")

def load_config(path: str | None = None) -> AppConfig:
    p = pathlib.Path(path) if path else _DEF_PATH
    raw: dict[str, Any] = {}
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return TypeAdapter(AppConfig).validate_python(raw)

cfg: AppConfig = load_config()
