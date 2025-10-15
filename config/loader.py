# config/loader.py
from __future__ import annotations
import functools
import pathlib
from typing import Any
import yaml  # pip install pyyaml
from pydantic import TypeAdapter
from .models import AppConfig

_DEF_PATH = pathlib.Path(__file__).with_name("config.yaml")

def load_config(path: str | None = None) -> AppConfig:
    """設定ファイルを読み込み `AppConfig` を生成する。"""
    p = pathlib.Path(path) if path else _DEF_PATH
    raw: dict[str, Any] = {}
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return TypeAdapter(AppConfig).validate_python(raw)

@functools.lru_cache(maxsize=1)
def load_default_config() -> AppConfig:
    """デフォルト設定をキャッシュ付きで返すヘルパー。"""
    return load_config()
