#
# HyperSPIN code - hspn_surrogate_models
#
# Copyright 2025 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM25-0396
"""HyperSPIN."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

_log_ctx = threading.local()


def set_log_context(**kwargs) -> None:
    """Set shared log context"""
    for k, v in kwargs.items():
        setattr(_log_ctx, k, v)


def clear_log_context() -> None:
    """Reset log context state."""
    _log_ctx.__dict__.clear()


class GlobalLogContextFilter(logging.Filter):
    """Filter to inject context into logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        for attr in ["rank", "world_size", "backend"]:
            value = getattr(_log_ctx, attr, None)
            if not hasattr(record, attr):
                setattr(record, attr, value)
        return True


def _patch_formatter(formatter: logging.Formatter) -> None:
    """Patch the formatter to always include rank/world_size/backend placeholders."""
    base_format = formatter._fmt
    if not base_format:
        return

    if "%(rank)" not in base_format and "%(world_size)" not in base_format:
        formatter._fmt = f"[rank=%(rank)s world_size=%(world_size)s] {base_format}"

    base_format = formatter._style._fmt
    if not base_format:
        return
    if "%(rank)" not in base_format and "%(world_size)" not in base_format:
        formatter._style._fmt = f"[rank=%(rank)s world_size=%(world_size)s] {base_format}"


def install_global_log_context() -> None:
    """Inject context into all log records and patch format strings to include rank info."""
    root_logger = logging.getLogger()
    filter_instance = GlobalLogContextFilter()

    for handler in root_logger.handlers:
        handler.addFilter(filter_instance)
        if hasattr(handler, "formatter") and handler.formatter:
            _patch_formatter(handler.formatter)

    _orig_addHandler = logging.Logger.addHandler  # Patch future handlers too

    def _addHandlerWithPatch(self, hdlr):
        hdlr.addFilter(filter_instance)
        if hasattr(hdlr, "formatter") and hdlr.formatter:
            _patch_formatter(hdlr.formatter)
        _orig_addHandler(self, hdlr)

    setattr(logging.Logger, "addHandler", _addHandlerWithPatch)


def _load_head_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "ray_address" not in data:
        raise ValueError(f"{path} missing 'ray_address'")
    return data


def _resolve_ray_addr(json_path: Optional[str] = None) -> str:
    """Get the Ray address from a JSON file (defaults to '.ray-head.json').
    Usage in YAML: address: ${rayaddr:} or ${rayaddr:/abs/path/to/file.json}
    """
    return _resolve_json_key("ray_address", json_path)


def _resolve_json_key(key: str, json_path: Optional[str] = None) -> str:
    """Get a value from a json file.
    Usage: ${jsonkey:ray_address} or ${jsonkey:port,.ray-head.json}
    """
    p = Path(json_path) if json_path else Path(".ray-head.json")
    data = _load_head_json(p)
    if key not in data:
        raise KeyError(f"{key!r} not in {p}")
    return str(data[key])


def _env_int(key: str, default: int) -> int:
    """Fetch an environment variable as int."""
    import os

    value = os.environ.get(key, default)
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"Environment variable {key} must be an int, got {value!r}") from e


def _register_oc_resolvers() -> None:
    OmegaConf.register_new_resolver("env.int", _env_int, use_cache=True)
    OmegaConf.register_new_resolver("rayaddr", _resolve_ray_addr, use_cache=False)
    OmegaConf.register_new_resolver("jsonkey", _resolve_json_key, use_cache=False)


_register_oc_resolvers()
