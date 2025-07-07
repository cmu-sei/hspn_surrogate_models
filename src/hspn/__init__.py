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

import logging
import threading

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
