"""Loader for the repository-local HPT differential-validation dumper."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "codex" / "differential_validation" / "hpt_debug_dump.py"
        if candidate.is_file():
            spec = spec_from_file_location("_hpt_debug_dump_impl", candidate)
            module = module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise ImportError("cannot locate codex/differential_validation/hpt_debug_dump.py")


dump_boundary = _load().dump_boundary

__all__ = ["dump_boundary"]
