"""Load the verifier directory as a package so reward.py's relative imports work.

Run from the verifier directory:
    uv run --no-project --with pytest --with "math-verify[antlr4_13_2]==0.9.0" \
        --with sympy --with pylatexenc --with latex2sympy2_extended pytest -q tests
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

PACKAGE = "verl090_verifier"

if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE] = package


@pytest.fixture(scope="session")
def reward():
    return importlib.import_module(f"{PACKAGE}.reward")


@pytest.fixture(scope="session")
def origin_reward():
    return importlib.import_module(f"{PACKAGE}.reward_origin")


@pytest.fixture(scope="session")
def legacy():
    return importlib.import_module(f"{PACKAGE}.legacy_entropy_math")
