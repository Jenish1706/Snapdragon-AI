import importlib

import isingedge

MODULES = ["profile", "sensitivity", "qubo", "solve", "export", "runtime", "bench"]


def test_version() -> None:
    assert isingedge.__version__


def test_subpackages_import() -> None:
    for name in MODULES:
        importlib.import_module(f"isingedge.{name}")
