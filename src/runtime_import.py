"""Serialize first imports of optional numerical and ML libraries.

Python's import lock protects individual modules, but packages such as NumPy
initialize many submodules. Concurrent first imports through different
application modules have exposed partially initialized NumPy submodules on
Windows. Route lazy imports through this process-wide reentrant lock so nested
imports remain safe while other requests wait for initialization to finish.
"""

import importlib
import threading


_IMPORT_LOCK = threading.RLock()


def import_module(name: str, package: str | None = None):
    """Import *name* while excluding concurrent lazy package imports."""
    with _IMPORT_LOCK:
        return importlib.import_module(name, package)
