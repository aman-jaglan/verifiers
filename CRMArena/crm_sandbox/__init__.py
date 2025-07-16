"""crm_sandbox package root.

This file allows `import crm_sandbox` to succeed when the sandbox code is
vendored inside another repository.  It exposes the commonly-used sub‐modules
so external projects (e.g. Verifiers) can simply do::

    from crm_sandbox.env import TOOLS

without worrying about the internal structure.
"""
from importlib import import_module
import sys as _sys
from pathlib import Path as _Path

# Ensure that relative imports inside sub-packages work even when crm_sandbox
# is added to sys.path via a .egg-link.
_this_dir = _Path(__file__).resolve().parent
if str(_this_dir) not in _sys.path:
    _sys.path.insert(0, str(_this_dir))

# Re-export most-used sub-packages for convenience
for _name in [
    "agents",
    "env",
    "data",
]:
    try:
        globals()[_name] = import_module(f"crm_sandbox.{_name}")
    except ModuleNotFoundError:
        pass

del _import_module, _sys, _Path, _this_dir, _name 