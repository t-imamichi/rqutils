"""Shared setup for the sharded child scripts, which ``conftest.run_sharded_child`` subprocesses.

Each script runs under ``XLA_FLAGS=--xla_force_host_platform_device_count``, which XLA reads at backend
initialization -- after ``conftest`` has imported jax, so it cannot be an in-process test. Scripts are
files rather than inline strings so ruff and ty check them, and are named without ``test_`` so pytest
does not collect them. The parent test class owns each script's rationale.

Import this module first: it enables x64 before any array exists, as ``conftest`` does for the suite.
"""

import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

from jax.sharding import AxisType

# The fixture generators live in `test/conftest.py`, one directory up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mesh(num_devices: int) -> jax.sharding.Mesh:
    """The repo's standard mesh: one explicit axis named ``'x'`` over the first ``num_devices``."""
    return jax.make_mesh(
        (num_devices,), ("x",), devices=jax.devices()[:num_devices], axis_types=(AxisType.Explicit,)
    )


def emit(result: dict) -> None:
    """Print ``result`` as the child's only output, so a child that dies partway prints nothing."""
    print(json.dumps(result))
