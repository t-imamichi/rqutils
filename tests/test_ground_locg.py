"""Tests for :mod:`rqutils.ground_locg`.

Organized by defect. Most tests correspond to a numbered item in ``docs/locg.md``, the audit of the
previous implementation of this module; each such test names its item and the measured old failure.
That audit's closing warning shapes the design here:

    Aggregate random sampling does not exercise data-dependent branches evenly. [...] Both of my
    safety nets missed a sign error that a two-line targeted test caught at 4000/4000 failures.

So targeted per-branch tests are the backbone and randomized sweeps are a supplement.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import herm, lowest, rel_resid, symmetrize

from rqutils.ground_locg import _project_out


class TestProjectOut:
    """``_project_out`` returns either exactly zero or a vector of norm >= 0.99 (item I6)."""

    def test_vector_in_basis_span_returns_exactly_zero(self):
        """A residual lying wholly in span{x, y} must come back as exact zero, not as noise.

        The zeroing guard is what item I7's convergence check keys off: a zeroed search direction
        means {x, y} already spans the residual, so no further iteration can lower theta.
        """
        basis = (jnp.array([1., 0., 0.]), jnp.array([0., 1., 0.]))
        out = np.asarray(_project_out(basis, jnp.array([1., 1., 0.])))
        assert np.array_equal(out, np.zeros(3))

    def test_orthogonal_vector_is_not_normalized_to_unity(self):
        """The postcondition is norm >= 0.99, NOT norm == 1.

        Item I6: callers feeding this to a standard Rayleigh-Ritz step must renormalize themselves.
        At shift 1e9 a |p| of 0.999 displaced theta by 2e6, below the true minimum. This test exists
        so that a future edit "tidying" the trailing subtraction into a normalization fails loudly.
        """
        basis = (jnp.array([1., 0., 0.]), jnp.array([0., 1., 0.]))
        out = np.asarray(_project_out(basis, jnp.array([0., 0., 2.])))
        assert np.linalg.norm(out) >= 0.99
        assert np.allclose(out, [0., 0., 1.])
