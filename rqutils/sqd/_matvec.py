"""The kernel names :func:`sqd`'s ``matvec`` takes, and their validator."""

from typing import Any, Literal, get_args

type DenseMatvec = Literal["onthefly", "indices", "tables"]
type SparseMatvec = Literal["pairs", "csr", "ell"]
type Matvec = DenseMatvec | SparseMatvec
_DENSE_MATVECS: tuple[DenseMatvec, ...] = get_args(DenseMatvec.__value__)
#: The kernels whose operator arrays :func:`sqd` builds host-side; single-device for now.
_SPARSE_MATVECS: tuple[SparseMatvec, ...] = get_args(SparseMatvec.__value__)
_MATVECS = _DENSE_MATVECS + _SPARSE_MATVECS


def _check_matvec(matvec: Any) -> None:
    """Raise unless ``matvec`` is one of the kernel names in ``_MATVECS``.

    Every branch on ``matvec`` is an equality test with an implicit ``else``, so an unvalidated value
    would be absorbed into some kernel rather than reported.

    Args:
        matvec: The caller's value, unvalidated.

    Raises:
        TypeError: If it is not a ``str`` (the removed ``cache_level`` tuple included).
        ValueError: If it is not one of those names.
    """
    if not isinstance(matvec, str):
        raise TypeError(f"`matvec` must be a str, one of {_MATVECS}; got {matvec!r}")
    if matvec not in _MATVECS:
        raise ValueError(f"`matvec` is {matvec!r}, but must be one of {_MATVECS}")
