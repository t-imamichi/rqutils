"""The kernels :func:`sqd`'s ``matvec`` takes, and their validator."""

from enum import StrEnum
from typing import Any


class Matvec(StrEnum):
    """A :func:`~rqutils.sqd.sqd` matvec kernel, named by what it stores (module documentation).

    Pass a member (``Matvec.ELL``); a plain string is rejected. Members are ``str`` subclasses only so
    that the kernels' internal equality tests stay plain.
    """

    ONTHEFLY = "onthefly"
    INDICES = "indices"
    TABLES = "tables"
    PAIRS = "pairs"
    CSR = "csr"
    ELL = "ell"


_DENSE_MATVECS = (Matvec.ONTHEFLY, Matvec.INDICES, Matvec.TABLES)
#: The kernels whose operator arrays :func:`sqd` builds host-side; single-device for now.
_SPARSE_MATVECS = (Matvec.PAIRS, Matvec.CSR, Matvec.ELL)
_MATVECS = tuple(Matvec)


def _check_matvec(matvec: Any) -> None:
    """Raise unless ``matvec`` is a :class:`Matvec` member.

    Every branch on ``matvec`` is an equality test with an implicit ``else``, so an unvalidated value
    would be absorbed into some kernel rather than reported.

    Args:
        matvec: The caller's value, unvalidated.

    Raises:
        TypeError: If it is not a :class:`Matvec` member: a plain string such as ``"ell"``, or the
            removed ``cache_level`` tuple, included.
    """
    if not isinstance(matvec, Matvec):
        names = ", ".join(f"Matvec.{m.name}" for m in Matvec)
        raise TypeError(f"`matvec` must be a Matvec member, one of {names}; got {matvec!r}")
