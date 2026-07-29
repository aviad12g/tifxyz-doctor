"""Integrity, interoperability, and geometry review for TIFXYZ surfaces."""

from ._version import __version__
from .audit import AuditConfig, audit_mesh
from .integrity import audit_tifxyz_integrity, dumps_integrity_report
from .io import TifxyzData, load_tifxyz

__all__ = [
    "AuditConfig",
    "TifxyzData",
    "audit_mesh",
    "audit_tifxyz_integrity",
    "dumps_integrity_report",
    "load_tifxyz",
    "__version__",
]
