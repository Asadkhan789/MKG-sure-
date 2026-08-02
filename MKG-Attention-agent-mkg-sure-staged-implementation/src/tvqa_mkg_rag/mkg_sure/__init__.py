"""Staged MKG-Sure implementation.

The package is intentionally offline-first: expensive frozen backbones are run in
separate stages and their outputs are persisted before lightweight heads are
trained. Existing MKG-Attention modules remain untouched.
"""

from .config import MKGSureConfig, load_mkg_sure_config
from .types import GraphEdge, GraphNode, GraphPath, ProvenanceRecord, SourceUnit

__all__ = [
    "MKGSureConfig",
    "GraphEdge",
    "GraphNode",
    "GraphPath",
    "ProvenanceRecord",
    "SourceUnit",
    "load_mkg_sure_config",
]
