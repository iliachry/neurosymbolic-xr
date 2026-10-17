"""
Integration Layer

This module contains platform integrations:
- poetics: Poetics VR platform integration
"""

from .poetics import (
    PoeticsIntegration,
    VenueContext,
    SessionContext,
    create_poetics_integration
)

__all__ = [
    'PoeticsIntegration',
    'VenueContext',
    'SessionContext',
    'create_poetics_integration',
]
