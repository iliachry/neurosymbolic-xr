"""
Integration Layer

This module contains platform integrations:
- aste: ASTE VR (https://aste.iliachry.gr) neurosymbolic verification & biometric provenance
- poetics: Poetics VR venue management platform integration
"""

from .aste import (
    ASTENeurosymbolicVerifier,
    BiometricPacket,
    BiometricProvenance,
    GroundedFact,
    RoomVerificationResult,
    TruthValue,
)

__all__ = [
    "ASTENeurosymbolicVerifier",
    "BiometricPacket",
    "BiometricProvenance",
    "GroundedFact",
    "RoomVerificationResult",
    "TruthValue",
]

try:
    from .poetics import (
        PoeticsIntegration,
        SessionContext,
        VenueContext,
        create_poetics_integration,
    )

    __all__.extend(
        [
            "PoeticsIntegration",
            "VenueContext",
            "SessionContext",
            "create_poetics_integration",
        ]
    )
except ImportError:
    pass
