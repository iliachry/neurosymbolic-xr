"""
Neurosymbolic XR Framework

A unified neurosymbolic AI architecture for Extended Reality systems,
integrating neural perception with symbolic reasoning for intelligent
spatial computing and embodied serious games (https://aste.iliachry.gr).
"""

__version__ = "0.2.0"
__author__ = "Ilias Chrysovergis"

from .core.bridge import GroundingBridge
from .integration.aste import (
    ASTENeurosymbolicVerifier,
    BiometricPacket,
    BiometricProvenance,
    GroundedFact,
    RoomVerificationResult,
    TruthValue,
)

__all__ = [
    "GroundingBridge",
    "ASTENeurosymbolicVerifier",
    "BiometricPacket",
    "BiometricProvenance",
    "GroundedFact",
    "RoomVerificationResult",
    "TruthValue",
]
