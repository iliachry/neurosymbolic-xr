"""
Core Layer - Grounding Bridge

This module contains the bidirectional grounding bridge between
neural and symbolic representations.
"""

from .bridge import (
    GroundingBridge,
    NeuralScene,
    SymbolicScene
)

__all__ = [
    'GroundingBridge',
    'NeuralScene',
    'SymbolicScene',
]
