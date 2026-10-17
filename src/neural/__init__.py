"""
Neural Layer - Perception Components

This module contains neural network components for perception:
- gaussian_splatting: 3D Gaussian Splatting for scene capture
- segmentation: SAM 2 for zero-shot segmentation
- clip_grounding: CLIP for visual-language grounding
"""

from .gaussian_splatting import GaussianSplattingEngine, GaussianPrimitive, create_gaussian_engine
from .segmentation import SegmentationEngine, SegmentationMask, create_segmentation_engine
from .clip_grounding import CLIPGroundingEngine, GroundedObject, create_grounding_engine

__all__ = [
    # Gaussian Splatting
    'GaussianSplattingEngine',
    'GaussianPrimitive',
    'create_gaussian_engine',
    
    # Segmentation
    'SegmentationEngine',
    'SegmentationMask',
    'create_segmentation_engine',
    
    # CLIP Grounding
    'CLIPGroundingEngine',
    'GroundedObject',
    'create_grounding_engine',
]
