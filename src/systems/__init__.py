"""
Neurosymbolic AI Systems

This module contains the four main neurosymbolic AI systems:
1. Spatial Reasoning Engine - 3DGS + Symbolic Scene Graphs
2. Hybrid SLAM - Visual-Inertial + Semantic Memory
3. Training Assistant - Hand/Gaze + Procedural Reasoning
4. Asset Generator - Diffusion Models + Geometric Constraints
"""

from .spatial_reasoning import (
    SpatialReasoningEngine,
    SpatialEntity,
    SpatialRelation,
    create_spatial_reasoning_engine
)
from .hybrid_slam import (
    HybridSLAM,
    Pose,
    Landmark,
    SemanticMemory,
    create_hybrid_slam
)
from .training_assistant import (
    TrainingAssistant,
    TrainingProcedure,
    ProcedureStep,
    HandPose,
    GazePose,
    HandGesture,
    StepStatus,
    create_training_assistant,
    create_example_procedure
)
from .asset_generator import (
    AssetGenerator,
    AssetSpecification,
    GeneratedAsset,
    GeometricConstraint,
    AssetType,
    ConstraintType,
    create_asset_generator,
    create_example_specification
)

__all__ = [
    # Spatial Reasoning
    'SpatialReasoningEngine',
    'SpatialEntity',
    'SpatialRelation',
    'create_spatial_reasoning_engine',
    
    # Hybrid SLAM
    'HybridSLAM',
    'Pose',
    'Landmark',
    'SemanticMemory',
    'create_hybrid_slam',
    
    # Training Assistant
    'TrainingAssistant',
    'TrainingProcedure',
    'ProcedureStep',
    'HandPose',
    'GazePose',
    'HandGesture',
    'StepStatus',
    'create_training_assistant',
    'create_example_procedure',
    
    # Asset Generator
    'AssetGenerator',
    'AssetSpecification',
    'GeneratedAsset',
    'GeometricConstraint',
    'AssetType',
    'ConstraintType',
    'create_asset_generator',
    'create_example_specification',
]
