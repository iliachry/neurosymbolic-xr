"""
Neurosymbolic XR Framework

A unified neurosymbolic AI architecture for Extended Reality systems,
integrating neural perception with symbolic reasoning for intelligent
spatial computing.
"""

__version__ = "0.1.0"
__author__ = "Ilias Chrysovergis"
__email__ = "ilias@example.com"

from neurosymbolic_xr.core.bridge import GroundingBridge
from neurosymbolic_xr.core.reasoner import HybridReasoner
from neurosymbolic_xr.core.explainer import ExplainableReasoner

# Neural components
from neurosymbolic_xr.neural.gaussian_splatting import GaussianSplattingEngine
from neurosymbolic_xr.neural.sam2_segmenter import SAM2Segmenter
from neurosymbolic_xr.neural.clip_encoder import CLIPEncoder

# Symbolic components
from neurosymbolic_xr.symbolic.knowledge_graph import KnowledgeGraph
from neurosymbolic_xr.symbolic.spatial_ontology import SpatialOntology
from neurosymbolic_xr.symbolic.spatial_reasoner import SpatialReasoner

# Systems
from neurosymbolic_xr.systems.spatial_reasoning.engine import SpatialReasoningEngine
from neurosymbolic_xr.systems.hybrid_slam.slam import HybridSLAM
from neurosymbolic_xr.systems.training_assistant.assistant import TrainingAssistant
from neurosymbolic_xr.systems.asset_generator.generator import AssetGenerator

__all__ = [
    # Core
    "GroundingBridge",
    "HybridReasoner",
    "ExplainableReasoner",
    # Neural
    "GaussianSplattingEngine",
    "SAM2Segmenter",
    "CLIPEncoder",
    # Symbolic
    "KnowledgeGraph",
    "SpatialOntology",
    "SpatialReasoner",
    # Systems
    "SpatialReasoningEngine",
    "HybridSLAM",
    "TrainingAssistant",
    "AssetGenerator",
]


class NeurosymbolicXR:
    """
    Main entry point for the Neurosymbolic XR framework.
    
    This class provides a unified interface to all four neurosymbolic systems:
    1. Spatial Reasoning Engine
    2. Hybrid SLAM
    3. VR Training Assistant
    4. Neural-Symbolic Asset Generator
    
    Example:
        >>> nxr = NeurosymbolicXR()
        >>> scene = nxr.capture_scene()
        >>> results = nxr.query("Find all horizontal surfaces")
    """
    
    def __init__(
        self,
        neural_config: str = "configs/neural_default.yaml",
        symbolic_config: str = "configs/symbolic_default.yaml",
        device: str = "cuda"
    ):
        """
        Initialize the Neurosymbolic XR framework.
        
        Args:
            neural_config: Path to neural network configuration
            symbolic_config: Path to symbolic reasoning configuration
            device: Compute device ('cuda' or 'cpu')
        """
        self.device = device
        
        # Initialize core components
        self.bridge = GroundingBridge(neural_config, symbolic_config, device)
        self.reasoner = HybridReasoner(symbolic_config)
        self.explainer = ExplainableReasoner()
        
        # Initialize systems
        self.spatial_reasoning = SpatialReasoningEngine(self.bridge, self.reasoner)
        self.hybrid_slam = HybridSLAM(self.bridge, self.reasoner)
        self.training_assistant = TrainingAssistant(self.bridge, self.reasoner)
        self.asset_generator = AssetGenerator(self.bridge, self.reasoner)
    
    def capture_scene(self, images=None, poses=None):
        """Capture 3D scene using 3D Gaussian Splatting."""
        return self.spatial_reasoning.capture_scene(images, poses)
    
    def query(self, query_string: str):
        """Execute natural language spatial query."""
        return self.spatial_reasoning.query(query_string)
    
    def track_pose(self, frame, imu_data=None):
        """Track 6DoF pose with semantic context."""
        return self.hybrid_slam.track(frame, imu_data)
    
    def monitor_procedure(self, procedure_spec):
        """Monitor and validate procedural task execution."""
        return self.training_assistant.monitor(procedure_spec)
    
    def generate_asset(self, spec):
        """Generate 3D asset with constraints."""
        return self.asset_generator.generate(spec)
