"""
Grounding Bridge: Neural-Symbolic Interface

This module implements the bidirectional grounding between neural perception
and symbolic reasoning, enabling seamless translation between continuous
representations and discrete logical predicates.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
from dataclasses import dataclass
from enum import Enum


class GroundingMode(Enum):
    """Grounding direction"""
    NEURAL_TO_SYMBOLIC = "n2s"  # Bottom-up perception
    SYMBOLIC_TO_NEURAL = "s2n"  # Top-down attention


@dataclass
class NeuralScene:
    """Neural scene representation"""
    gaussians: List[Dict[str, Any]]  # 3D Gaussian primitives
    masks: List[np.ndarray]  # Segmentation masks
    embeddings: np.ndarray  # CLIP embeddings
    depth_map: Optional[np.ndarray] = None
    camera_pose: Optional[np.ndarray] = None


@dataclass
class SymbolicScene:
    """Symbolic scene representation"""
    entities: List[Dict[str, Any]]  # Object instances
    relations: List[Tuple[str, str, str]]  # (predicate, subject, object)
    properties: Dict[str, Dict[str, Any]]  # Entity properties
    constraints: List[str]  # Logical constraints


class GroundingBridge:
    """
    Bidirectional bridge between neural and symbolic representations.
    
    Implements:
    - Neural → Symbolic: Perception to logic (bottom-up)
    - Symbolic → Neural: Queries to attention (top-down)
    - Uncertainty propagation
    - Explanation generation
    """
    
    def __init__(
        self,
        neural_config: str,
        symbolic_config: str,
        device: str = "cuda"
    ):
        """
        Initialize grounding bridge.
        
        Args:
            neural_config: Path to neural configuration
            symbolic_config: Path to symbolic configuration
            device: Compute device
        """
        self.device = device
        self.neural_config = neural_config
        self.symbolic_config = symbolic_config
        
        # Thresholds for grounding
        self.confidence_threshold = 0.7
        self.spatial_tolerance = 0.05  # 5cm
        self.angular_tolerance = np.deg2rad(10)  # 10 degrees
        
    def ground_scene(
        self,
        neural_scene: NeuralScene,
        mode: GroundingMode = GroundingMode.NEURAL_TO_SYMBOLIC
    ) -> SymbolicScene:
        """
        Ground neural scene to symbolic representation.
        
        Args:
            neural_scene: Neural perception output
            mode: Grounding direction
            
        Returns:
            Symbolic scene graph
        """
        if mode == GroundingMode.NEURAL_TO_SYMBOLIC:
            return self._neural_to_symbolic(neural_scene)
        else:
            raise NotImplementedError("Symbolic-to-neural grounding")
    
    def _neural_to_symbolic(self, neural_scene: NeuralScene) -> SymbolicScene:
        """
        Convert neural outputs to symbolic predicates.
        
        Pipeline:
        1. Extract object instances from Gaussians + masks
        2. Compute spatial relationships
        3. Ground visual features to semantic labels
        4. Build knowledge graph
        """
        # 1. Extract entities
        entities = self._extract_entities(
            neural_scene.gaussians,
            neural_scene.masks,
            neural_scene.embeddings
        )
        
        # 2. Compute relations
        relations = self._compute_relations(entities)
        
        # 3. Extract properties
        properties = self._extract_properties(entities, neural_scene)
        
        # 4. Generate constraints
        constraints = self._generate_constraints(entities, relations)
        
        return SymbolicScene(
            entities=entities,
            relations=relations,
            properties=properties,
            constraints=constraints
        )
    
    def _extract_entities(
        self,
        gaussians: List[Dict],
        masks: List[np.ndarray],
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Extract object instances from neural outputs.
        
        Each entity has:
        - id: Unique identifier
        - type: Semantic class
        - position: 3D centroid [x, y, z]
        - orientation: Quaternion [w, x, y, z]
        - bounding_box: [min, max] corners
        - confidence: Detection confidence
        """
        entities = []
        
        for i, (mask, embedding) in enumerate(zip(masks, embeddings)):
            # Compute 3D centroid from Gaussians within mask
            masked_gaussians = [
                g for g in gaussians 
                if self._gaussian_in_mask(g, mask)
            ]
            
            if not masked_gaussians:
                continue
            
            # Compute centroid
            positions = np.array([g['position'] for g in masked_gaussians])
            centroid = np.mean(positions, axis=0)
            
            # Compute bounding box
            bbox_min = np.min(positions, axis=0)
            bbox_max = np.max(positions, axis=0)
            
            # Estimate orientation (PCA on positions)
            orientation = self._estimate_orientation(positions)
            
            # Semantic labeling (placeholder - would use CLIP)
            semantic_type = self._classify_object(embedding)
            
            entity = {
                'id': f'obj_{i:03d}',
                'type': semantic_type,
                'position': centroid.tolist(),
                'orientation': orientation.tolist(),
                'bounding_box': {
                    'min': bbox_min.tolist(),
                    'max': bbox_max.tolist()
                },
                'confidence': 0.85,  # Placeholder
                'embedding': embedding
            }
            
            entities.append(entity)
        
        return entities
    
    def _compute_relations(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        """
        Compute spatial relationships between entities.
        
        Relations:
        - on(A, B): A is supported by B
        - in(A, B): A is contained within B
        - above(A, B): A is vertically above B
        - near(A, B): A is close to B
        - facing(A, B): A is oriented toward B
        """
        relations = []
        
        for i, entity_a in enumerate(entities):
            for j, entity_b in enumerate(entities):
                if i >= j:  # Avoid duplicates and self-relations
                    continue
                
                # Check spatial predicates
                pos_a = np.array(entity_a['position'])
                pos_b = np.array(entity_b['position'])
                bbox_a = entity_a['bounding_box']
                bbox_b = entity_b['bounding_box']
                
                # On relation: A bottom touches B top
                if self._is_on(pos_a, bbox_a, pos_b, bbox_b):
                    relations.append(('on', entity_a['id'], entity_b['id']))
                
                # Above relation: A is higher than B
                if pos_a[2] > pos_b[2] + 0.1:  # 10cm threshold
                    relations.append(('above', entity_a['id'], entity_b['id']))
                
                # Near relation: Distance < threshold
                distance = np.linalg.norm(pos_a - pos_b)
                if distance < 0.5:  # 50cm threshold
                    relations.append(('near', entity_a['id'], entity_b['id']))
                
                # In relation: A inside B's bounding box
                if self._is_inside(pos_a, bbox_b):
                    relations.append(('in', entity_a['id'], entity_b['id']))
        
        return relations
    
    def _extract_properties(
        self,
        entities: List[Dict[str, Any]],
        neural_scene: NeuralScene
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract semantic properties for each entity.
        
        Properties:
        - color: Dominant color
        - material: Surface material
        - shape: Geometric shape
        - size: Physical dimensions
        - affordances: Functional capabilities
        """
        properties = {}
        
        for entity in entities:
            entity_id = entity['id']
            bbox = entity['bounding_box']
            
            # Compute size
            size = np.array(bbox['max']) - np.array(bbox['min'])
            volume = np.prod(size)
            
            # Extract color (placeholder - would analyze Gaussians)
            color = self._extract_dominant_color(entity, neural_scene)
            
            # Infer affordances from type
            affordances = self._infer_affordances(entity['type'])
            
            properties[entity_id] = {
                'dimensions': size.tolist(),
                'volume': float(volume),
                'color': color,
                'affordances': affordances,
                'horizontal': self._is_horizontal_surface(entity),
                'graspable': volume < 0.001  # < 1L
            }
        
        return properties
    
    def _generate_constraints(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Tuple[str, str, str]]
    ) -> List[str]:
        """
        Generate logical constraints from scene structure.
        
        Examples:
        - "forall X, on(X, floor) => supported(X)"
        - "forall X Y, in(X, Y) => smaller(X, Y)"
        """
        constraints = []
        
        # Transitivity of support
        constraints.append(
            "forall X Y Z, (on(X, Y) & on(Y, Z)) => supports(Z, X)"
        )
        
        # Containment implies smaller
        constraints.append(
            "forall X Y, in(X, Y) => smaller(X, Y)"
        )
        
        # Objects on surfaces must be above them
        constraints.append(
            "forall X Y, on(X, Y) => above(X, Y)"
        )
        
        return constraints
    
    # Helper methods
    
    def _gaussian_in_mask(self, gaussian: Dict, mask: np.ndarray) -> bool:
        """Check if Gaussian center projects into mask."""
        # Placeholder - would project 3D position to 2D
        return True
    
    def _estimate_orientation(self, positions: np.ndarray) -> np.ndarray:
        """Estimate orientation via PCA."""
        # Compute principal axes
        centered = positions - np.mean(positions, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Convert to quaternion (placeholder)
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    def _classify_object(self, embedding: np.ndarray) -> str:
        """Classify object from CLIP embedding."""
        # Placeholder - would compare with text embeddings
        return "unknown"
    
    def _is_on(
        self,
        pos_a: np.ndarray,
        bbox_a: Dict,
        pos_b: np.ndarray,
        bbox_b: Dict
    ) -> bool:
        """Check if A is resting on B."""
        # A's bottom should be near B's top
        a_bottom = bbox_a['min'][2]
        b_top = bbox_b['max'][2]
        
        vertical_gap = abs(a_bottom - b_top)
        
        # Check horizontal overlap
        a_min_xy = np.array(bbox_a['min'][:2])
        a_max_xy = np.array(bbox_a['max'][:2])
        b_min_xy = np.array(bbox_b['min'][:2])
        b_max_xy = np.array(bbox_b['max'][:2])
        
        overlap_x = min(a_max_xy[0], b_max_xy[0]) > max(a_min_xy[0], b_min_xy[0])
        overlap_y = min(a_max_xy[1], b_max_xy[1]) > max(a_min_xy[1], b_min_xy[1])
        
        return vertical_gap < self.spatial_tolerance and overlap_x and overlap_y
    
    def _is_inside(self, pos: np.ndarray, bbox: Dict) -> bool:
        """Check if position is inside bounding box."""
        min_corner = np.array(bbox['min'])
        max_corner = np.array(bbox['max'])
        
        return np.all(pos >= min_corner) and np.all(pos <= max_corner)
    
    def _extract_dominant_color(
        self,
        entity: Dict,
        neural_scene: NeuralScene
    ) -> str:
        """Extract dominant color from Gaussians."""
        # Placeholder - would analyze SH coefficients
        return "unknown"
    
    def _infer_affordances(self, object_type: str) -> List[str]:
        """Infer functional affordances from object type."""
        affordance_map = {
            'table': ['support', 'place_on'],
            'chair': ['sit_on', 'support'],
            'container': ['contain', 'hold'],
            'tool': ['grasp', 'manipulate'],
            'surface': ['support', 'place_on']
        }
        return affordance_map.get(object_type, [])
    
    def _is_horizontal_surface(self, entity: Dict) -> bool:
        """Check if entity is a horizontal surface."""
        # Check if orientation is mostly upright
        orientation = np.array(entity['orientation'])
        # Placeholder - would check quaternion
        return True
    
    def explain_grounding(
        self,
        neural_scene: NeuralScene,
        symbolic_scene: SymbolicScene
    ) -> str:
        """
        Generate human-readable explanation of grounding process.
        
        Returns:
            Explanation text
        """
        explanation = "Grounding Process:\n\n"
        
        explanation += f"Detected {len(symbolic_scene.entities)} objects:\n"
        for entity in symbolic_scene.entities[:5]:  # Show first 5
            explanation += f"  - {entity['id']}: {entity['type']} at {entity['position']}\n"
        
        explanation += f"\nIdentified {len(symbolic_scene.relations)} spatial relations:\n"
        for rel in symbolic_scene.relations[:5]:
            explanation += f"  - {rel[0]}({rel[1]}, {rel[2]})\n"
        
        return explanation
