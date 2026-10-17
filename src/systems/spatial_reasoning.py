"""
Spatial Reasoning Engine - Integrates Neural Perception with Symbolic Scene Graphs

Combines 3D Gaussian Splatting, SAM 2 segmentation, and CLIP grounding
to build symbolic scene representations with spatial relationships.

This is the first of four neurosymbolic AI systems.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from ..neural.gaussian_splatting import GaussianSplattingEngine, GaussianPrimitive
from ..neural.segmentation import SegmentationEngine, SegmentationMask
from ..neural.clip_grounding import CLIPGroundingEngine, GroundedObject
from ..core.bridge import GroundingBridge, NeuralScene, SymbolicScene


# Define spatial ontology namespace
SPATIAL = Namespace("http://neurosymbolic-xr.org/spatial#")
QUIMA = Namespace("http://quima-vr.org/ontology#")


@dataclass
class SpatialEntity:
    """Entity in 3D space with geometric and semantic properties."""
    
    id: str
    category: str
    position: np.ndarray  # (3,) xyz center
    bbox_3d: np.ndarray  # (8, 3) 3D bounding box corners
    volume: float  # Cubic meters
    gaussians: List[GaussianPrimitive]  # Associated 3D Gaussians
    mask: SegmentationMask  # 2D segmentation mask
    grounding: GroundedObject  # Visual-language grounding
    properties: Dict[str, any]  # Semantic properties
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'category': self.category,
            'position': self.position.tolist(),
            'bbox_3d': self.bbox_3d.tolist(),
            'volume': self.volume,
            'properties': self.properties
        }


@dataclass
class SpatialRelation:
    """Spatial relationship between two entities."""
    
    subject: str  # Subject entity ID
    predicate: str  # Relationship type (on, in, above, near, etc.)
    object: str  # Object entity ID
    confidence: float  # Confidence score [0, 1]
    geometric_evidence: Dict[str, float]  # Geometric measurements
    
    def to_triple(self) -> Tuple[str, str, str]:
        """Convert to RDF triple."""
        return (self.subject, self.predicate, self.object)


class SpatialReasoningEngine:
    """
    Spatial Reasoning Engine for neurosymbolic scene understanding.
    
    Pipeline:
    1. Capture scene with 3D Gaussian Splatting
    2. Segment objects with SAM 2
    3. Ground objects with CLIP
    4. Build symbolic scene graph with spatial relationships
    5. Enable logical reasoning over scene structure
    
    Args:
        gaussian_config: Configuration for 3DGS engine
        segmentation_config: Configuration for SAM 2
        grounding_config: Configuration for CLIP
        device: Computation device
    """
    
    def __init__(
        self,
        gaussian_config: dict,
        segmentation_config: dict,
        grounding_config: dict,
        device: str = "cuda"
    ):
        self.device = device
        
        # Initialize neural components
        self.gaussian_engine = GaussianSplattingEngine(**gaussian_config)
        self.segmentation_engine = SegmentationEngine(**segmentation_config)
        self.grounding_engine = CLIPGroundingEngine(**grounding_config)
        
        # Initialize grounding bridge
        self.bridge = GroundingBridge(device=device)
        
        # Scene state
        self.entities: Dict[str, SpatialEntity] = {}
        self.relations: List[SpatialRelation] = []
        self.scene_graph: Graph = Graph()
        
        # Bind namespaces
        self.scene_graph.bind("spatial", SPATIAL)
        self.scene_graph.bind("quima", QUIMA)
        
        # Spatial relationship thresholds
        self.relation_thresholds = {
            'on': {'vertical_overlap': 0.7, 'vertical_distance': 0.1},
            'in': {'containment': 0.8},
            'above': {'vertical_distance': 0.0, 'horizontal_overlap': 0.3},
            'below': {'vertical_distance': 0.0, 'horizontal_overlap': 0.3},
            'near': {'distance': 1.0},  # meters
            'far': {'distance': 3.0},
            'left_of': {'horizontal_distance': 0.0},
            'right_of': {'horizontal_distance': 0.0},
            'in_front_of': {'depth_distance': 0.0},
            'behind': {'depth_distance': 0.0}
        }
    
    def capture_and_understand_scene(
        self,
        images: List[np.ndarray],
        camera_poses: List[np.ndarray],
        camera_intrinsics: np.ndarray,
        num_iterations: int = 10000
    ) -> SymbolicScene:
        """
        Complete pipeline: capture scene and build symbolic representation.
        
        Args:
            images: List of (H, W, 3) RGB images [0, 255]
            camera_poses: List of (4, 4) camera-to-world transforms
            camera_intrinsics: (3, 3) camera intrinsic matrix
            num_iterations: Number of 3DGS optimization iterations
            
        Returns:
            SymbolicScene with entities and relations
        """
        print("Step 1/5: Capturing scene with 3D Gaussian Splatting...")
        gaussians = self._capture_3d_scene(images, camera_poses, camera_intrinsics, num_iterations)
        
        print("Step 2/5: Segmenting objects with SAM 2...")
        masks = self._segment_objects(images[0])  # Use first view for segmentation
        
        print("Step 3/5: Grounding objects with CLIP...")
        grounded_objects = self._ground_objects(images[0], masks)
        
        print("Step 4/5: Building spatial entities...")
        entities = self._build_spatial_entities(gaussians, masks, grounded_objects)
        
        print("Step 5/5: Computing spatial relationships...")
        relations = self._compute_spatial_relations(entities)
        
        # Build scene graph
        self._build_scene_graph(entities, relations)
        
        # Create symbolic scene
        symbolic_scene = self._create_symbolic_scene(entities, relations)
        
        print(f"Scene understanding complete: {len(entities)} entities, {len(relations)} relations")
        
        return symbolic_scene
    
    def query_scene(self, query: str) -> List[Dict]:
        """
        Query scene using natural language or SPARQL.
        
        Args:
            query: Natural language query or SPARQL query
            
        Returns:
            List of query results
        """
        # Check if SPARQL query
        if query.strip().upper().startswith("SELECT"):
            return self._execute_sparql_query(query)
        else:
            return self._execute_natural_language_query(query)
    
    def find_objects(
        self,
        category: Optional[str] = None,
        properties: Optional[Dict] = None,
        relationships: Optional[List[Tuple[str, str]]] = None
    ) -> List[SpatialEntity]:
        """
        Find objects matching criteria.
        
        Args:
            category: Object category (e.g., "chair", "table")
            properties: Property constraints (e.g., {'color': 'red', 'size': 'large'})
            relationships: Relationship constraints (e.g., [('on', 'table'), ('near', 'window')])
            
        Returns:
            List of matching SpatialEntity objects
        """
        matching_entities = []
        
        for entity in self.entities.values():
            # Check category
            if category and entity.category != category:
                continue
            
            # Check properties
            if properties:
                match = True
                for prop_key, prop_value in properties.items():
                    if prop_key not in entity.properties:
                        match = False
                        break
                    if entity.properties[prop_key] != prop_value:
                        match = False
                        break
                if not match:
                    continue
            
            # Check relationships
            if relationships:
                match = True
                for rel_type, rel_object in relationships:
                    if not self._has_relationship(entity.id, rel_type, rel_object):
                        match = False
                        break
                if not match:
                    continue
            
            matching_entities.append(entity)
        
        return matching_entities
    
    def get_relationships(
        self,
        entity_id: str,
        relation_type: Optional[str] = None
    ) -> List[SpatialRelation]:
        """
        Get all relationships involving an entity.
        
        Args:
            entity_id: Entity identifier
            relation_type: Optional filter by relationship type
            
        Returns:
            List of SpatialRelation objects
        """
        relationships = []
        
        for relation in self.relations:
            if relation.subject == entity_id or relation.object == entity_id:
                if relation_type is None or relation.predicate == relation_type:
                    relationships.append(relation)
        
        return relationships
    
    def verify_constraint(self, constraint: str) -> bool:
        """
        Verify spatial constraint using symbolic reasoning.
        
        Args:
            constraint: Constraint in natural language or logic
                       (e.g., "laptop is on desk", "all chairs are near table")
            
        Returns:
            True if constraint is satisfied
        """
        # Parse constraint (simplified implementation)
        # In production, use proper NLP parsing
        
        if " is on " in constraint:
            parts = constraint.split(" is on ")
            subject = parts[0].strip()
            obj = parts[1].strip()
            
            # Find entities
            subject_entities = self.find_objects(category=subject)
            object_entities = self.find_objects(category=obj)
            
            if not subject_entities or not object_entities:
                return False
            
            # Check if any subject is on any object
            for subj_ent in subject_entities:
                for obj_ent in object_entities:
                    if self._has_relationship(subj_ent.id, 'on', obj_ent.id):
                        return True
            
            return False
        
        # Add more constraint patterns as needed
        return False
    
    # Internal methods
    
    def _capture_3d_scene(
        self,
        images: List[np.ndarray],
        camera_poses: List[np.ndarray],
        camera_intrinsics: np.ndarray,
        num_iterations: int
    ) -> List[GaussianPrimitive]:
        """Capture 3D scene using Gaussian Splatting."""
        # Convert numpy arrays to tensors
        images_tensor = [torch.from_numpy(img).float() / 255.0 for img in images]
        poses_tensor = [torch.from_numpy(pose).float() for pose in camera_poses]
        intrinsics_tensor = torch.from_numpy(camera_intrinsics).float()
        
        # Capture scene
        result = self.gaussian_engine.capture_scene(
            images=images_tensor,
            camera_poses=poses_tensor,
            camera_intrinsics=intrinsics_tensor,
            num_iterations=num_iterations
        )
        
        return result['gaussians']
    
    def _segment_objects(self, image: np.ndarray) -> List[SegmentationMask]:
        """Segment objects in image using SAM 2."""
        masks = self.segmentation_engine.segment_image(
            image=image,
            mode="automatic"
        )
        
        return masks
    
    def _ground_objects(
        self,
        image: np.ndarray,
        masks: List[SegmentationMask]
    ) -> List[GroundedObject]:
        """Ground segmented objects to natural language using CLIP."""
        mask_arrays = [mask.mask for mask in masks]
        
        grounded_objects = self.grounding_engine.ground_objects(
            image=image,
            masks=mask_arrays,
            classify=True
        )
        
        return grounded_objects
    
    def _build_spatial_entities(
        self,
        gaussians: List[GaussianPrimitive],
        masks: List[SegmentationMask],
        grounded_objects: List[GroundedObject]
    ) -> Dict[str, SpatialEntity]:
        """Build spatial entities from neural outputs."""
        entities = {}
        
        for i, (mask, grounded) in enumerate(zip(masks, grounded_objects)):
            entity_id = f"entity_{i}"
            
            # Associate Gaussians with this mask
            entity_gaussians = self._associate_gaussians_with_mask(gaussians, mask)
            
            # Compute 3D properties
            position = self._compute_entity_position(entity_gaussians)
            bbox_3d = self._compute_3d_bbox(entity_gaussians)
            volume = self._compute_volume(bbox_3d)
            
            # Extract properties
            properties = {
                'confidence': grounded.similarity,
                'area_2d': mask.area,
                'stability': mask.stability_score
            }
            
            if grounded.attributes:
                properties.update(grounded.attributes)
            
            # Create entity
            entity = SpatialEntity(
                id=entity_id,
                category=grounded.category or "unknown",
                position=position,
                bbox_3d=bbox_3d,
                volume=volume,
                gaussians=entity_gaussians,
                mask=mask,
                grounding=grounded,
                properties=properties
            )
            
            entities[entity_id] = entity
        
        self.entities = entities
        return entities
    
    def _compute_spatial_relations(
        self,
        entities: Dict[str, SpatialEntity]
    ) -> List[SpatialRelation]:
        """Compute spatial relationships between entities."""
        relations = []
        
        entity_list = list(entities.values())
        
        # Pairwise relationship computation
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i+1:]:
                # Compute all possible relationships
                entity_relations = self._compute_pairwise_relations(entity1, entity2)
                relations.extend(entity_relations)
        
        self.relations = relations
        return relations
    
    def _compute_pairwise_relations(
        self,
        entity1: SpatialEntity,
        entity2: SpatialEntity
    ) -> List[SpatialRelation]:
        """Compute spatial relationships between two entities."""
        relations = []
        
        # Geometric measurements
        pos1, pos2 = entity1.position, entity2.position
        bbox1, bbox2 = entity1.bbox_3d, entity2.bbox_3d
        
        distance = np.linalg.norm(pos1 - pos2)
        vertical_dist = pos1[2] - pos2[2]  # z-axis
        horizontal_dist = np.linalg.norm(pos1[:2] - pos2[:2])
        
        # Check "on" relationship
        if self._check_on_relation(entity1, entity2):
            relation = SpatialRelation(
                subject=entity1.id,
                predicate="on",
                object=entity2.id,
                confidence=0.9,
                geometric_evidence={'vertical_distance': abs(vertical_dist), 'distance': distance}
            )
            relations.append(relation)
        
        # Check "above" relationship
        if vertical_dist > 0.1 and horizontal_dist < 1.0:
            relation = SpatialRelation(
                subject=entity1.id,
                predicate="above",
                object=entity2.id,
                confidence=0.8,
                geometric_evidence={'vertical_distance': vertical_dist}
            )
            relations.append(relation)
        
        # Check "below" relationship
        if vertical_dist < -0.1 and horizontal_dist < 1.0:
            relation = SpatialRelation(
                subject=entity1.id,
                predicate="below",
                object=entity2.id,
                confidence=0.8,
                geometric_evidence={'vertical_distance': abs(vertical_dist)}
            )
            relations.append(relation)
        
        # Check "near" relationship
        if distance < self.relation_thresholds['near']['distance']:
            relation = SpatialRelation(
                subject=entity1.id,
                predicate="near",
                object=entity2.id,
                confidence=0.7,
                geometric_evidence={'distance': distance}
            )
            relations.append(relation)
        
        return relations
    
    def _check_on_relation(self, entity1: SpatialEntity, entity2: SpatialEntity) -> bool:
        """Check if entity1 is on entity2."""
        # Simplified check: entity1 is above entity2 with small vertical gap
        pos1, pos2 = entity1.position, entity2.position
        bbox1, bbox2 = entity1.bbox_3d, entity2.bbox_3d
        
        # Entity1 should be above entity2
        if pos1[2] <= pos2[2]:
            return False
        
        # Vertical distance should be small
        vertical_gap = pos1[2] - bbox2[:, 2].max()
        if vertical_gap > 0.1:  # 10cm threshold
            return False
        
        # Horizontal overlap
        overlap = self._compute_horizontal_overlap(bbox1, bbox2)
        if overlap < 0.3:
            return False
        
        return True
    
    def _build_scene_graph(
        self,
        entities: Dict[str, SpatialEntity],
        relations: List[SpatialRelation]
    ) -> None:
        """Build RDF scene graph."""
        # Clear existing graph
        self.scene_graph = Graph()
        self.scene_graph.bind("spatial", SPATIAL)
        self.scene_graph.bind("quima", QUIMA)
        
        # Add entities
        for entity_id, entity in entities.items():
            entity_uri = SPATIAL[entity_id]
            
            # Type
            self.scene_graph.add((entity_uri, RDF.type, SPATIAL.Entity))
            self.scene_graph.add((entity_uri, SPATIAL.category, Literal(entity.category)))
            
            # Position
            self.scene_graph.add((entity_uri, SPATIAL.x, Literal(float(entity.position[0]), datatype=XSD.float)))
            self.scene_graph.add((entity_uri, SPATIAL.y, Literal(float(entity.position[1]), datatype=XSD.float)))
            self.scene_graph.add((entity_uri, SPATIAL.z, Literal(float(entity.position[2]), datatype=XSD.float)))
            
            # Volume
            self.scene_graph.add((entity_uri, SPATIAL.volume, Literal(float(entity.volume), datatype=XSD.float)))
            
            # Properties
            for prop_key, prop_value in entity.properties.items():
                if isinstance(prop_value, (int, float)):
                    self.scene_graph.add((entity_uri, SPATIAL[prop_key], Literal(prop_value)))
                elif isinstance(prop_value, str):
                    self.scene_graph.add((entity_uri, SPATIAL[prop_key], Literal(prop_value)))
        
        # Add relations
        for relation in relations:
            subject_uri = SPATIAL[relation.subject]
            object_uri = SPATIAL[relation.object]
            predicate_uri = SPATIAL[relation.predicate]
            
            self.scene_graph.add((subject_uri, predicate_uri, object_uri))
    
    def _create_symbolic_scene(
        self,
        entities: Dict[str, SpatialEntity],
        relations: List[SpatialRelation]
    ) -> SymbolicScene:
        """Create SymbolicScene representation."""
        # Convert entities to symbolic format
        symbolic_entities = [
            {
                'id': ent.id,
                'type': ent.category,
                'position': ent.position.tolist(),
                'properties': ent.properties
            }
            for ent in entities.values()
        ]
        
        # Convert relations to triples
        symbolic_relations = [rel.to_triple() for rel in relations]
        
        # Extract properties
        symbolic_properties = {}
        for ent in entities.values():
            symbolic_properties[ent.id] = ent.properties
        
        # Generate constraints (simplified)
        symbolic_constraints = []
        
        return SymbolicScene(
            entities=symbolic_entities,
            relations=symbolic_relations,
            properties=symbolic_properties,
            constraints=symbolic_constraints
        )
    
    # Helper methods
    
    @staticmethod
    def _associate_gaussians_with_mask(
        gaussians: List[GaussianPrimitive],
        mask: SegmentationMask
    ) -> List[GaussianPrimitive]:
        """Associate 3D Gaussians with 2D segmentation mask."""
        # Simplified: return all gaussians
        # In production, project Gaussians to 2D and check mask overlap
        return gaussians[:100]  # Limit for performance
    
    @staticmethod
    def _compute_entity_position(gaussians: List[GaussianPrimitive]) -> np.ndarray:
        """Compute entity center from Gaussians."""
        if not gaussians:
            return np.zeros(3)
        
        positions = np.array([g.position.numpy() for g in gaussians])
        return positions.mean(axis=0)
    
    @staticmethod
    def _compute_3d_bbox(gaussians: List[GaussianPrimitive]) -> np.ndarray:
        """Compute 3D bounding box from Gaussians."""
        if not gaussians:
            return np.zeros((8, 3))
        
        positions = np.array([g.position.numpy() for g in gaussians])
        
        min_coords = positions.min(axis=0)
        max_coords = positions.max(axis=0)
        
        # 8 corners of bounding box
        corners = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]]
        ])
        
        return corners
    
    @staticmethod
    def _compute_volume(bbox_3d: np.ndarray) -> float:
        """Compute volume from 3D bounding box."""
        if bbox_3d.shape[0] < 8:
            return 0.0
        
        dims = bbox_3d.max(axis=0) - bbox_3d.min(axis=0)
        volume = dims[0] * dims[1] * dims[2]
        
        return float(volume)
    
    @staticmethod
    def _compute_horizontal_overlap(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute horizontal overlap between two 3D bounding boxes."""
        # Project to XY plane
        min1 = bbox1[:, :2].min(axis=0)
        max1 = bbox1[:, :2].max(axis=0)
        min2 = bbox2[:, :2].min(axis=0)
        max2 = bbox2[:, :2].max(axis=0)
        
        # Compute intersection
        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        
        if (inter_max > inter_min).all():
            inter_area = (inter_max - inter_min).prod()
        else:
            inter_area = 0.0
        
        # Compute union
        area1 = (max1 - min1).prod()
        area2 = (max2 - min2).prod()
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _has_relationship(self, subject_id: str, predicate: str, object_id: str) -> bool:
        """Check if relationship exists."""
        for relation in self.relations:
            if (relation.subject == subject_id and 
                relation.predicate == predicate and 
                relation.object == object_id):
                return True
        return False
    
    def _execute_sparql_query(self, query: str) -> List[Dict]:
        """Execute SPARQL query on scene graph."""
        results = []
        
        for row in self.scene_graph.query(query):
            result_dict = {f"var{i}": str(val) for i, val in enumerate(row)}
            results.append(result_dict)
        
        return results
    
    def _execute_natural_language_query(self, query: str) -> List[Dict]:
        """Execute natural language query (simplified implementation)."""
        # Use CLIP to find matching objects
        results = []
        
        for entity in self.entities.values():
            # Compute similarity with query
            # Simplified: just return all entities
            results.append(entity.to_dict())
        
        return results


def create_spatial_reasoning_engine(config: dict) -> SpatialReasoningEngine:
    """
    Factory function to create Spatial Reasoning Engine.
    
    Args:
        config: Configuration dict with nested configs for components
        
    Returns:
        Configured SpatialReasoningEngine
    """
    return SpatialReasoningEngine(
        gaussian_config=config.get('gaussian', {}),
        segmentation_config=config.get('segmentation', {}),
        grounding_config=config.get('grounding', {}),
        device=config.get('device', 'cuda')
    )
