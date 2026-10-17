"""
Visual-Language Grounding Engine using CLIP

Provides open-vocabulary object recognition and visual-language alignment.
Enables natural language queries over visual scenes.

References:
- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (2021)
- https://github.com/openai/CLIP
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

try:
    import clip
except ImportError:
    print("Warning: CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")


@dataclass
class GroundedObject:
    """Object grounded to natural language description."""
    
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: Optional[np.ndarray]  # (H, W) binary mask
    text_query: str  # Natural language description
    similarity: float  # CLIP similarity score [0, 1]
    embedding: np.ndarray  # Visual embedding vector
    category: Optional[str] = None  # Predicted category
    attributes: Optional[Dict[str, float]] = None  # Attribute scores
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'bbox': self.bbox,
            'text_query': self.text_query,
            'similarity': self.similarity,
            'category': self.category,
            'attributes': self.attributes
        }


class CLIPGroundingEngine:
    """
    Visual-language grounding engine using CLIP.
    
    Features:
    - Open-vocabulary object recognition
    - Natural language scene queries
    - Zero-shot classification
    - Visual similarity search
    - Attribute recognition
    - Spatial relationship grounding
    
    Args:
        model_name: CLIP model variant ('RN50', 'RN101', 'ViT-B/32', 'ViT-L/14')
        device: Computation device
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L/14",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        
        # Load CLIP model
        try:
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load CLIP model: {e}")
            self.model = None
            self.preprocess = None
        
        # Text embedding cache
        self.text_cache = {}
        
        # Common object categories for zero-shot classification
        self.common_categories = [
            "person", "chair", "table", "desk", "bed", "sofa", "door", "window",
            "monitor", "keyboard", "mouse", "laptop", "phone", "book", "cup",
            "bottle", "plant", "lamp", "clock", "picture", "shelf", "cabinet",
            "floor", "wall", "ceiling", "box", "bag", "container"
        ]
        
        # Common attributes
        self.common_attributes = {
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray'],
            'size': ['small', 'medium', 'large'],
            'material': ['wood', 'metal', 'plastic', 'glass', 'fabric', 'paper'],
            'state': ['open', 'closed', 'empty', 'full', 'clean', 'dirty']
        }
        
        # Precompute category embeddings
        self._precompute_category_embeddings()
    
    def _precompute_category_embeddings(self) -> None:
        """Precompute text embeddings for common categories."""
        if self.model is None:
            return
        
        with torch.no_grad():
            # Category embeddings
            category_texts = [f"a photo of a {cat}" for cat in self.common_categories]
            category_tokens = clip.tokenize(category_texts).to(self.device)
            self.category_embeddings = self.model.encode_text(category_tokens)
            self.category_embeddings = F.normalize(self.category_embeddings, dim=-1)
            
            # Attribute embeddings
            self.attribute_embeddings = {}
            for attr_type, attr_values in self.common_attributes.items():
                attr_texts = [f"{val}" for val in attr_values]
                attr_tokens = clip.tokenize(attr_texts).to(self.device)
                attr_embeds = self.model.encode_text(attr_tokens)
                attr_embeds = F.normalize(attr_embeds, dim=-1)
                self.attribute_embeddings[attr_type] = attr_embeds
    
    def ground_objects(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        text_queries: Optional[List[str]] = None,
        classify: bool = True
    ) -> List[GroundedObject]:
        """
        Ground segmented objects to natural language descriptions.
        
        Args:
            image: (H, W, 3) RGB image [0, 255]
            masks: List of (H, W) binary masks
            text_queries: Optional list of text queries to match against
            classify: Whether to classify objects into categories
            
        Returns:
            List of GroundedObject instances
        """
        if self.model is None:
            return self._fallback_grounding(image, masks)
        
        grounded_objects = []
        
        # Extract visual features for each mask
        for mask in masks:
            # Compute bounding box
            bbox = self._mask_to_bbox(mask)
            
            # Extract masked region
            masked_image = self._extract_masked_region(image, mask, bbox)
            
            # Encode visual features
            visual_embedding = self._encode_image(masked_image)
            
            # Match against text queries or classify
            if text_queries:
                best_query, similarity = self._match_text_queries(
                    visual_embedding, text_queries
                )
                category = None
            else:
                best_query = None
                similarity = 0.0
                category = None
            
            if classify:
                category, category_score = self._classify_object(visual_embedding)
                if best_query is None:
                    best_query = category
                    similarity = category_score
            
            # Recognize attributes
            attributes = self._recognize_attributes(visual_embedding)
            
            grounded_obj = GroundedObject(
                bbox=bbox,
                mask=mask,
                text_query=best_query or "unknown",
                similarity=float(similarity),
                embedding=visual_embedding.cpu().numpy(),
                category=category,
                attributes=attributes
            )
            grounded_objects.append(grounded_obj)
        
        return grounded_objects
    
    def query_scene(
        self,
        image: np.ndarray,
        query: str,
        masks: Optional[List[np.ndarray]] = None,
        top_k: int = 5
    ) -> List[GroundedObject]:
        """
        Query scene with natural language and return matching objects.
        
        Args:
            image: (H, W, 3) RGB image
            query: Natural language query (e.g., "red chair", "laptop on desk")
            masks: Optional pre-computed masks (if None, use sliding window)
            top_k: Number of top matches to return
            
        Returns:
            List of top-k GroundedObject instances sorted by similarity
        """
        if self.model is None:
            return []
        
        # Encode text query
        query_embedding = self._encode_text(query)
        
        if masks is not None:
            # Use provided masks
            grounded_objects = self.ground_objects(image, masks, [query], classify=False)
        else:
            # Use sliding window approach
            grounded_objects = self._sliding_window_query(image, query_embedding)
        
        # Sort by similarity and return top-k
        grounded_objects.sort(key=lambda x: x.similarity, reverse=True)
        return grounded_objects[:top_k]
    
    def compute_similarity(
        self,
        image: np.ndarray,
        text: str,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute CLIP similarity between image (or masked region) and text.
        
        Args:
            image: (H, W, 3) RGB image
            text: Text description
            mask: Optional binary mask to focus on specific region
            
        Returns:
            Similarity score [0, 1]
        """
        if self.model is None:
            return 0.0
        
        # Apply mask if provided
        if mask is not None:
            bbox = self._mask_to_bbox(mask)
            image = self._extract_masked_region(image, mask, bbox)
        
        # Encode image and text
        image_embedding = self._encode_image(image)
        text_embedding = self._encode_text(text)
        
        # Compute cosine similarity
        similarity = (image_embedding @ text_embedding.T).item()
        
        # Normalize to [0, 1]
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def verify_relationship(
        self,
        image: np.ndarray,
        object1_mask: np.ndarray,
        object2_mask: np.ndarray,
        relationship: str
    ) -> float:
        """
        Verify spatial relationship between two objects using CLIP.
        
        Args:
            image: (H, W, 3) RGB image
            object1_mask: Binary mask for first object
            object2_mask: Binary mask for second object
            relationship: Relationship description (e.g., "on", "next to", "above")
            
        Returns:
            Confidence score [0, 1]
        """
        if self.model is None:
            return 0.0
        
        # Get object categories
        obj1_grounded = self.ground_objects(image, [object1_mask], classify=True)[0]
        obj2_grounded = self.ground_objects(image, [object2_mask], classify=True)[0]
        
        obj1_cat = obj1_grounded.category or "object"
        obj2_cat = obj2_grounded.category or "object"
        
        # Create relationship query
        relationship_text = f"{obj1_cat} {relationship} {obj2_cat}"
        
        # Extract combined region
        combined_mask = np.logical_or(object1_mask, object2_mask).astype(np.uint8)
        bbox = self._mask_to_bbox(combined_mask)
        region = self._extract_masked_region(image, combined_mask, bbox)
        
        # Compute similarity
        similarity = self.compute_similarity(region, relationship_text)
        
        return similarity
    
    def find_objects_by_attributes(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        attributes: Dict[str, str]
    ) -> List[GroundedObject]:
        """
        Find objects matching specified attributes.
        
        Args:
            image: (H, W, 3) RGB image
            masks: List of object masks
            attributes: Dict of attribute_type -> value (e.g., {'color': 'red', 'size': 'large'})
            
        Returns:
            List of matching GroundedObject instances
        """
        # Ground all objects
        grounded_objects = self.ground_objects(image, masks, classify=True)
        
        # Filter by attributes
        matching_objects = []
        
        for obj in grounded_objects:
            if obj.attributes is None:
                continue
            
            matches = True
            for attr_type, attr_value in attributes.items():
                if attr_type not in obj.attributes:
                    matches = False
                    break
                
                # Check if specified value has high score
                if attr_value not in obj.attributes[attr_type]:
                    matches = False
                    break
                
                if obj.attributes[attr_type][attr_value] < 0.5:
                    matches = False
                    break
            
            if matches:
                matching_objects.append(obj)
        
        return matching_objects
    
    # Helper methods
    
    def _encode_image(self, image: np.ndarray) -> Tensor:
        """
        Encode image to CLIP embedding.
        
        Args:
            image: (H, W, 3) RGB image [0, 255]
            
        Returns:
            (D,) normalized embedding tensor
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Preprocess and encode
        with torch.no_grad():
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features.squeeze(0)
    
    def _encode_text(self, text: str) -> Tensor:
        """
        Encode text to CLIP embedding with caching.
        
        Args:
            text: Text string
            
        Returns:
            (D,) normalized embedding tensor
        """
        if text in self.text_cache:
            return self.text_cache[text]
        
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        text_embedding = text_features.squeeze(0)
        self.text_cache[text] = text_embedding
        
        return text_embedding
    
    def _match_text_queries(
        self,
        visual_embedding: Tensor,
        text_queries: List[str]
    ) -> Tuple[str, float]:
        """
        Match visual embedding against text queries.
        
        Args:
            visual_embedding: (D,) visual feature vector
            text_queries: List of text queries
            
        Returns:
            (best_query, similarity_score)
        """
        # Encode all text queries
        text_embeddings = []
        for query in text_queries:
            text_emb = self._encode_text(query)
            text_embeddings.append(text_emb)
        
        text_embeddings = torch.stack(text_embeddings)
        
        # Compute similarities
        similarities = (visual_embedding @ text_embeddings.T)
        
        # Get best match
        best_idx = similarities.argmax().item()
        best_similarity = similarities[best_idx].item()
        
        # Normalize to [0, 1]
        best_similarity = (best_similarity + 1) / 2
        
        return text_queries[best_idx], best_similarity
    
    def _classify_object(self, visual_embedding: Tensor) -> Tuple[str, float]:
        """
        Classify object into predefined categories.
        
        Args:
            visual_embedding: (D,) visual feature vector
            
        Returns:
            (category, confidence)
        """
        # Compute similarities with category embeddings
        similarities = (visual_embedding @ self.category_embeddings.T)
        
        # Get best match
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        
        # Normalize to [0, 1]
        best_score = (best_score + 1) / 2
        
        return self.common_categories[best_idx], best_score
    
    def _recognize_attributes(self, visual_embedding: Tensor) -> Dict[str, Dict[str, float]]:
        """
        Recognize object attributes (color, size, material, state).
        
        Args:
            visual_embedding: (D,) visual feature vector
            
        Returns:
            Dict mapping attribute_type -> {value: score}
        """
        attributes = {}
        
        for attr_type, attr_embeds in self.attribute_embeddings.items():
            # Compute similarities
            similarities = (visual_embedding @ attr_embeds.T)
            
            # Normalize to [0, 1]
            similarities = (similarities + 1) / 2
            
            # Create dict of value -> score
            attr_values = self.common_attributes[attr_type]
            attr_scores = {
                val: float(similarities[i].item())
                for i, val in enumerate(attr_values)
            }
            
            attributes[attr_type] = attr_scores
        
        return attributes
    
    def _sliding_window_query(
        self,
        image: np.ndarray,
        query_embedding: Tensor,
        window_sizes: List[int] = [128, 256, 512],
        stride: int = 64
    ) -> List[GroundedObject]:
        """
        Use sliding window to find objects matching query.
        
        Args:
            image: (H, W, 3) RGB image
            query_embedding: (D,) text embedding
            window_sizes: List of window sizes to try
            stride: Sliding window stride
            
        Returns:
            List of GroundedObject instances
        """
        H, W = image.shape[:2]
        grounded_objects = []
        
        for window_size in window_sizes:
            for y in range(0, H - window_size + 1, stride):
                for x in range(0, W - window_size + 1, stride):
                    # Extract window
                    window = image[y:y+window_size, x:x+window_size]
                    
                    # Encode window
                    window_embedding = self._encode_image(window)
                    
                    # Compute similarity
                    similarity = (window_embedding @ query_embedding.T).item()
                    similarity = (similarity + 1) / 2
                    
                    # Threshold
                    if similarity > 0.7:
                        bbox = (x, y, x + window_size, y + window_size)
                        
                        grounded_obj = GroundedObject(
                            bbox=bbox,
                            mask=None,
                            text_query="",
                            similarity=float(similarity),
                            embedding=window_embedding.cpu().numpy()
                        )
                        grounded_objects.append(grounded_obj)
        
        # Non-maximum suppression
        grounded_objects = self._nms_grounded_objects(grounded_objects, iou_threshold=0.5)
        
        return grounded_objects
    
    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert binary mask to bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return (0, 0, 0, 0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return (int(x1), int(y1), int(x2), int(y2))
    
    @staticmethod
    def _extract_masked_region(
        image: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = 10
    ) -> np.ndarray:
        """
        Extract masked region with padding and white background.
        
        Args:
            image: (H, W, 3) RGB image
            mask: (H, W) binary mask
            bbox: (x1, y1, x2, y2) bounding box
            padding: Padding around bbox
            
        Returns:
            (H', W', 3) extracted region
        """
        x1, y1, x2, y2 = bbox
        H, W = image.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(W, x2 + padding)
        y2 = min(H, y2 + padding)
        
        # Extract region
        region = image[y1:y2, x1:x2].copy()
        region_mask = mask[y1:y2, x1:x2]
        
        # Apply white background
        region[region_mask == 0] = 255
        
        return region
    
    @staticmethod
    def _nms_grounded_objects(
        objects: List[GroundedObject],
        iou_threshold: float = 0.5
    ) -> List[GroundedObject]:
        """Non-maximum suppression for grounded objects."""
        if len(objects) == 0:
            return []
        
        # Sort by similarity
        objects = sorted(objects, key=lambda x: x.similarity, reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, obj in enumerate(objects):
            if i in suppressed:
                continue
            
            keep.append(obj)
            
            # Suppress overlapping objects
            for j, other_obj in enumerate(objects[i+1:], start=i+1):
                if j in suppressed:
                    continue
                
                iou = CLIPGroundingEngine._compute_bbox_iou(obj.bbox, other_obj.bbox)
                
                if iou > iou_threshold:
                    suppressed.add(j)
        
        return keep
    
    @staticmethod
    def _compute_bbox_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    def _fallback_grounding(
        self,
        image: np.ndarray,
        masks: List[np.ndarray]
    ) -> List[GroundedObject]:
        """Fallback grounding when CLIP is not available."""
        grounded_objects = []
        
        for mask in masks:
            bbox = self._mask_to_bbox(mask)
            
            grounded_obj = GroundedObject(
                bbox=bbox,
                mask=mask,
                text_query="unknown",
                similarity=0.0,
                embedding=np.zeros(512),
                category="unknown"
            )
            grounded_objects.append(grounded_obj)
        
        return grounded_objects


def create_grounding_engine(config: dict) -> CLIPGroundingEngine:
    """
    Factory function to create CLIP grounding engine from config.
    
    Args:
        config: Configuration dict with keys:
            - model_name: str ('RN50', 'ViT-B/32', 'ViT-L/14')
            - device: str
            
    Returns:
        Configured CLIPGroundingEngine
    """
    return CLIPGroundingEngine(
        model_name=config.get('model_name', 'ViT-L/14'),
        device=config.get('device', 'cuda')
    )
