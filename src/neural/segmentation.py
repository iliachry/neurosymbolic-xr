"""
Zero-Shot Segmentation Engine using SAM 2 (Segment Anything Model 2)

Provides automatic and prompted segmentation for scene understanding.
Optimized for real-time XR applications with temporal consistency.

References:
- Ravi et al. "SAM 2: Segment Anything in Images and Videos" (2024)
- https://github.com/facebookresearch/segment-anything-2
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
except ImportError:
    print("Warning: SAM 2 not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")


@dataclass
class SegmentationMask:
    """Single segmentation mask with metadata."""
    
    mask: np.ndarray  # (H, W) binary mask
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area: int  # Number of pixels
    confidence: float  # Prediction confidence [0, 1]
    stability_score: float  # Mask stability score [0, 1]
    category: Optional[str] = None  # Semantic category (if classified)
    instance_id: Optional[int] = None  # Instance ID for tracking
    
    def __post_init__(self):
        """Compute derived properties."""
        if self.area == 0:
            self.area = int(self.mask.sum())
        if len(self.bbox) == 0:
            self.bbox = self._compute_bbox()
    
    def _compute_bbox(self) -> Tuple[int, int, int, int]:
        """Compute bounding box from mask."""
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)
        if not rows.any() or not cols.any():
            return (0, 0, 0, 0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return (int(x1), int(y1), int(x2), int(y2))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'mask': self.mask.tolist(),
            'bbox': self.bbox,
            'area': self.area,
            'confidence': self.confidence,
            'stability_score': self.stability_score,
            'category': self.category,
            'instance_id': self.instance_id
        }


class SegmentationEngine:
    """
    Zero-shot segmentation engine using SAM 2.
    
    Features:
    - Automatic mask generation (everything mode)
    - Prompted segmentation (points, boxes, masks)
    - Video segmentation with temporal consistency
    - Multi-scale processing for small objects
    - Real-time performance optimization
    
    Args:
        model_size: SAM 2 model size ('tiny', 'small', 'base', 'large')
        device: Computation device
        use_video_mode: Enable video predictor for temporal consistency
    """
    
    def __init__(
        self,
        model_size: str = "large",
        device: str = "cuda",
        use_video_mode: bool = False
    ):
        self.model_size = model_size
        self.device = device
        self.use_video_mode = use_video_mode
        
        # Model configuration
        model_configs = {
            'tiny': 'sam2_hiera_t.yaml',
            'small': 'sam2_hiera_s.yaml',
            'base': 'sam2_hiera_b+.yaml',
            'large': 'sam2_hiera_l.yaml'
        }
        
        checkpoint_paths = {
            'tiny': 'checkpoints/sam2_hiera_tiny.pt',
            'small': 'checkpoints/sam2_hiera_small.pt',
            'base': 'checkpoints/sam2_hiera_base_plus.pt',
            'large': 'checkpoints/sam2_hiera_large.pt'
        }
        
        config_file = model_configs.get(model_size, model_configs['large'])
        checkpoint = checkpoint_paths.get(model_size, checkpoint_paths['large'])
        
        # Build SAM 2 model
        try:
            self.sam2_model = build_sam2(config_file, checkpoint, device=device)
            
            if use_video_mode:
                self.predictor = SAM2VideoPredictor(self.sam2_model)
            else:
                self.predictor = SAM2ImagePredictor(self.sam2_model)
        except Exception as e:
            print(f"Warning: Could not load SAM 2 model: {e}")
            self.predictor = None
        
        # Automatic mask generation parameters
        self.points_per_side = 32
        self.points_per_batch = 64
        self.pred_iou_thresh = 0.88
        self.stability_score_thresh = 0.95
        self.stability_score_offset = 1.0
        self.box_nms_thresh = 0.7
        self.crop_n_layers = 0
        self.crop_nms_thresh = 0.7
        
        # Video tracking state
        self.video_state = None
        self.tracked_objects = {}
    
    def segment_image(
        self,
        image: np.ndarray,
        mode: str = "automatic",
        prompts: Optional[Dict] = None
    ) -> List[SegmentationMask]:
        """
        Segment image using SAM 2.
        
        Args:
            image: (H, W, 3) RGB image [0, 255]
            mode: Segmentation mode ('automatic', 'prompted')
            prompts: Prompt dict with keys:
                - points: (N, 2) point coordinates
                - point_labels: (N,) labels (1=foreground, 0=background)
                - boxes: (M, 4) bounding boxes [x1, y1, x2, y2]
                - mask_input: (1, H, W) previous mask for refinement
                
        Returns:
            List of SegmentationMask objects
        """
        if self.predictor is None:
            return self._fallback_segmentation(image)
        
        # Set image
        self.predictor.set_image(image)
        
        if mode == "automatic":
            return self._automatic_mask_generation(image)
        elif mode == "prompted":
            return self._prompted_segmentation(prompts)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _automatic_mask_generation(self, image: np.ndarray) -> List[SegmentationMask]:
        """
        Generate all masks automatically using grid prompting.
        
        Args:
            image: (H, W, 3) RGB image
            
        Returns:
            List of SegmentationMask objects sorted by area
        """
        H, W = image.shape[:2]
        
        # Generate point grid
        points = self._generate_point_grid(H, W)
        
        # Batch prediction
        all_masks = []
        all_scores = []
        all_ious = []
        
        for i in range(0, len(points), self.points_per_batch):
            batch_points = points[i:i + self.points_per_batch]
            batch_labels = np.ones(len(batch_points), dtype=np.int32)
            
            # Predict masks
            masks, iou_predictions, low_res_masks = self.predictor.predict(
                point_coords=batch_points,
                point_labels=batch_labels,
                multimask_output=True
            )
            
            # Filter by IoU threshold
            for j in range(len(batch_points)):
                for k in range(masks.shape[1]):  # Multiple masks per point
                    mask = masks[j, k]
                    iou = iou_predictions[j, k]
                    
                    if iou > self.pred_iou_thresh:
                        stability = self._calculate_stability_score(mask)
                        
                        if stability > self.stability_score_thresh:
                            all_masks.append(mask)
                            all_scores.append(stability)
                            all_ious.append(iou)
        
        # Non-maximum suppression
        if len(all_masks) > 0:
            masks_nms = self._nms_masks(all_masks, all_scores, self.box_nms_thresh)
        else:
            masks_nms = []
        
        # Convert to SegmentationMask objects
        segmentation_masks = []
        for i, mask_idx in enumerate(masks_nms):
            mask = all_masks[mask_idx]
            score = all_scores[mask_idx]
            iou = all_ious[mask_idx]
            
            seg_mask = SegmentationMask(
                mask=mask.astype(np.uint8),
                bbox=(),  # Computed in __post_init__
                area=0,  # Computed in __post_init__
                confidence=float(iou),
                stability_score=float(score)
            )
            segmentation_masks.append(seg_mask)
        
        # Sort by area (largest first)
        segmentation_masks.sort(key=lambda x: x.area, reverse=True)
        
        return segmentation_masks
    
    def _prompted_segmentation(self, prompts: Dict) -> List[SegmentationMask]:
        """
        Segment using user prompts (points, boxes, or masks).
        
        Args:
            prompts: Dict with prompt information
            
        Returns:
            List of SegmentationMask objects
        """
        point_coords = prompts.get('points', None)
        point_labels = prompts.get('point_labels', None)
        boxes = prompts.get('boxes', None)
        mask_input = prompts.get('mask_input', None)
        
        # Predict masks
        masks, iou_predictions, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=boxes,
            mask_input=mask_input,
            multimask_output=True
        )
        
        # Convert to SegmentationMask objects
        segmentation_masks = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            iou = iou_predictions[i]
            stability = self._calculate_stability_score(mask)
            
            seg_mask = SegmentationMask(
                mask=mask.astype(np.uint8),
                bbox=(),
                area=0,
                confidence=float(iou),
                stability_score=float(stability)
            )
            segmentation_masks.append(seg_mask)
        
        return segmentation_masks
    
    def segment_video(
        self,
        frames: List[np.ndarray],
        initial_prompts: Optional[Dict[int, Dict]] = None
    ) -> Dict[int, List[SegmentationMask]]:
        """
        Segment video with temporal consistency.
        
        Args:
            frames: List of (H, W, 3) RGB frames
            initial_prompts: Dict mapping frame_idx -> prompts for initialization
            
        Returns:
            Dict mapping frame_idx -> List[SegmentationMask]
        """
        if not self.use_video_mode:
            raise ValueError("Video mode not enabled. Set use_video_mode=True")
        
        # Initialize video state
        self.predictor.reset_state()
        
        # Add frames to predictor
        for frame_idx, frame in enumerate(frames):
            self.predictor.add_frame(frame)
        
        # Initialize objects from prompts
        if initial_prompts:
            for frame_idx, prompts in initial_prompts.items():
                for obj_id, prompt in prompts.items():
                    self.predictor.add_new_points(
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=prompt.get('points'),
                        labels=prompt.get('point_labels')
                    )
        
        # Propagate masks through video
        video_segments = {}
        
        for frame_idx in range(len(frames)):
            frame_masks = self.predictor.propagate_in_video(frame_idx)
            
            # Convert to SegmentationMask objects
            segmentation_masks = []
            for obj_id, mask_data in frame_masks.items():
                mask = mask_data['mask']
                score = mask_data.get('score', 1.0)
                
                seg_mask = SegmentationMask(
                    mask=mask.astype(np.uint8),
                    bbox=(),
                    area=0,
                    confidence=float(score),
                    stability_score=float(score),
                    instance_id=obj_id
                )
                segmentation_masks.append(seg_mask)
            
            video_segments[frame_idx] = segmentation_masks
        
        return video_segments
    
    def refine_mask(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        refinement_points: Optional[np.ndarray] = None,
        refinement_labels: Optional[np.ndarray] = None
    ) -> SegmentationMask:
        """
        Refine existing mask with additional prompts.
        
        Args:
            image: (H, W, 3) RGB image
            initial_mask: (H, W) binary mask
            refinement_points: (N, 2) additional points
            refinement_labels: (N,) point labels
            
        Returns:
            Refined SegmentationMask
        """
        self.predictor.set_image(image)
        
        # Convert mask to low-res format
        mask_input = torch.from_numpy(initial_mask).float().unsqueeze(0).unsqueeze(0)
        mask_input = F.interpolate(mask_input, size=(256, 256), mode='bilinear')
        
        # Predict refined mask
        masks, iou_predictions, _ = self.predictor.predict(
            point_coords=refinement_points,
            point_labels=refinement_labels,
            mask_input=mask_input.numpy(),
            multimask_output=False
        )
        
        refined_mask = masks[0]
        iou = iou_predictions[0]
        stability = self._calculate_stability_score(refined_mask)
        
        return SegmentationMask(
            mask=refined_mask.astype(np.uint8),
            bbox=(),
            area=0,
            confidence=float(iou),
            stability_score=float(stability)
        )
    
    # Helper methods
    
    def _generate_point_grid(self, H: int, W: int) -> np.ndarray:
        """
        Generate regular grid of points for automatic segmentation.
        
        Args:
            H: Image height
            W: Image width
            
        Returns:
            (N, 2) array of point coordinates
        """
        n_points = self.points_per_side
        
        # Generate grid
        x = np.linspace(0, W - 1, n_points)
        y = np.linspace(0, H - 1, n_points)
        xv, yv = np.meshgrid(x, y)
        
        points = np.stack([xv.ravel(), yv.ravel()], axis=-1)
        
        return points
    
    def _calculate_stability_score(
        self,
        mask: np.ndarray,
        threshold_offset: float = 1.0
    ) -> float:
        """
        Calculate mask stability score by comparing thresholded versions.
        
        Args:
            mask: (H, W) mask with values [0, 1]
            threshold_offset: Offset for threshold comparison
            
        Returns:
            Stability score [0, 1]
        """
        # Threshold at different levels
        thresh_high = 0.5 + threshold_offset / 255.0
        thresh_low = 0.5 - threshold_offset / 255.0
        
        mask_high = (mask > thresh_high).astype(np.float32)
        mask_low = (mask > thresh_low).astype(np.float32)
        
        # Compute IoU between thresholded masks
        intersection = (mask_high * mask_low).sum()
        union = mask_high.sum() + mask_low.sum() - intersection
        
        if union == 0:
            return 0.0
        
        stability = intersection / union
        return float(stability)
    
    def _nms_masks(
        self,
        masks: List[np.ndarray],
        scores: List[float],
        iou_threshold: float
    ) -> List[int]:
        """
        Non-maximum suppression for masks based on IoU.
        
        Args:
            masks: List of binary masks
            scores: List of mask scores
            iou_threshold: IoU threshold for suppression
            
        Returns:
            List of indices to keep
        """
        if len(masks) == 0:
            return []
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        
        keep = []
        suppressed = set()
        
        for idx in sorted_indices:
            if idx in suppressed:
                continue
            
            keep.append(idx)
            
            # Suppress overlapping masks
            for other_idx in sorted_indices:
                if other_idx == idx or other_idx in suppressed:
                    continue
                
                iou = self._compute_mask_iou(masks[idx], masks[other_idx])
                
                if iou > iou_threshold:
                    suppressed.add(other_idx)
        
        return keep
    
    @staticmethod
    def _compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    def _fallback_segmentation(self, image: np.ndarray) -> List[SegmentationMask]:
        """
        Fallback segmentation using traditional computer vision.
        Used when SAM 2 is not available.
        
        Args:
            image: (H, W, 3) RGB image
            
        Returns:
            List of SegmentationMask objects
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert to SegmentationMask objects
        segmentation_masks = []
        
        for contour in contours:
            # Create mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            
            area = int(cv2.contourArea(contour))
            
            # Filter small regions
            if area < 100:
                continue
            
            seg_mask = SegmentationMask(
                mask=mask,
                bbox=(),
                area=area,
                confidence=0.5,
                stability_score=0.5
            )
            segmentation_masks.append(seg_mask)
        
        # Sort by area
        segmentation_masks.sort(key=lambda x: x.area, reverse=True)
        
        return segmentation_masks


def create_segmentation_engine(config: dict) -> SegmentationEngine:
    """
    Factory function to create segmentation engine from config.
    
    Args:
        config: Configuration dict with keys:
            - model_size: str ('tiny', 'small', 'base', 'large')
            - device: str
            - use_video_mode: bool
            
    Returns:
        Configured SegmentationEngine
    """
    return SegmentationEngine(
        model_size=config.get('model_size', 'large'),
        device=config.get('device', 'cuda'),
        use_video_mode=config.get('use_video_mode', False)
    )
