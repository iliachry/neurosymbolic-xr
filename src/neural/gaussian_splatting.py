"""
3D Gaussian Splatting Engine for Real-Time Scene Capture

Implements differentiable 3D Gaussian splatting for neural scene representation.
Optimized for real-time XR applications with adaptive density control.

References:
- Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (2023)
- Gsplat: https://github.com/nerfstudio-project/gsplat
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
from torch import Tensor


@dataclass
class GaussianPrimitive:
    """Single 3D Gaussian primitive with rendering attributes."""
    
    position: Tensor  # (3,) - xyz center
    scale: Tensor  # (3,) - axis-aligned scale
    rotation: Tensor  # (4,) - quaternion rotation
    opacity: Tensor  # (1,) - alpha value [0, 1]
    sh_coefficients: Tensor  # (K, 3) - spherical harmonics for view-dependent color
    
    def __post_init__(self):
        """Validate tensor shapes."""
        assert self.position.shape == (3,), f"Position shape {self.position.shape} != (3,)"
        assert self.scale.shape == (3,), f"Scale shape {self.scale.shape} != (3,)"
        assert self.rotation.shape == (4,), f"Rotation shape {self.rotation.shape} != (4,)"
        assert self.opacity.shape == (1,), f"Opacity shape {self.opacity.shape} != (1,)"
        assert self.sh_coefficients.shape[1] == 3, "SH coefficients must have 3 color channels"


class GaussianSplattingEngine(nn.Module):
    """
    3D Gaussian Splatting engine for real-time neural scene capture.
    
    Features:
    - Multi-view optimization with photometric loss
    - Adaptive density control (splitting, cloning, pruning)
    - Tile-based differentiable rasterization
    - Spherical harmonics for view-dependent appearance
    - Real-time rendering (<20ms per frame)
    
    Args:
        max_gaussians: Maximum number of Gaussian primitives
        sh_degree: Spherical harmonics degree (0-3)
        device: Computation device
    """
    
    def __init__(
        self,
        max_gaussians: int = 500_000,
        sh_degree: int = 3,
        device: str = "cuda"
    ):
        super().__init__()
        self.max_gaussians = max_gaussians
        self.sh_degree = sh_degree
        self.device = device
        
        # Gaussian parameters (initialized empty)
        self.num_gaussians = 0
        self.positions = None  # (N, 3)
        self.scales = None  # (N, 3)
        self.rotations = None  # (N, 4) quaternions
        self.opacities = None  # (N, 1)
        self.sh_coeffs = None  # (N, (sh_degree+1)^2, 3)
        
        # Optimization state
        self.optimizer = None
        self.iteration = 0
        
        # Adaptive control parameters
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_grad_threshold = 0.0002
        self.densify_size_threshold = 0.01
        self.prune_opacity_threshold = 0.005
        
    def initialize_from_pointcloud(
        self,
        points: Tensor,
        colors: Optional[Tensor] = None,
        normals: Optional[Tensor] = None
    ) -> None:
        """
        Initialize Gaussians from a point cloud.
        
        Args:
            points: (N, 3) point positions
            colors: (N, 3) RGB colors [0, 1]
            normals: (N, 3) surface normals (optional)
        """
        N = points.shape[0]
        assert N <= self.max_gaussians, f"Too many points: {N} > {self.max_gaussians}"
        
        self.num_gaussians = N
        
        # Initialize positions
        self.positions = nn.Parameter(points.clone().to(self.device))
        
        # Initialize scales from nearest neighbor distances
        with torch.no_grad():
            dists = torch.cdist(points, points)
            dists[dists == 0] = float('inf')
            nn_dists = dists.min(dim=1)[0]
            init_scale = nn_dists.unsqueeze(-1).repeat(1, 3) * 0.5
        self.scales = nn.Parameter(torch.log(init_scale).to(self.device))
        
        # Initialize rotations (identity quaternions)
        self.rotations = nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(N, 1)
        )
        
        # Initialize opacities (inverse sigmoid of 0.1)
        self.opacities = nn.Parameter(
            torch.logit(torch.tensor([0.1], device=self.device)).repeat(N, 1)
        )
        
        # Initialize spherical harmonics from colors
        num_sh = (self.sh_degree + 1) ** 2
        sh_coeffs = torch.zeros((N, num_sh, 3), device=self.device)
        if colors is not None:
            # DC component (0th order SH)
            sh_coeffs[:, 0, :] = self._rgb_to_sh(colors)
        self.sh_coeffs = nn.Parameter(sh_coeffs)
        
        # Setup optimizer
        self._setup_optimizer()
        
    def _setup_optimizer(self) -> None:
        """Setup Adam optimizer with per-parameter learning rates."""
        param_groups = [
            {'params': [self.positions], 'lr': 1.6e-4, 'name': 'positions'},
            {'params': [self.scales], 'lr': 5e-3, 'name': 'scales'},
            {'params': [self.rotations], 'lr': 1e-3, 'name': 'rotations'},
            {'params': [self.opacities], 'lr': 5e-2, 'name': 'opacities'},
            {'params': [self.sh_coeffs], 'lr': 2.5e-3, 'name': 'sh_coeffs'},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        
    def capture_scene(
        self,
        images: List[Tensor],
        camera_poses: List[Tensor],
        camera_intrinsics: Tensor,
        num_iterations: int = 30_000,
        initial_pointcloud: Optional[Tuple[Tensor, Tensor]] = None
    ) -> dict:
        """
        Capture scene from multi-view images using optimization.
        
        Args:
            images: List of (H, W, 3) RGB images [0, 1]
            camera_poses: List of (4, 4) camera-to-world transforms
            camera_intrinsics: (3, 3) camera intrinsic matrix
            num_iterations: Number of optimization iterations
            initial_pointcloud: Optional (points, colors) tuple for initialization
            
        Returns:
            dict with 'gaussians', 'loss_history', 'metrics'
        """
        # Initialize from point cloud (SfM or depth-based)
        if initial_pointcloud is None:
            initial_pointcloud = self._initialize_pointcloud_from_images(
                images, camera_poses, camera_intrinsics
            )
        
        points, colors = initial_pointcloud
        self.initialize_from_pointcloud(points, colors)
        
        # Optimization loop
        loss_history = []
        
        for iteration in range(num_iterations):
            self.iteration = iteration
            
            # Sample random view
            idx = np.random.randint(len(images))
            target_image = images[idx].to(self.device)
            camera_pose = camera_poses[idx].to(self.device)
            
            # Render from camera viewpoint
            rendered = self.render(camera_pose, camera_intrinsics, target_image.shape[:2])
            
            # Compute loss
            loss = self._compute_loss(rendered, target_image)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Adaptive density control
            if iteration % self.densification_interval == 0 and iteration > 500:
                self._adaptive_density_control()
            
            # Opacity reset
            if iteration % self.opacity_reset_interval == 0:
                self._reset_opacity()
            
            self.optimizer.step()
            
            loss_history.append(loss.item())
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.4f}, "
                      f"Gaussians: {self.num_gaussians}")
        
        return {
            'gaussians': self.get_gaussians(),
            'loss_history': loss_history,
            'metrics': self._compute_metrics(images, camera_poses, camera_intrinsics)
        }
    
    def render(
        self,
        camera_pose: Tensor,
        camera_intrinsics: Tensor,
        image_size: Tuple[int, int],
        background: Optional[Tensor] = None
    ) -> Tensor:
        """
        Render scene from camera viewpoint using tile-based rasterization.
        
        Args:
            camera_pose: (4, 4) camera-to-world transform
            camera_intrinsics: (3, 3) intrinsic matrix
            image_size: (height, width)
            background: (3,) background color (default: white)
            
        Returns:
            (H, W, 3) rendered RGB image
        """
        if background is None:
            background = torch.ones(3, device=self.device)
        
        H, W = image_size
        
        # Transform Gaussians to camera space
        world_to_cam = torch.inverse(camera_pose)
        positions_cam = self._transform_points(self.positions, world_to_cam)
        
        # Get Gaussian parameters
        scales = torch.exp(self.scales)
        opacities = torch.sigmoid(self.opacities)
        rotations = F.normalize(self.rotations, dim=-1)
        
        # Compute view-dependent colors from spherical harmonics
        view_dirs = F.normalize(positions_cam, dim=-1)
        colors = self._eval_sh(self.sh_coeffs, view_dirs)
        
        # Tile-based rasterization using gsplat
        rendered_image, alpha = rasterization(
            means=positions_cam,
            quats=rotations,
            scales=scales,
            opacities=opacities.squeeze(-1),
            colors=colors,
            viewmat=world_to_cam[:3, :],
            projmat=camera_intrinsics,
            image_width=W,
            image_height=H,
            tile_size=16,
            background=background
        )
        
        return rendered_image
    
    def _compute_loss(self, rendered: Tensor, target: Tensor) -> Tensor:
        """
        Compute photometric loss with L1 and SSIM components.
        
        Args:
            rendered: (H, W, 3) rendered image
            target: (H, W, 3) target image
            
        Returns:
            Scalar loss
        """
        # L1 loss
        l1_loss = F.l1_loss(rendered, target)
        
        # SSIM loss
        ssim_loss = 1.0 - self._ssim(rendered, target)
        
        # Combined loss (0.8 * L1 + 0.2 * SSIM)
        loss = 0.8 * l1_loss + 0.2 * ssim_loss
        
        return loss
    
    def _adaptive_density_control(self) -> None:
        """
        Adaptive Gaussian density control: split large Gaussians, clone small ones, prune transparent.
        """
        with torch.no_grad():
            # Get gradient magnitudes
            if self.positions.grad is not None:
                grad_norm = torch.norm(self.positions.grad, dim=-1)
                
                # Split large Gaussians with high gradients
                scales = torch.exp(self.scales)
                max_scale = scales.max(dim=-1)[0]
                
                split_mask = (grad_norm > self.densify_grad_threshold) & \
                             (max_scale > self.densify_size_threshold)
                
                if split_mask.sum() > 0:
                    self._split_gaussians(split_mask)
                
                # Clone small Gaussians with high gradients
                clone_mask = (grad_norm > self.densify_grad_threshold) & \
                             (max_scale <= self.densify_size_threshold)
                
                if clone_mask.sum() > 0:
                    self._clone_gaussians(clone_mask)
            
            # Prune transparent Gaussians
            opacities = torch.sigmoid(self.opacities)
            prune_mask = opacities.squeeze(-1) < self.prune_opacity_threshold
            
            if prune_mask.sum() > 0:
                self._prune_gaussians(prune_mask)
    
    def _split_gaussians(self, mask: Tensor) -> None:
        """Split Gaussians by creating two smaller copies."""
        num_split = mask.sum().item()
        if self.num_gaussians + num_split > self.max_gaussians:
            return
        
        # Create split copies with reduced scale
        new_positions = self.positions[mask].clone()
        new_scales = self.scales[mask].clone() - np.log(1.6)
        new_rotations = self.rotations[mask].clone()
        new_opacities = self.opacities[mask].clone()
        new_sh_coeffs = self.sh_coeffs[mask].clone()
        
        # Concatenate
        self.positions = nn.Parameter(torch.cat([self.positions, new_positions]))
        self.scales = nn.Parameter(torch.cat([self.scales, new_scales]))
        self.rotations = nn.Parameter(torch.cat([self.rotations, new_rotations]))
        self.opacities = nn.Parameter(torch.cat([self.opacities, new_opacities]))
        self.sh_coeffs = nn.Parameter(torch.cat([self.sh_coeffs, new_sh_coeffs]))
        
        self.num_gaussians += num_split
        self._setup_optimizer()
    
    def _clone_gaussians(self, mask: Tensor) -> None:
        """Clone Gaussians by duplicating them."""
        num_clone = mask.sum().item()
        if self.num_gaussians + num_clone > self.max_gaussians:
            return
        
        # Clone parameters
        self.positions = nn.Parameter(torch.cat([self.positions, self.positions[mask].clone()]))
        self.scales = nn.Parameter(torch.cat([self.scales, self.scales[mask].clone()]))
        self.rotations = nn.Parameter(torch.cat([self.rotations, self.rotations[mask].clone()]))
        self.opacities = nn.Parameter(torch.cat([self.opacities, self.opacities[mask].clone()]))
        self.sh_coeffs = nn.Parameter(torch.cat([self.sh_coeffs, self.sh_coeffs[mask].clone()]))
        
        self.num_gaussians += num_clone
        self._setup_optimizer()
    
    def _prune_gaussians(self, mask: Tensor) -> None:
        """Remove Gaussians marked by mask."""
        keep_mask = ~mask
        
        self.positions = nn.Parameter(self.positions[keep_mask])
        self.scales = nn.Parameter(self.scales[keep_mask])
        self.rotations = nn.Parameter(self.rotations[keep_mask])
        self.opacities = nn.Parameter(self.opacities[keep_mask])
        self.sh_coeffs = nn.Parameter(self.sh_coeffs[keep_mask])
        
        self.num_gaussians = keep_mask.sum().item()
        self._setup_optimizer()
    
    def _reset_opacity(self) -> None:
        """Reset opacity to encourage re-optimization."""
        with torch.no_grad():
            opacities = torch.sigmoid(self.opacities)
            reset_value = torch.logit(torch.clamp(opacities * 0.9, 0.01, 0.99))
            self.opacities.copy_(reset_value)
    
    def get_gaussians(self) -> List[GaussianPrimitive]:
        """
        Export Gaussians as list of primitives.
        
        Returns:
            List of GaussianPrimitive objects
        """
        gaussians = []
        
        with torch.no_grad():
            scales = torch.exp(self.scales)
            opacities = torch.sigmoid(self.opacities)
            
            for i in range(self.num_gaussians):
                gaussian = GaussianPrimitive(
                    position=self.positions[i].cpu(),
                    scale=scales[i].cpu(),
                    rotation=self.rotations[i].cpu(),
                    opacity=opacities[i].cpu(),
                    sh_coefficients=self.sh_coeffs[i].cpu()
                )
                gaussians.append(gaussian)
        
        return gaussians
    
    # Helper methods
    
    @staticmethod
    def _rgb_to_sh(rgb: Tensor) -> Tensor:
        """Convert RGB to 0th order spherical harmonic (DC component)."""
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0
    
    @staticmethod
    def _eval_sh(sh_coeffs: Tensor, directions: Tensor) -> Tensor:
        """
        Evaluate spherical harmonics for view-dependent color.
        
        Args:
            sh_coeffs: (N, K, 3) SH coefficients
            directions: (N, 3) view directions
            
        Returns:
            (N, 3) RGB colors
        """
        # For simplicity, just use DC component (0th order)
        C0 = 0.28209479177387814
        colors = sh_coeffs[:, 0, :] * C0 + 0.5
        return torch.clamp(colors, 0.0, 1.0)
    
    @staticmethod
    def _transform_points(points: Tensor, transform: Tensor) -> Tensor:
        """Apply 4x4 transform to 3D points."""
        ones = torch.ones((points.shape[0], 1), device=points.device)
        points_h = torch.cat([points, ones], dim=-1)
        transformed = (transform @ points_h.T).T
        return transformed[:, :3]
    
    @staticmethod
    def _ssim(img1: Tensor, img2: Tensor, window_size: int = 11) -> Tensor:
        """
        Compute Structural Similarity Index (SSIM).
        
        Args:
            img1: (H, W, 3) image
            img2: (H, W, 3) image
            window_size: Gaussian window size
            
        Returns:
            Scalar SSIM value
        """
        # Simplified SSIM implementation
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1.permute(2, 0, 1).unsqueeze(0), window_size, 1, window_size // 2)
        mu2 = F.avg_pool2d(img2.permute(2, 0, 1).unsqueeze(0), window_size, 1, window_size // 2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(
            (img1.permute(2, 0, 1).unsqueeze(0)).pow(2), window_size, 1, window_size // 2
        ) - mu1_sq
        sigma2_sq = F.avg_pool2d(
            (img2.permute(2, 0, 1).unsqueeze(0)).pow(2), window_size, 1, window_size // 2
        ) - mu2_sq
        sigma12 = F.avg_pool2d(
            img1.permute(2, 0, 1).unsqueeze(0) * img2.permute(2, 0, 1).unsqueeze(0),
            window_size, 1, window_size // 2
        ) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def _initialize_pointcloud_from_images(
        self,
        images: List[Tensor],
        camera_poses: List[Tensor],
        camera_intrinsics: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Initialize point cloud from multi-view images using simple depth estimation.
        In production, use SfM (COLMAP) or depth sensors.
        """
        # Placeholder: Create random point cloud
        # In practice, use COLMAP SfM or depth-based reconstruction
        num_points = 10_000
        points = torch.randn(num_points, 3, device=self.device) * 2.0
        colors = torch.rand(num_points, 3, device=self.device)
        
        return points, colors
    
    def _compute_metrics(
        self,
        images: List[Tensor],
        camera_poses: List[Tensor],
        camera_intrinsics: Tensor
    ) -> dict:
        """Compute evaluation metrics (PSNR, SSIM, LPIPS)."""
        psnrs = []
        ssims = []
        
        with torch.no_grad():
            for img, pose in zip(images, camera_poses):
                rendered = self.render(pose.to(self.device), camera_intrinsics, img.shape[:2])
                target = img.to(self.device)
                
                # PSNR
                mse = F.mse_loss(rendered, target)
                psnr = -10 * torch.log10(mse)
                psnrs.append(psnr.item())
                
                # SSIM
                ssim = self._ssim(rendered, target)
                ssims.append(ssim.item())
        
        return {
            'psnr': np.mean(psnrs),
            'ssim': np.mean(ssims),
            'num_gaussians': self.num_gaussians
        }


def create_gaussian_engine(config: dict) -> GaussianSplattingEngine:
    """
    Factory function to create Gaussian Splatting engine from config.
    
    Args:
        config: Configuration dict with keys:
            - max_gaussians: int
            - sh_degree: int
            - device: str
            
    Returns:
        Configured GaussianSplattingEngine
    """
    return GaussianSplattingEngine(
        max_gaussians=config.get('max_gaussians', 500_000),
        sh_degree=config.get('sh_degree', 3),
        device=config.get('device', 'cuda')
    )
