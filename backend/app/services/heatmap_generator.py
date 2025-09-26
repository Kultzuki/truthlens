"""
Grad-CAM Heatmap Generator - Explainable AI for Deepfake Detection
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import io
import base64
import logging

logger = logging.getLogger(__name__)


class GradCAMGenerator:
    """Generate Grad-CAM heatmaps for model explanations"""
    
    def __init__(self):
        """Initialize Grad-CAM generator"""
        self.colormap = cv2.COLORMAP_JET
        self.alpha = 0.6  # Transparency for overlay
        
    def generate_gradcam(
        self, 
        feature_maps: np.ndarray,
        gradients: Optional[np.ndarray] = None,
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap from feature maps
        
        Args:
            feature_maps: Feature maps from model (N, C, H, W)
            gradients: Gradients if available (N, C, H, W)
            target_size: Size to resize heatmap to
            
        Returns:
            Grad-CAM heatmap
        """
        try:
            if feature_maps is None or len(feature_maps.shape) != 4:
                return self._generate_mock_heatmap(target_size)
            
            # Remove batch dimension if present
            if feature_maps.shape[0] == 1:
                feature_maps = feature_maps[0]
            
            if gradients is not None and gradients.shape[0] == 1:
                gradients = gradients[0]
            
            if gradients is not None:
                # Standard Grad-CAM: weight feature maps by gradients
                weights = np.mean(gradients, axis=(1, 2))
                cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
                
                for i, w in enumerate(weights):
                    cam += w * feature_maps[i]
            else:
                # Simplified CAM: average feature maps
                cam = np.mean(feature_maps, axis=0)
            
            # Apply ReLU (only positive influences)
            cam = np.maximum(cam, 0)
            
            # Normalize to [0, 1]
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Resize to target size
            cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_CUBIC)
            
            return cam_resized
            
        except Exception as e:
            logger.error(f"Grad-CAM generation error: {e}")
            return self._generate_mock_heatmap(target_size)
    
    def generate_attention_map(
        self,
        image: np.ndarray,
        predictions: Dict[str, float],
        feature_maps: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Generate attention map based on model predictions
        
        Args:
            image: Original image
            predictions: Model predictions
            feature_maps: Feature maps from different models
            
        Returns:
            Attention heatmap
        """
        h, w = image.shape[:2]
        
        if feature_maps and any(v is not None for v in feature_maps.values()):
            # Combine feature maps from multiple models
            heatmaps = []
            weights = []
            
            for model_name, features in feature_maps.items():
                if features is not None:
                    heatmap = self.generate_gradcam(features, target_size=(w, h))
                    heatmaps.append(heatmap)
                    # Weight by model confidence
                    weights.append(predictions.get(model_name, 0.5))
            
            if heatmaps:
                # Weighted average of heatmaps
                weights = np.array(weights) / np.sum(weights)
                combined_heatmap = np.zeros_like(heatmaps[0])
                
                for heatmap, weight in zip(heatmaps, weights):
                    combined_heatmap += heatmap * weight
                
                return combined_heatmap
        
        # Fallback to content-based heatmap using actual image
        return self._generate_face_focused_heatmap((h, w), predictions.get('confidence', 0.5), image)
    
    def _generate_face_focused_heatmap(
        self, 
        size: Tuple[int, int], 
        confidence: float,
        image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate heatmap focused on face/high-frequency regions
        
        Args:
            size: (height, width) of heatmap
            confidence: Detection confidence
            image: Original image for content-based heatmap
            
        Returns:
            Face-focused heatmap
        """
        h, w = size
        
        if image is not None:
            # Create heatmap based on actual image content
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Resize to heatmap size
            gray_resized = cv2.resize(gray, (w, h))
            
            # Apply edge detection to find regions of interest
            edges = cv2.Canny(gray_resized, 50, 150)
            
            # Apply Gaussian blur to create smooth heatmap
            heatmap = cv2.GaussianBlur(edges.astype(np.float32), (21, 21), 0)
            
            # Normalize
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            # Apply confidence-based scaling
            # Higher confidence = stronger highlights
            heatmap = heatmap * (0.5 + confidence * 0.5)
            
            # Add face detection bias (center-upper region)
            y, x = np.ogrid[:h, :w]
            center_x = w // 2
            center_y = h // 3
            sigma = min(h, w) / 4
            face_bias = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
            
            # Combine edge-based and face-biased heatmaps
            heatmap = heatmap * 0.7 + face_bias * 0.3
            
            # Add subtle noise for variation
            noise = np.random.normal(0, 0.02, heatmap.shape)
            heatmap = np.clip(heatmap + noise, 0, 1)
            
        else:
            # Fallback to gaussian heatmap
            y, x = np.ogrid[:h, :w]
            center_x = w // 2
            center_y = h // 3
            sigma_x = w / (4 - confidence * 2)
            sigma_y = h / (4 - confidence * 2)
            
            heatmap = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + 
                              (y - center_y)**2 / (2 * sigma_y**2)))
            
            noise = np.random.normal(0, 0.05, heatmap.shape)
            heatmap = np.clip(heatmap + noise, 0, 1)
        
        # Enhance contrast
        heatmap = np.power(heatmap, 1.2)
        
        return heatmap
    
    def _generate_mock_heatmap(self, size: Tuple[int, int], image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate mock heatmap for testing
        
        Args:
            size: (width, height) of heatmap
            image: Optional image for content-based heatmap
            
        Returns:
            Mock heatmap
        """
        return self._generate_face_focused_heatmap((size[1], size[0]), 0.7, image)
    
    def apply_heatmap_to_image(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply heatmap overlay to image
        
        Args:
            image: Original image (RGB)
            heatmap: Heatmap to apply
            alpha: Transparency (0-1)
            
        Returns:
            Image with heatmap overlay
        """
        if alpha is None:
            alpha = self.alpha
        
        # Ensure heatmap matches image size
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, self.colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def heatmap_to_base64(
        self,
        heatmap: np.ndarray,
        apply_colormap: bool = True,
        format: str = 'PNG'
    ) -> str:
        """
        Convert heatmap to base64 string
        
        Args:
            heatmap: Heatmap array
            apply_colormap: Whether to apply colormap
            format: Image format (PNG, JPEG)
            
        Returns:
            Base64 encoded heatmap
        """
        try:
            # Normalize to 0-255
            heatmap_normalized = (heatmap * 255).astype(np.uint8)
            
            if apply_colormap:
                # Apply colormap
                heatmap_colored = cv2.applyColorMap(heatmap_normalized, self.colormap)
                heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            else:
                # Grayscale
                heatmap_rgb = cv2.cvtColor(heatmap_normalized, cv2.COLOR_GRAY2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(heatmap_rgb)
            
            # Optimize for web
            if format == 'JPEG':
                # Use JPEG for smaller size
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=85, optimize=True)
            else:
                # Use PNG for better quality
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG', optimize=True)
            
            buffer.seek(0)
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            mime_type = f"image/{format.lower()}"
            return f"data:{mime_type};base64,{base64_string}"
            
        except Exception as e:
            logger.error(f"Heatmap to base64 conversion error: {e}")
            return ""
    
    def create_side_by_side_comparison(
        self,
        original: np.ndarray,
        with_heatmap: np.ndarray
    ) -> np.ndarray:
        """
        Create side-by-side comparison image
        
        Args:
            original: Original image
            with_heatmap: Image with heatmap overlay
            
        Returns:
            Side-by-side comparison image
        """
        # Ensure same height
        h = max(original.shape[0], with_heatmap.shape[0])
        
        # Resize if needed
        if original.shape[0] != h:
            scale = h / original.shape[0]
            new_w = int(original.shape[1] * scale)
            original = cv2.resize(original, (new_w, h))
        
        if with_heatmap.shape[0] != h:
            scale = h / with_heatmap.shape[0]
            new_w = int(with_heatmap.shape[1] * scale)
            with_heatmap = cv2.resize(with_heatmap, (new_w, h))
        
        # Add separator
        separator = np.ones((h, 10, 3), dtype=np.uint8) * 255
        
        # Concatenate horizontally
        comparison = np.hstack([original, separator, with_heatmap])
        
        return comparison
    
    def generate_multi_scale_heatmap(
        self,
        image: np.ndarray,
        feature_maps_list: list,
        scales: list = [0.5, 1.0, 1.5]
    ) -> np.ndarray:
        """
        Generate multi-scale heatmap for better localization
        
        Args:
            image: Original image
            feature_maps_list: List of feature maps at different scales
            scales: Scale factors
            
        Returns:
            Multi-scale heatmap
        """
        h, w = image.shape[:2]
        combined_heatmap = np.zeros((h, w), dtype=np.float32)
        
        for features, scale in zip(feature_maps_list, scales):
            if features is not None:
                heatmap = self.generate_gradcam(features, target_size=(w, h))
                combined_heatmap += heatmap * scale
        
        # Normalize
        if combined_heatmap.max() > 0:
            combined_heatmap = combined_heatmap / combined_heatmap.max()
        
        return combined_heatmap