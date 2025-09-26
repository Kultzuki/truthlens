"""
Deepfake Detection Service - Core ML Pipeline
"""

import os
import time
from typing import Dict, Any

import cv2
import numpy as np

from app.models.analysis import (
    AnalysisSignal, 
    FrameAnalysis, 
    AnalysisMetadata,
    AnalysisRegion
)
from app.services.ensemble_detector import EnsembleDetector
from app.services.local_cache import LocalCache


class DeepfakeDetectorService:
    """Service for deepfake detection using ensemble of models"""
    
    def __init__(self):
        # Load configuration from environment
        config = {
            "mesonet_path": os.getenv("MESONET_MODEL_PATH", "./models/mesonet4.onnx"),
            "xception_path": os.getenv("XCEPTION_MODEL_PATH", "./models/xception.onnx"),
            "mesonet_weight": float(os.getenv("MESONET_WEIGHT", "0.4")),
            "xception_weight": float(os.getenv("XCEPTION_WEIGHT", "0.6")),
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
        }
        
        self.max_frames_to_process = int(os.getenv("MAX_FRAMES_TO_PROCESS", "30"))
        
        # Initialize ensemble detector
        self.ensemble = EnsembleDetector(config)
        self.model_loaded = False
        
        # Initialize local cache
        self.cache = LocalCache()
    
    async def analyze_media(self, file_path: str, input_type: str) -> Dict[str, Any]:
        """
        Main analysis method for media files
        
        Args:
            file_path: Path to media file
            input_type: Type of input ('image', 'video', 'url')
        
        Returns:
            Analysis results dictionary
        """
        start_time = time.time()
        
        # Check cache first
        cached_result = self.cache.get(file_path)
        if cached_result:
            print(f"Using cached result for {file_path}")
            return cached_result
        
        # Warmup models on first use
        if not self.model_loaded:
            await self.ensemble.warmup_models()
            self.model_loaded = True
        
        try:
            if input_type == "image" or file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                # Single image analysis
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"Cannot load image: {file_path}")
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Use ensemble for analysis
                result = await self.ensemble.analyze_image(image_rgb, return_individual_scores=True)
                
                # Create response structure
                analysis_result = {
                    "verdict": result["verdict"],
                    "confidence": result["confidence"],
                    "signals": result.get("signals", []),
                    "frame_analysis": None,  # Single image doesn't have frame analysis
                    "metadata": AnalysisMetadata(
                        processed_frames=1,
                        total_frames=1,
                        resolution=f"{image_rgb.shape[1]}x{image_rgb.shape[0]}"
                    ),
                    "heatmap": result.get("heatmap"),
                    "individual_scores": result.get("individual_scores", {})
                }
                
                # Cache the result
                self.cache.set(file_path, analysis_result)
                
                return analysis_result
            
            else:
                # Video analysis using ensemble
                video_result = await self.ensemble.analyze_video(
                    file_path, 
                    max_frames=self.max_frames_to_process,
                    skip_similar=True
                )
                
                # Use video analysis results from ensemble
                analysis_result = {
                    "verdict": video_result["verdict"],
                    "confidence": video_result["confidence"],
                    "signals": video_result["signals"],
                    "frame_analysis": video_result["frame_analysis"],
                    "metadata": video_result["metadata"],
                    "heatmap": video_result["heatmap"]
                }
                
                return analysis_result
        
        except Exception as e:
            print(f"Analysis error: {e}")
            # Return error result
            return {
                "verdict": "unknown",
                "confidence": 0.0,
                "signals": [],
                "frame_analysis": None,
                "metadata": AnalysisMetadata(
                    frames_analyzed=0,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    model_version="error",
                    resolution="unknown"
                ),
                "heatmap": None
            }