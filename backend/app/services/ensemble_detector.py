"""
Ensemble Deepfake Detector - Combines MesoNet and Xception
"""

import numpy as np
import cv2
import asyncio
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

from app.services.models.mesonet import MesoNetModel
from app.services.models.xception import XceptionModel
from app.services.heatmap_generator import GradCAMGenerator
from app.models.analysis import (
    AnalysisSignal,
    FrameAnalysis,
    AnalysisMetadata,
    AnalysisRegion
)

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """Ensemble detector combining multiple models for robust deepfake detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Model weights - ADD THESE MISSING ATTRIBUTES
        self.mesonet_weight = config.get("mesonet_weight", 0.4)
        self.xception_weight = config.get("xception_weight", 0.6)
        
        # Normalize weights to sum to 1.0
        total_weight = self.mesonet_weight + self.xception_weight
        self.mesonet_weight /= total_weight
        self.xception_weight /= total_weight
        
        # Store as model_weights dict for compatibility
        self.model_weights = {
            "mesonet": self.mesonet_weight,
            "xception": self.xception_weight
        }
        
        # Confidence thresholds
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.high_confidence_threshold = 0.85
        self.low_confidence_threshold = 0.3
        
        # Initialize models
        self.mesonet = MesoNetModel(self.config.get("mesonet_path"))
        self.xception = XceptionModel(self.config.get("xception_path"))
        
        # Initialize heatmap generator
        self.heatmap_generator = GradCAMGenerator()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Model warmup flags
        self.models_warmed_up = False
        
    async def warmup_models(self):
        """Warmup models for faster inference"""
        if self.models_warmed_up:
            return
        
        logger.info("Warming up models...")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Run inference once to warmup
        try:
            await self.analyze_image(dummy_image)
            self.models_warmed_up = True
            logger.info("Models warmed up successfully")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
    
    def _run_model_inference(
        self, 
        model_name: str, 
        image: np.ndarray
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Run inference for a specific model
        
        Args:
            model_name: Name of the model
            image: Input image
            
        Returns:
            Tuple of (confidence, feature_map)
        """
        try:
            if model_name == "mesonet":
                preprocessed = self.mesonet.preprocess(image)
                confidence, features = self.mesonet.predict(preprocessed)
            elif model_name == "xception":
                preprocessed = self.xception.preprocess(image)
                confidence, features = self.xception.predict(preprocessed)
            else:
                return 0.5, None
            
            return confidence, features
            
        except Exception as e:
            logger.error(f"{model_name} inference error: {e}")
            return 0.5, None
    
    async def analyze_image(
        self, 
        image: np.ndarray,
        return_individual_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze single image with ensemble
        
        Args:
            image: Input image (RGB)
            return_individual_scores: Whether to return individual model scores
            
        Returns:
            Analysis results
        """
        start_time = time.time()
        
        # Run models in parallel
        loop = asyncio.get_event_loop()
        
        # Create tasks for parallel execution
        tasks = []
        for model_name in ["mesonet", "xception"]:
            task = loop.run_in_executor(
                self.executor,
                self._run_model_inference,
                model_name,
                image
            )
            tasks.append(task)
        
        # Wait for all models
        results = await asyncio.gather(*tasks)
        
        # Extract results
        model_predictions = {}
        feature_maps = {}
        
        mesonet_conf, mesonet_features = results[0]
        xception_conf, xception_features = results[1]
        
        model_predictions["mesonet"] = mesonet_conf
        model_predictions["xception"] = xception_conf
        feature_maps["mesonet"] = mesonet_features
        feature_maps["xception"] = xception_features
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(model_predictions)
        
        # Determine verdict
        verdict = self._determine_verdict(ensemble_confidence)
        
        # Generate heatmap
        heatmap = self.heatmap_generator.generate_attention_map(
            image,
            {"confidence": ensemble_confidence, **model_predictions},
            feature_maps
        )
        
        # Convert heatmap to base64
        heatmap_base64 = self.heatmap_generator.heatmap_to_base64(heatmap)
        
        # Create analysis signals
        signals = self._create_analysis_signals(model_predictions, ensemble_confidence)
        
        # Processing time
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            "verdict": verdict,
            "confidence": ensemble_confidence,
            "heatmap": heatmap_base64,
            "signals": signals,
            "processing_time_ms": int(processing_time)
        }
        
        if return_individual_scores:
            result["individual_scores"] = model_predictions
        
        return result
    
    async def analyze_video(
        self,
        video_path: str,
        max_frames: int = 30,
        skip_similar: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze video with ensemble
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            skip_similar: Skip similar consecutive frames
            
        Returns:
            Video analysis results
        """
        start_time = time.time()
        
        # Extract frames
        frames = self._extract_video_frames(video_path, max_frames, skip_similar)
        
        if not frames:
            raise ValueError("No frames could be extracted from video")
        
        # Analyze each frame
        frame_results = []
        all_confidences = []
        all_verdicts = []
        
        for i, frame in enumerate(frames):
            frame_analysis = await self.analyze_image(frame, return_individual_scores=True)
            
            all_confidences.append(frame_analysis["confidence"])
            all_verdicts.append(frame_analysis["verdict"])
            
            # Create frame analysis object
            frame_result = FrameAnalysis(
                timestamp=i * (1000 / len(frames)),  # Approximate timestamp
                confidence=frame_analysis["confidence"],
                regions=[]  # Could add face detection regions here
            )
            frame_results.append(frame_result)
        
        # Calculate overall video metrics
        overall_confidence = np.mean(all_confidences)
        confidence_std = np.std(all_confidences)
        
        # Determine overall verdict
        fake_frames = sum(1 for v in all_verdicts if v == "fake")
        fake_ratio = fake_frames / len(all_verdicts)
        
        if fake_ratio > 0.6:
            overall_verdict = "fake"
        elif fake_ratio < 0.3:
            overall_verdict = "real"
        else:
            overall_verdict = "unknown"
        
        # Generate heatmap from middle frame
        middle_frame_idx = len(frames) // 2
        middle_frame_analysis = await self.analyze_image(frames[middle_frame_idx])
        
        # Create signals
        signals = [
            AnalysisSignal(
                name="Ensemble Confidence",
                score=overall_confidence,
                description=f"Combined MesoNet and Xception analysis"
            ),
            AnalysisSignal(
                name="Temporal Consistency",
                score=1.0 - min(confidence_std * 2, 1.0),
                description=f"Consistency across {len(frames)} frames"
            ),
            AnalysisSignal(
                name="Fake Frame Ratio",
                score=fake_ratio,
                description=f"{fake_frames}/{len(frames)} frames detected as fake"
            )
        ]
        
        # Metadata
        metadata = AnalysisMetadata(
            processed_frames=len(frames),
            total_frames=self._get_video_frame_count(video_path),
            duration=self._get_video_duration(video_path),
            resolution=f"{frames[0].shape[1]}x{frames[0].shape[0]}"
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "verdict": overall_verdict,
            "confidence": overall_confidence,
            "confidence_std": confidence_std,
            "heatmap": middle_frame_analysis["heatmap"],
            "signals": signals,
            "frame_analysis": frame_results,
            "metadata": metadata,
            "processing_time_ms": int(processing_time)
        }
    
    def _calculate_ensemble_confidence(self, predictions: Dict[str, float]) -> float:
        """Calculate weighted ensemble confidence with better calibration"""
        
        mesonet_conf = predictions.get("mesonet", 0.5)
        xception_conf = predictions.get("xception", 0.5)
        
        # Apply weights
        ensemble_conf = (
            mesonet_conf * self.mesonet_weight + 
            xception_conf * self.xception_weight
        )
        
        # Improved calibration - push away from center
        if ensemble_conf > 0.5:
            # If leaning fake, push it more fake
            ensemble_conf = 0.5 + (ensemble_conf - 0.5) * 1.3
        else:
            # If leaning real, push it more real
            ensemble_conf = 0.5 - (0.5 - ensemble_conf) * 1.3
        
        # Ensure valid range
        return np.clip(ensemble_conf, 0.0, 1.0)
    
    def _calibrate_confidence(self, confidence: float) -> float:
        """
        Calibrate confidence score
        
        Args:
            confidence: Raw confidence
            
        Returns:
            Calibrated confidence
        """
        # Apply sigmoid calibration for better discrimination
        # This pushes values toward 0 or 1
        temperature = 2.5
        calibrated = 1 / (1 + np.exp(-temperature * (confidence - 0.5)))
        
        return float(calibrated)
    
    def _determine_verdict(self, confidence: float) -> str:
        """
        Determine verdict based on confidence with better thresholds
        
        Args:
            confidence: Fake probability (0.0 = real, 1.0 = fake)
        """
        # More decisive thresholds
        if confidence >= 0.65:  # 65% fake confidence = FAKE
            return "fake"
        elif confidence <= 0.35:  # 35% fake confidence = REAL  
            return "real"
        else:
            return "inconclusive"  # Only 35-65% range is inconclusive
    
    def _create_analysis_signals(
        self, 
        model_predictions: Dict[str, float],
        ensemble_confidence: float
    ) -> List[AnalysisSignal]:
        """
        Create analysis signals
        
        Args:
            model_predictions: Individual model predictions
            ensemble_confidence: Ensemble confidence
            
        Returns:
            List of analysis signals
        """
        signals = []
        
        # Ensemble signal
        signals.append(AnalysisSignal(
            name="Ensemble Analysis",
            score=ensemble_confidence,
            description="Combined analysis from MesoNet and Xception"
        ))
        
        # Individual model signals
        signals.append(AnalysisSignal(
            name="MesoNet Detection",
            score=model_predictions.get("mesonet", 0.5),
            description="Lightweight CNN optimized for face manipulation"
        ))
        
        signals.append(AnalysisSignal(
            name="Xception Detection",
            score=model_predictions.get("xception", 0.5),
            description="Deep CNN with separable convolutions"
        ))
        
        # Agreement signal
        agreement = 1.0 - abs(model_predictions.get("mesonet", 0.5) - 
                             model_predictions.get("xception", 0.5))
        signals.append(AnalysisSignal(
            name="Model Agreement",
            score=agreement,
            description="Consensus between detection models"
        ))
        
        return signals
    
    def _extract_video_frames(
        self,
        video_path: str,
        max_frames: int,
        skip_similar: bool
    ) -> List[np.ndarray]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video
            max_frames: Maximum frames to extract
            skip_similar: Skip similar frames
            
        Returns:
            List of frames
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, total_frames // max_frames)
        
        prev_frame = None
        frame_count = 0
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if skip_similar and prev_frame is not None:
                    # Check similarity
                    similarity = self._calculate_frame_similarity(prev_frame, frame_rgb)
                    if similarity > 0.95:
                        frame_count += 1
                        continue
                
                frames.append(frame_rgb)
                prev_frame = frame_rgb
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def _calculate_frame_similarity(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two frames
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Similarity score (0-1)
        """
        # Resize for faster comparison
        size = (64, 64)
        f1_small = cv2.resize(frame1, size)
        f2_small = cv2.resize(frame2, size)
        
        # Calculate structural similarity
        diff = np.abs(f1_small.astype(float) - f2_small.astype(float))
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        return similarity
    
    def _get_video_frame_count(self, video_path: str) -> int:
        """Get total frame count of video"""
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps > 0:
            return frame_count / fps
        return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about ensemble models"""
        return {
            "ensemble": {
                "models": ["MesoNet", "Xception"],
                "weights": self.model_weights,
                "confidence_threshold": self.confidence_threshold
            },
            "mesonet": self.mesonet.get_model_info(),
            "xception": self.xception.get_model_info()
        }
