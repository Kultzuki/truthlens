"""
Deepfake Detection Service - Core ML Pipeline
"""

import os
import time
import aiohttp
import base64
from typing import Dict, Any
from pathlib import Path

import cv2
import numpy as np

from app.models.analysis import (
    AnalysisSignal, 
    FrameAnalysis, 
    AnalysisMetadata,
    AnalysisRegion
)
from app.services.ensemble_detector import EnsembleDetector
from app.services.realitydefender_service import RealityDefenderService


class DeepfakeDetectorService:
    """Service for deepfake detection using RealityDefender as PRIMARY + ensemble as BACKUP"""

    def __init__(self):
        # Service configuration
        self.primary_service = os.getenv("PRIMARY_SERVICE", "realitydefender")
        self.backup_service = os.getenv("BACKUP_SERVICE", "ensemble")
        self.fallback_on_error = os.getenv("FALLBACK_ON_ERROR", "true").lower() == "true"
        self.fallback_timeout_ms = int(os.getenv("FALLBACK_TIMEOUT_MS", "5000"))

        # Initialize RealityDefender service (PRIMARY)
        self.reality_defender = RealityDefenderService()

        # Load configuration for ensemble backup
        config = {
            "mesonet_path": os.getenv("MESONET_MODEL_PATH", "./app/models/mesonet4.onnx"),
            "xception_path": os.getenv("XCEPTION_MODEL_PATH", "./app/models/xception.onnx"),
            "mesonet_weight": float(os.getenv("MESONET_WEIGHT", "0.4")),
            "xception_weight": float(os.getenv("XCEPTION_WEIGHT", "0.6")),
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        }

        self.max_frames_to_process = int(os.getenv("MAX_FRAMES_TO_PROCESS", "30"))

        # Initialize ensemble detector as backup
        self.ensemble = EnsembleDetector(config)
        self.ensemble_loaded = False

        # Service statistics
        self.stats = {
            "total_requests": 0,
            "primary_success": 0,
            "backup_used": 0,
            "errors": 0
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and configuration"""
        return {
            "primary_service": self.primary_service,
            "backup_service": self.backup_service,
            "fallback_enabled": self.fallback_on_error,
            "reality_defender": self.reality_defender.get_service_info(),
            "ensemble_loaded": self.ensemble_loaded,
            "statistics": self.stats.copy()
        }

    async def analyze_media(self, file_path: str, input_type: str) -> Dict[str, Any]:
        """
        Main analysis method - RealityDefender as PRIMARY, ensemble as BACKUP

        Args:
            file_path: Path to the media file
            input_type: Type of input ("image" or "video")

        Returns:
            Analysis results in standardized format
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        # Determine MIME type for RealityDefender
        file_type = self._get_mime_type(file_path, input_type)

        # PRIMARY: Try RealityDefender first
        if self.primary_service == "realitydefender" and self.reality_defender.is_available():
            try:
                print("🎯 Using RealityDefender as PRIMARY detection service...")

                result = await self.reality_defender.detect_file(file_path, file_type)

                self.stats["primary_success"] += 1
                processing_time = (time.time() - start_time) * 1000

                print(f"✅ PRIMARY service success: {result['verdict']} ({result['confidence']:.3f}) in {processing_time:.0f}ms")
                return result

            except Exception as e:
                print(f"❌ PRIMARY service (RealityDefender) failed: {e}")
                self.stats["errors"] += 1

                if not self.fallback_on_error:
                    print("🚫 Fallback disabled - returning error")
                    raise Exception(f"Primary detection service failed: {str(e)}")

                print("🔄 Falling back to BACKUP service (ensemble models)...")

        # BACKUP: Fallback to ensemble models
        if self.fallback_on_error:
            try:
                self.stats["backup_used"] += 1
                result = await self._analyze_with_ensemble(file_path, input_type)

                processing_time = (time.time() - start_time) * 1000
                print(f"✅ BACKUP service success: {result['verdict']} ({result['confidence']:.3f}) in {processing_time:.0f}ms")

                # Mark result as coming from backup service
                if "metadata" in result:
                    result["metadata"]["service_used"] = "backup_ensemble"
                    result["metadata"]["primary_failed"] = True

                return result

            except Exception as e:
                print(f"❌ BACKUP service (ensemble) also failed: {e}")
                self.stats["errors"] += 1
                raise Exception(f"Both primary and backup detection services failed. Primary: RealityDefender error, Backup: {str(e)}")

        # If we get here, no service was available
        raise Exception("No detection service available")
    
    async def _analyze_with_ensemble(self, file_path: str, input_type: str) -> Dict[str, Any]:
        """BACKUP analysis using ensemble models (MesoNet + Xception)"""

        # Warmup models on first use
        if not self.ensemble_loaded:
            try:
                print("🔧 Warming up ensemble models (backup service)...")
                await self.ensemble.warmup_models()
                self.ensemble_loaded = True
                print("✅ Ensemble models ready")
            except Exception as e:
                print(f"⚠️ Ensemble model warmup failed: {e}")
                raise Exception(f"Failed to initialize backup detection models: {str(e)}")

        try:
            if input_type == "image":
                # Single image analysis
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"Cannot load image: {file_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Use ensemble for analysis
                result = await self.ensemble.analyze_image(image_rgb, return_individual_scores=True)

                # Add backup service metadata
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["source"] = "Backup Ensemble (MesoNet + Xception)"
                result["metadata"]["service_type"] = "backup"

                # Add backup service signal
                if "signals" not in result:
                    result["signals"] = []
                result["signals"].insert(0, {
                    "name": "Backup Service Notice",
                    "score": result.get("confidence", 0.5),
                    "description": "Analysis performed by backup ensemble models due to primary service unavailability"
                })

                return result

            else:
                # Video analysis
                video_result = await self.ensemble.analyze_video(
                    file_path,
                    max_frames=self.max_frames_to_process,
                    skip_similar=True
                )

                # Add backup service metadata
                if "metadata" not in video_result:
                    video_result["metadata"] = {}
                video_result["metadata"]["source"] = "Backup Ensemble (MesoNet + Xception)"
                video_result["metadata"]["service_type"] = "backup"

                # Add backup service signal
                if "signals" not in video_result:
                    video_result["signals"] = []
                video_result["signals"].insert(0, AnalysisSignal(
                    name="Backup Service Notice",
                    score=video_result.get("confidence", 0.5),
                    description="Analysis performed by backup ensemble models due to primary service unavailability"
                ))

                return video_result

        except Exception as e:
            print(f"❌ Backup ensemble analysis error: {e}")
            import traceback
            traceback.print_exc()

            # Return error result
            return {
                "verdict": "unknown",
                "confidence": 0.0,
                "signals": [
                    {
                        "name": "Analysis Error",
                        "score": 0.0,
                        "description": f"Both primary and backup services failed: {str(e)}"
                    }
                ],
                "frame_analysis": None,
                "metadata": {
                    "processed_frames": 0,
                    "total_frames": 0,
                    "resolution": "unknown",
                    "source": "Error - All Services Failed",
                    "service_type": "error"
                },
                "heatmap": None
            }
    
    def _get_mime_type(self, file_path: str, input_type: str) -> str:
        """Get MIME type from file extension"""
        extension = file_path.lower().split('.')[-1]
        
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg', 
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'mp4': 'video/mp4',
            'avi': 'video/avi',
            'mov': 'video/mov',
            'webm': 'video/webm'
        }
        
        return mime_types.get(extension, 'application/octet-stream')



