"""
RealityDefender API Service - Using Official SDK
Enhanced for Primary Detection with Robust Error Handling
"""

import os
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

# Import the official Reality Defender SDK
try:
    from realitydefender import RealityDefender
    try:
        from realitydefender.exceptions import RealityDefenderError
    except ImportError:
        # Fallback if exceptions module doesn't exist
        class RealityDefenderError(Exception):
            pass
    REALITYDEFENDER_AVAILABLE = True
except ImportError:
    # Fallback if RealityDefender SDK is not installed
    logger.warning("RealityDefender SDK not installed. Install with: pip install realitydefender")
    RealityDefender = None
    class RealityDefenderError(Exception):
        pass
    REALITYDEFENDER_AVAILABLE = False


class RealityDefenderService:
    """Primary deepfake detection using RealityDefender Official SDK"""

    def __init__(self):
        # Load environment variables with fallback
        from dotenv import load_dotenv
        load_dotenv()  # Load .env file

        # Configuration from environment with hardcoded fallback
        self.api_key = os.getenv("REALITYDEFENDER_API_KEY") or "rd_ab34e8cdb05776ae_960a78b5a7117cfdeb8de770d2705274"
        self.enabled = os.getenv("REALITYDEFENDER_ENABLED", "true").lower() == "true"
        self.timeout = int(os.getenv("REALITYDEFENDER_TIMEOUT", "60000"))  # milliseconds
        self.max_retries = int(os.getenv("REALITYDEFENDER_MAX_RETRIES", "2"))

        # SDK client
        self.client = None
        self._initialize_client()

        # Service status tracking
        self.last_health_check = 0
        self.health_check_interval = 300  # 5 minutes
        self.is_healthy = False

    def _initialize_client(self):
        """Initialize the Reality Defender SDK client"""
        if not self.enabled:
            print("RealityDefender service is disabled")
            logger.info("RealityDefender service is disabled")
            return

        if not REALITYDEFENDER_AVAILABLE:
            print("❌ RealityDefender SDK not installed. Run: pip install realitydefender")
            logger.error("RealityDefender SDK not installed")
            self.client = None
            self.is_healthy = False
            return

        if self.api_key and self.api_key != "your-api-key-here":
            try:
                print(f"🔧 Initializing RealityDefender SDK with API key: {self.api_key[:8]}...{self.api_key[-4:]}")
                self.client = RealityDefender(api_key=self.api_key)
                self.is_healthy = True
                print("✅ RealityDefender SDK client initialized successfully")
                logger.info("✅ RealityDefender SDK client initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize RealityDefender SDK: {e}")
                logger.error(f"❌ Failed to initialize RealityDefender SDK: {e}")
                self.client = None
                self.is_healthy = False
        else:
            print("⚠️ RealityDefender API key not configured")
            logger.warning("⚠️ RealityDefender API key not configured")
            self.client = None
            self.is_healthy = False

    def is_available(self) -> bool:
        """Check if RealityDefender SDK is configured and available"""
        available = self.enabled and REALITYDEFENDER_AVAILABLE and self.client is not None
        print(f"🔍 Availability check: enabled={self.enabled}, sdk_available={REALITYDEFENDER_AVAILABLE}, client={self.client is not None}, result={available}")
        return available

    async def health_check(self) -> bool:
        """Perform health check on RealityDefender service"""
        current_time = time.time()

        # Skip if recently checked
        if current_time - self.last_health_check < self.health_check_interval:
            return self.is_healthy

        if not self.client:
            self.is_healthy = False
            return False

        try:
            # Simple health check - could be enhanced with actual API call
            self.is_healthy = True
            self.last_health_check = current_time
            logger.debug("✅ RealityDefender health check passed")
            return True
        except Exception as e:
            logger.warning(f"⚠️ RealityDefender health check failed: {e}")
            self.is_healthy = False
            self.last_health_check = current_time
            return False

    async def detect_file(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Detect deepfake using RealityDefender Official SDK

        Args:
            file_path: Path to the file
            file_type: MIME type of the file

        Returns:
            Detection results in standardized format
        """
        if not self.is_available():
            raise ValueError("RealityDefender SDK not configured or available")

        # Perform health check
        if not await self.health_check():
            raise Exception("RealityDefender service is not healthy")

        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"🔍 Using RealityDefender SDK for {file_path} (attempt {attempt + 1})")

                # Use the SDK's upload and get_result methods as per official docs
                # These methods are already async, so we call them directly
                upload_response = await self.client.upload(file_path=file_path)

                request_id = upload_response["request_id"]
                logger.info(f"📤 File uploaded successfully. Request ID: {request_id}")

                # Get results by polling until completion
                result = await self.client.get_result(request_id)

                processing_time = (time.time() - start_time) * 1000
                logger.info(f"✅ RealityDefender analysis complete: {result['status']} (Score: {result['score']:.3f}) in {processing_time:.0f}ms")

                # Transform SDK result to our standardized format
                return self._transform_sdk_result(result, file_path, processing_time)

            except RealityDefenderError as e:
                logger.error(f"❌ RealityDefender SDK error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    raise Exception(f"RealityDefender SDK failed after {self.max_retries + 1} attempts: {str(e)}")
                await asyncio.sleep(1)  # Brief delay before retry

            except Exception as e:
                logger.error(f"❌ Unexpected error during RealityDefender detection (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    raise Exception(f"RealityDefender detection failed after {self.max_retries + 1} attempts: {str(e)}")
                await asyncio.sleep(1)  # Brief delay before retry

    def _transform_sdk_result(self, sdk_result: Dict[str, Any], file_path: str, processing_time: float = 0) -> Dict[str, Any]:
        """
        Transform RealityDefender SDK result to our standardized format

        Args:
            sdk_result: Result dictionary from RealityDefender SDK
            file_path: Path to the analyzed file
            processing_time: Processing time in milliseconds

        Returns:
            Standardized detection result
        """

        # Extract confidence score (SDK returns score between 0 and 1)
        confidence = sdk_result.get('score', 0.5)
        status = sdk_result.get('status', 'UNKNOWN')

        # Determine verdict based on RealityDefender status and confidence
        if status in ['MANIPULATED', 'FAKE']:
            verdict = "fake"
        elif status in ['AUTHENTIC', 'REAL']:
            verdict = "real"
        elif confidence >= 0.65:
            verdict = "fake"
        elif confidence <= 0.35:
            verdict = "real"
        else:
            verdict = "inconclusive"

        # Build standardized response
        result = {
            "verdict": verdict,
            "confidence": confidence,
            "signals": [
                {
                    "name": "RealityDefender Primary",
                    "score": confidence,
                    "description": f"Official Reality Defender detection service (Status: {status})"
                }
            ],
            "frame_analysis": None,
            "metadata": {
                "processed_frames": 1,
                "total_frames": 1,
                "resolution": "unknown",
                "source": "RealityDefender SDK",
                "status": status,
                "processing_time_ms": int(processing_time)
            },
            "heatmap": None,  # SDK doesn't provide heatmaps in current version
            "individual_scores": {
                "realitydefender": confidence
            }
        }

        # Add model-specific results if available
        models = sdk_result.get('models', [])
        if models:
            model_signals = []
            model_scores = {}

            for model in models:
                model_name = model.get('name', 'unknown_model')
                model_score = model.get('score', 0.5)
                model_status = model.get('status', 'completed')

                model_signals.append({
                    "name": f"RealityDefender - {model_name}",
                    "score": model_score,
                    "description": f"Model: {model_name} (Status: {model_status})"
                })

                model_scores[model_name.lower().replace(' ', '_')] = model_score

            # Add model-specific signals
            result["signals"].extend(model_signals)
            result["individual_scores"].update(model_scores)

        return result

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about RealityDefender service status"""
        return {
            "service": "RealityDefender",
            "enabled": self.enabled,
            "available": self.is_available(),
            "healthy": self.is_healthy,
            "api_key_configured": bool(self.api_key and self.api_key != "your-api-key-here"),
            "timeout_ms": self.timeout,
            "max_retries": self.max_retries,
            "last_health_check": self.last_health_check
        }

