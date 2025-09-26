"""
Pydantic Models for Deepfake Analysis
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import base64


class AnalysisRegion(BaseModel):
    """Region of interest in analysis"""
    x: float = Field(..., ge=0, le=1, description="X coordinate (normalized 0-1)")
    y: float = Field(..., ge=0, le=1, description="Y coordinate (normalized 0-1)")
    width: float = Field(..., ge=0, le=1, description="Width (normalized 0-1)")
    height: float = Field(..., ge=0, le=1, description="Height (normalized 0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")


class AnalysisSignal(BaseModel):
    """Individual analysis signal/feature"""
    name: str = Field(..., description="Signal name")
    score: float = Field(..., ge=0, le=1, description="Signal score (0-1)")
    description: Optional[str] = Field(None, description="Signal description")


class FrameAnalysis(BaseModel):
    """Analysis results for a specific frame"""
    timestamp: float = Field(..., ge=0, description="Frame timestamp in seconds")
    confidence: float = Field(..., ge=0, le=1, description="Frame confidence score")
    regions: List[AnalysisRegion] = Field(default_factory=list, description="Detected regions")


class AnalysisMetadata(BaseModel):
    """Metadata about the analyzed media"""
    processed_frames: Optional[int] = Field(None, description="Number of frames processed")
    total_frames: Optional[int] = Field(None, description="Total frames in video")
    resolution: Optional[str] = Field(None, description="Media resolution (e.g., '1920x1080')")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    format: Optional[str] = Field(None, description="Media format")


class AnalysisRequest(BaseModel):
    """Request model for analysis"""
    url: Optional[str] = Field(None, description="URL to analyze (if not uploading file)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/video.mp4"
            }
        }


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    verdict: Literal["real", "fake", "unknown"] = Field(..., description="Analysis verdict")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence score")
    input_type: Literal["image", "video", "url"] = Field(..., description="Type of input analyzed")
    input: str = Field(..., description="Input identifier (filename or URL)")
    latency_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    signals: List[AnalysisSignal] = Field(default_factory=list, description="Analysis signals")
    frame_analysis: Optional[List[FrameAnalysis]] = Field(None, description="Per-frame analysis")
    metadata: Optional[AnalysisMetadata] = Field(None, description="Media metadata")
    heatmap: Optional[str] = Field(None, description="Base64 encoded heatmap image")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    
    @validator('heatmap')
    def validate_heatmap(cls, v):
        """Validate base64 heatmap - accepts data URLs"""
        if v is not None:
            # Accept data URLs (data:image/png;base64,...) or raw base64
            if v.startswith('data:image/'):
                # It's a data URL, extract the base64 part for validation
                try:
                    # Split on comma to get the base64 part
                    _, base64_data = v.split(',', 1)
                    base64.b64decode(base64_data)
                except Exception:
                    raise ValueError("Invalid base64 heatmap data")
            else:
                # It's raw base64
                try:
                    base64.b64decode(v)
                except Exception:
                    raise ValueError("Invalid base64 heatmap data")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "verdict": "fake",
                "confidence": 0.85,
                "input_type": "video",
                "input": "sample_video.mp4",
                "latency_ms": 2500,
                "signals": [
                    {
                        "name": "facial_inconsistency",
                        "score": 0.78,
                        "description": "Detected facial feature inconsistencies"
                    },
                    {
                        "name": "temporal_artifacts",
                        "score": 0.65,
                        "description": "Temporal inconsistencies between frames"
                    }
                ],
                "metadata": {
                    "processed_frames": 30,
                    "total_frames": 300,
                    "resolution": "1920x1080",
                    "duration": 10.0
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid file format",
                "detail": "Only MP4, AVI, JPEG, PNG files are supported",
                "code": "INVALID_FORMAT"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")