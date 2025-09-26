"""
Analysis Router - Deepfake Detection Endpoints
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from typing import Optional
import aiofiles
import os
import time
import uuid
from pathlib import Path

from app.models.analysis import (
    AnalysisResponse, 
    AnalysisRequest, 
    ErrorResponse,
    AnalysisSignal,
    AnalysisMetadata
)
from app.services.deepfake_detector import DeepfakeDetectorService
from app.services.file_handler import FileHandlerService

router = APIRouter()

# File upload constraints (following project rules)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ACCEPTED_FORMATS = {
    "video/mp4", "video/avi", "video/mov", "video/webm",
    "image/jpeg", "image/png", "image/gif", "image/webp"
}

# Initialize services
detector_service = DeepfakeDetectorService()
file_handler = FileHandlerService()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_media(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
) -> AnalysisResponse:
    """
    Analyze media for deepfake detection
    
    Args:
        file: Uploaded media file (video or image)
        url: URL to remote media file
        background_tasks: FastAPI background tasks
    
    Returns:
        AnalysisResponse: Detection results with confidence scores and heatmaps
    
    Raises:
        HTTPException: For validation errors or processing failures
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not file and not url:
            raise HTTPException(
                status_code=400,
                detail="Either file upload or URL must be provided"
            )
        
        if file and url:
            raise HTTPException(
                status_code=400,
                detail="Provide either file upload or URL, not both"
            )
        
        # Handle file upload
        if file:
            # Validate file type
            if file.content_type not in ACCEPTED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format. Accepted: {', '.join(ACCEPTED_FORMATS)}"
                )
            
            # Validate file size
            if file.size and file.size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            # Save uploaded file
            file_path = await file_handler.save_uploaded_file(file)
            input_identifier = file.filename or "uploaded_file"
            input_type = "video" if file.content_type.startswith("video") else "image"
        
        # Handle URL input
        else:
            try:
                file_path = await file_handler.download_from_url(url)
                # Determine type from downloaded file
                if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                    input_type = "video"
                else:
                    input_type = "image"
                input_identifier = url
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download from URL: {str(e)}"
                )
        
        # Perform deepfake analysis
        analysis_result = await detector_service.analyze_media(file_path, input_type)
        
        # Calculate processing time
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Create response
        response = AnalysisResponse(
            verdict=analysis_result["verdict"],
            confidence=analysis_result["confidence"],
            input_type=input_type,
            input=input_identifier,
            latency_ms=latency_ms,
            signals=analysis_result.get("signals", []),
            frame_analysis=analysis_result.get("frame_analysis"),
            metadata=analysis_result.get("metadata"),
            heatmap=analysis_result.get("heatmap")
        )
        
        # Schedule cleanup in background
        background_tasks.add_task(file_handler.cleanup_file, file_path)
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during analysis"
        )


@router.post("/analyze/batch")
async def analyze_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...)
) -> list[AnalysisResponse]:
    """
    Analyze multiple media files in batch
    
    Args:
        files: List of uploaded media files
        background_tasks: FastAPI background tasks
    
    Returns:
        List[AnalysisResponse]: Detection results for each file
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files per batch"
        )
    
    results = []
    for file in files:
        try:
            # Reuse single file analysis logic
            result = await analyze_media(background_tasks, file=file)
            results.append(result)
        except HTTPException as e:
            # Add error result for failed files
            error_result = AnalysisResponse(
                verdict="unknown",
                confidence=0.0,
                input_type="unknown",
                input=file.filename or "unknown",
                latency_ms=0,
                signals=[],
                frame_analysis=None,
                metadata=None,
                heatmap=None
            )
            results.append(error_result)
    
    return results


@router.get("/analyze/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Get status of long-running analysis (for future implementation)
    
    Args:
        analysis_id: Unique analysis identifier
    
    Returns:
        Analysis status information
    """
    # TODO: Implement analysis status tracking
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "progress": 100
    }


@router.get("/analyze/formats")
async def get_supported_formats():
    """
    Get list of supported file formats
    
    Returns:
        Dictionary of supported formats and constraints
    """
    return {
        "supported_formats": list(ACCEPTED_FORMATS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "max_batch_size": 10
    }