"""
File Handler Service - Media File Management
"""

import aiofiles
import aiohttp
import os
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import UploadFile, HTTPException


class FileHandlerService:
    """Service for handling file uploads, downloads, and cleanup"""
    
    def __init__(self):
        self.upload_dir = Path("./uploads")
        self.upload_dir.mkdir(exist_ok=True)
        
        # File size limits
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.max_download_size = 200 * 1024 * 1024  # 200MB for URLs
        
        # Supported formats
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        self.supported_formats = self.supported_video_formats | self.supported_image_formats
    
    def _generate_unique_filename(self, original_filename: str) -> str:
        """Generate unique filename to avoid conflicts"""
        file_extension = Path(original_filename).suffix.lower()
        unique_id = str(uuid.uuid4())
        return f"{unique_id}{file_extension}"
    
    def _validate_file_extension(self, filename: str) -> bool:
        """Validate file extension"""
        file_extension = Path(filename).suffix.lower()
        return file_extension in self.supported_formats
    
    async def save_uploaded_file(self, file: UploadFile) -> str:
        """
        Save uploaded file to disk
        
        Args:
            file: FastAPI UploadFile object
        
        Returns:
            str: Path to saved file
        
        Raises:
            HTTPException: For validation or I/O errors
        """
        try:
            # Validate filename
            if not file.filename:
                raise HTTPException(status_code=400, detail="Filename is required")
            
            if not self._validate_file_extension(file.filename):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file format. Supported: {', '.join(self.supported_formats)}"
                )
            
            # Generate unique filename
            unique_filename = self._generate_unique_filename(file.filename)
            file_path = self.upload_dir / unique_filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                
                # Check file size
                if len(content) > self.max_file_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum size: {self.max_file_size // (1024*1024)}MB"
                    )
                
                await f.write(content)
            
            return str(file_path)
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file: {str(e)}"
            )
    
    async def download_from_url(self, url: str) -> str:
        """
        Download file from URL
        
        Args:
            url: URL to download from
        
        Returns:
            str: Path to downloaded file
        
        Raises:
            HTTPException: For download or validation errors
        """
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise HTTPException(status_code=400, detail="Invalid URL format")
            
            # Extract filename from URL
            url_path = Path(parsed_url.path)
            filename = url_path.name if url_path.name else "downloaded_file"
            
            # Ensure filename has extension
            if not url_path.suffix:
                filename += ".mp4"  # Default to mp4 for videos
            
            if not self._validate_file_extension(filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format in URL. Supported: {', '.join(self.supported_formats)}"
                )
            
            # Generate unique filename
            unique_filename = self._generate_unique_filename(filename)
            file_path = self.upload_dir / unique_filename
            
            # Download file
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download from URL. Status: {response.status}"
                        )
                    
                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_download_size:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File too large. Maximum size: {self.max_download_size // (1024*1024)}MB"
                        )
                    
                    # Save file
                    async with aiofiles.open(file_path, 'wb') as f:
                        downloaded_size = 0
                        async for chunk in response.content.iter_chunked(8192):
                            downloaded_size += len(chunk)
                            
                            # Check size during download
                            if downloaded_size > self.max_download_size:
                                await f.close()
                                os.unlink(file_path)  # Clean up partial file
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"File too large. Maximum size: {self.max_download_size // (1024*1024)}MB"
                                )
                            
                            await f.write(chunk)
            
            return str(file_path)
        
        except HTTPException:
            raise
        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download from URL: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error during download: {str(e)}"
            )
    
    async def cleanup_file(self, file_path: str) -> None:
        """
        Clean up temporary file
        
        Args:
            file_path: Path to file to delete
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                print(f"Cleaned up file: {file_path}")
        except Exception as e:
            print(f"Failed to cleanup file {file_path}: {e}")
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get file information
        
        Args:
            file_path: Path to file
        
        Returns:
            dict: File information
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": "File not found"}
            
            stat = path.stat()
            return {
                "filename": path.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "extension": path.suffix.lower(),
                "is_video": path.suffix.lower() in self.supported_video_formats,
                "is_image": path.suffix.lower() in self.supported_image_formats,
                "created_at": stat.st_ctime,
                "modified_at": stat.st_mtime
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old uploaded files
        
        Args:
            max_age_hours: Maximum age of files to keep (in hours)
        
        Returns:
            int: Number of files cleaned up
        """
        import time
        
        cleaned_count = 0
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        try:
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        await self.cleanup_file(str(file_path))
                        cleaned_count += 1
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        return cleaned_count