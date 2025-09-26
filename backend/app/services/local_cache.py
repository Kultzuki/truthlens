"""
Local Storage Cache for Analysis Results
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class LocalCache:
    """Simple file-based cache for storing analysis results"""
    
    def __init__(self, cache_dir: str = "./cache"):
        """Initialize local cache"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_hours = 24  # Cache for 24 hours
        
    def _get_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key"""
        return self.cache_dir / f"{key}.json"
    
    def get(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached result for file"""
        try:
            # Generate cache key from file hash
            file_hash = self._get_file_hash(file_path)
            cache_path = self._get_cache_path(file_hash)
            
            if not cache_path.exists():
                return None
            
            # Load cached data
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is expired
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                # Cache expired, delete it
                cache_path.unlink()
                return None
            
            logger.info(f"Cache hit for file: {file_path}")
            return cached_data['result']
            
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return None
    
    def set(self, file_path: str, result: Dict[str, Any]) -> bool:
        """Cache analysis result for file"""
        try:
            # Generate cache key from file hash
            file_hash = self._get_file_hash(file_path)
            cache_path = self._get_cache_path(file_hash)
            
            # Prepare cache data
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'file_path': file_path,
                'result': result
            }
            
            # Save to cache
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Cached result for file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Cache write error: {e}")
            return False
    
    def clear_old_cache(self) -> int:
        """Clear expired cache entries"""
        cleared = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    cached_time = datetime.fromisoformat(cached_data['timestamp'])
                    if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                        cache_file.unlink()
                        cleared += 1
                except:
                    # Delete corrupted cache files
                    cache_file.unlink()
                    cleared += 1
            
            logger.info(f"Cleared {cleared} expired cache entries")
            return cleared
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            return 0