#!/usr/bin/env python3
"""
Test script to verify RealityDefender as primary service configuration
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.deepfake_detector import DeepfakeDetectorService
from app.services.realitydefender_service import RealityDefenderService


async def test_service_configuration():
    """Test the service configuration and status"""
    
    print("🧪 Testing Truthlens Service Configuration")
    print("=" * 50)
    
    # Test RealityDefender service directly
    print("\n1. Testing RealityDefender Service:")
    rd_service = RealityDefenderService()
    rd_info = rd_service.get_service_info()
    
    print(f"   ✓ Service: {rd_info['service']}")
    print(f"   ✓ Enabled: {rd_info['enabled']}")
    print(f"   ✓ Available: {rd_info['available']}")
    print(f"   ✓ Healthy: {rd_info['healthy']}")
    print(f"   ✓ API Key Configured: {rd_info['api_key_configured']}")
    print(f"   ✓ Timeout: {rd_info['timeout_ms']}ms")
    print(f"   ✓ Max Retries: {rd_info['max_retries']}")
    
    # Test main detector service
    print("\n2. Testing Main Detector Service:")
    detector = DeepfakeDetectorService()
    status = detector.get_service_status()
    
    print(f"   ✓ Primary Service: {status['primary_service']}")
    print(f"   ✓ Backup Service: {status['backup_service']}")
    print(f"   ✓ Fallback Enabled: {status['fallback_enabled']}")
    
    print("\n3. RealityDefender Status:")
    rd_status = status['reality_defender']
    print(f"   ✓ Service: {rd_status['service']}")
    print(f"   ✓ Enabled: {rd_status['enabled']}")
    print(f"   ✓ Available: {rd_status['available']}")
    print(f"   ✓ Healthy: {rd_status['healthy']}")
    
    print("\n4. Service Statistics:")
    stats = status['statistics']
    print(f"   ✓ Total Requests: {stats['total_requests']}")
    print(f"   ✓ Primary Success: {stats['primary_success']}")
    print(f"   ✓ Backup Used: {stats['backup_used']}")
    print(f"   ✓ Errors: {stats['errors']}")
    
    print("\n5. Environment Configuration:")
    env_vars = [
        "REALITYDEFENDER_API_KEY",
        "REALITYDEFENDER_ENABLED", 
        "PRIMARY_SERVICE",
        "BACKUP_SERVICE",
        "FALLBACK_ON_ERROR"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "NOT SET")
        if var == "REALITYDEFENDER_API_KEY" and value != "NOT SET":
            # Mask the API key for security
            value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***MASKED***"
        print(f"   ✓ {var}: {value}")
    
    print("\n" + "=" * 50)
    
    # Determine overall status
    if rd_status['available'] and rd_status['enabled']:
        print("🎯 STATUS: RealityDefender is PRIMARY service - READY!")
        print("   Primary detection will use RealityDefender SDK")
        print("   Backup detection will use local ensemble models")
    elif rd_status['enabled'] and not rd_status['available']:
        print("⚠️  STATUS: RealityDefender is enabled but NOT AVAILABLE")
        print("   Check your API key configuration in backend/.env")
        print("   System will fall back to ensemble models")
    else:
        print("🔄 STATUS: RealityDefender is DISABLED")
        print("   System will use ensemble models only")
    
    print("\n💡 Next Steps:")
    if not rd_status['api_key_configured']:
        print("   1. Get your API key from https://realitydefender.com")
        print("   2. Add it to backend/.env: REALITYDEFENDER_API_KEY=your-key")
        print("   3. Restart the backend server")
    else:
        print("   1. Start the backend server: uvicorn app.main:app --reload")
        print("   2. Test with a sample file upload")
        print("   3. Check /api/analyze/service-status endpoint")


def test_environment_setup():
    """Test environment setup"""
    print("\n🔧 Environment Setup Check:")
    
    # Check if .env file exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print("   ✓ .env file found")
    else:
        print("   ⚠️  .env file not found - copy from .env.example")
    
    # Check if models directory exists
    models_dir = Path(__file__).parent / "app" / "models"
    if models_dir.exists():
        print("   ✓ Models directory found")
        model_files = list(models_dir.glob("*.onnx"))
        print(f"   ✓ Found {len(model_files)} ONNX model files")
    else:
        print("   ⚠️  Models directory not found")


if __name__ == "__main__":
    print("Truthlens - RealityDefender Primary Service Test")
    print("=" * 60)
    
    # Test environment setup
    test_environment_setup()
    
    # Test service configuration
    asyncio.run(test_service_configuration())
    
    print("\n" + "=" * 60)
    print("Test completed! 🎉")
