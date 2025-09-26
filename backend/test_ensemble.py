#!/usr/bin/env python3
"""
Test script for Truthlens Ensemble Deepfake Detection
"""

import asyncio
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.ensemble_detector import EnsembleDetector
from app.services.models.mesonet import MesoNetModel
from app.services.models.xception import XceptionModel
from app.services.heatmap_generator import GradCAMGenerator


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


async def test_individual_models():
    """Test individual model components"""
    print_section("Testing Individual Models")
    
    # Test MesoNet
    print("\n1. Testing MesoNet Model:")
    mesonet = MesoNetModel()
    print(f"   - Model path: {mesonet.model_path}")
    print(f"   - Input size: {mesonet.input_size}")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test preprocessing
    preprocessed = mesonet.preprocess(dummy_image)
    print(f"   - Preprocessing: ✓ (shape: {preprocessed.shape})")
    
    # Test prediction
    confidence, features = mesonet.predict(preprocessed)
    print(f"   - Prediction: ✓ (confidence: {confidence:.3f})")
    print(f"   - Model info: {mesonet.get_model_info()['name']}")
    
    # Test Xception
    print("\n2. Testing Xception Model:")
    xception = XceptionModel()
    print(f"   - Model path: {xception.model_path}")
    print(f"   - Input size: {xception.input_size}")
    
    # Test preprocessing
    preprocessed = xception.preprocess(dummy_image)
    print(f"   - Preprocessing: ✓ (shape: {preprocessed.shape})")
    
    # Test prediction
    confidence, features = xception.predict(preprocessed)
    print(f"   - Prediction: ✓ (confidence: {confidence:.3f})")
    print(f"   - Model info: {xception.get_model_info()['name']}")


async def test_heatmap_generator():
    """Test Grad-CAM heatmap generator"""
    print_section("Testing Heatmap Generator")
    
    generator = GradCAMGenerator()
    
    # Create dummy feature maps
    dummy_features = np.random.randn(1, 64, 32, 32).astype(np.float32)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    print("\n1. Testing Grad-CAM generation:")
    heatmap = generator.generate_gradcam(dummy_features)
    print(f"   - Heatmap shape: {heatmap.shape}")
    print(f"   - Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    print("\n2. Testing heatmap overlay:")
    overlay = generator.apply_heatmap_to_image(dummy_image, heatmap)
    print(f"   - Overlay shape: {overlay.shape}")
    
    print("\n3. Testing base64 conversion:")
    base64_str = generator.heatmap_to_base64(heatmap)
    print(f"   - Base64 length: {len(base64_str)} chars")
    print(f"   - Starts with: {base64_str[:50]}...")


async def test_ensemble_detector():
    """Test ensemble detector"""
    print_section("Testing Ensemble Detector")
    
    config = {
        "mesonet_weight": 0.4,
        "xception_weight": 0.6,
        "confidence_threshold": 0.7
    }
    
    ensemble = EnsembleDetector(config)
    
    print("\n1. Model configuration:")
    model_info = ensemble.get_model_info()
    print(f"   - Models: {model_info['ensemble']['models']}")
    print(f"   - Weights: {model_info['ensemble']['weights']}")
    print(f"   - Threshold: {model_info['ensemble']['confidence_threshold']}")
    
    print("\n2. Testing model warmup:")
    start = time.time()
    await ensemble.warmup_models()
    print(f"   - Warmup time: {(time.time() - start)*1000:.1f}ms")
    
    print("\n3. Testing image analysis:")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    start = time.time()
    result = await ensemble.analyze_image(dummy_image, return_individual_scores=True)
    analysis_time = (time.time() - start) * 1000
    
    print(f"   - Verdict: {result['verdict']}")
    print(f"   - Confidence: {result['confidence']:.3f}")
    print(f"   - Analysis time: {analysis_time:.1f}ms")
    
    if 'individual_scores' in result:
        print(f"   - MesoNet score: {result['individual_scores'].get('mesonet', 0):.3f}")
        print(f"   - Xception score: {result['individual_scores'].get('xception', 0):.3f}")
    
    print(f"   - Signals: {len(result.get('signals', []))} generated")
    for signal in result.get('signals', [])[:3]:
        print(f"      • {signal.name}: {signal.score:.3f}")
    
    print(f"   - Heatmap: {'✓' if result.get('heatmap') else '✗'}")


async def test_video_processing():
    """Test video processing capabilities"""
    print_section("Testing Video Processing")
    
    # Create a mock video path (won't actually process)
    ensemble = EnsembleDetector()
    
    print("\n1. Frame similarity calculation:")
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame2 = np.clip(frame2.astype(int) + np.random.randint(-5, 5, frame2.shape), 0, 255).astype(np.uint8)
    
    similarity = ensemble._calculate_frame_similarity(frame1, frame2)
    print(f"   - Similar frames: {similarity:.3f}")
    
    frame3 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    similarity = ensemble._calculate_frame_similarity(frame1, frame3)
    print(f"   - Different frames: {similarity:.3f}")


async def benchmark_performance():
    """Benchmark performance metrics"""
    print_section("Performance Benchmark")
    
    ensemble = EnsembleDetector()
    await ensemble.warmup_models()
    
    # Test different image sizes
    sizes = [(224, 224), (480, 640), (720, 1280), (1080, 1920)]
    
    print("\nImage size benchmarks:")
    for height, width in sizes:
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Run multiple times for average
        times = []
        for _ in range(3):
            start = time.time()
            result = await ensemble.analyze_image(image)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        print(f"   - {width}x{height}: {avg_time:.1f}ms (avg), {min(times):.1f}ms (min)")


async def main():
    """Main test runner"""
    print("\n" + "🚀"*30)
    print("  TRUTHLENS ENSEMBLE DETECTOR TEST SUITE")
    print("🚀"*30)
    
    try:
        # Run all tests
        await test_individual_models()
        await test_heatmap_generator()
        await test_ensemble_detector()
        await test_video_processing()
        await benchmark_performance()
        
        print_section("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        
        # Summary
        print("\n📊 Summary:")
        print("   • MesoNet Model: ✓")
        print("   • Xception Model: ✓")
        print("   • Grad-CAM Heatmap: ✓")
        print("   • Ensemble Detector: ✓")
        print("   • Video Processing: ✓")
        print("   • Performance: Optimized for hackathon demo")
        
        print("\n💡 Next steps:")
        print("   1. Download actual ONNX models and place in ./models/")
        print("   2. Test with real images/videos")
        print("   3. Fine-tune weights and thresholds")
        print("   4. Set up Redis for caching")
        
    except Exception as e:
        print_section(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)