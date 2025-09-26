"""
Test to verify predictions vary based on image characteristics
"""

import asyncio
import numpy as np
import cv2
from app.services.ensemble_detector import EnsembleDetector

async def test_varied_predictions():
    # Initialize ensemble detector
    config = {
        "mesonet_path": "./models/mesonet4.onnx",
        "xception_path": "./models/xception.onnx",
        "mesonet_weight": 0.4,
        "xception_weight": 0.6,
        "confidence_threshold": 0.7
    }
    
    detector = EnsembleDetector(config)
    await detector.warmup_models()
    
    # Create test images with different characteristics
    test_images = []
    
    # 1. High quality, sharp image with edges
    img1 = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(10):
        cv2.rectangle(img1, (i*20, i*20), (i*20+50, i*20+50), 
                     (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)), -1)
    test_images.append(("Sharp with edges", img1))
    
    # 2. Blurry image
    img2 = cv2.GaussianBlur(img1, (21, 21), 10)
    test_images.append(("Blurry", img2))
    
    # 3. Low contrast grayscale
    img3 = np.full((224, 224, 3), 128, dtype=np.uint8)
    img3 += np.random.randint(-20, 20, img3.shape, dtype=np.int16).astype(np.uint8)
    test_images.append(("Low contrast", img3))
    
    # 4. High saturation colorful
    img4 = np.random.randint(200, 255, (224, 224, 3), dtype=np.uint8)
    test_images.append(("High saturation", img4))
    
    # 5. Face-like features (simple)
    img5 = np.full((224, 224, 3), [200, 170, 150], dtype=np.uint8)  # Skin tone
    cv2.circle(img5, (80, 80), 20, (50, 50, 50), -1)  # Eye
    cv2.circle(img5, (144, 80), 20, (50, 50, 50), -1)  # Eye
    cv2.ellipse(img5, (112, 140), (30, 15), 0, 0, 180, (150, 100, 100), -1)  # Mouth
    test_images.append(("Face-like", img5))
    
    # 6. Noise pattern
    img6 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_images.append(("Random noise", img6))
    
    # 7. Smooth gradient
    img7 = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        img7[i, :] = [i, 255-i, 128]
    test_images.append(("Smooth gradient", img7))
    
    # 8. High texture detail
    img8 = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 4):
        for j in range(0, 224, 4):
            img8[i:i+2, j:j+2] = np.random.randint(0, 255, 3)
    test_images.append(("High texture", img8))
    
    print("Testing Image Quality-Based Predictions")
    print("=" * 60)
    print(f"{'Image Type':<20} {'MesoNet':<12} {'Xception':<12} {'Ensemble':<12} {'Verdict':<10}")
    print("-" * 60)
    
    for name, img in test_images:
        result = await detector.analyze_image(img, return_individual_scores=True)
        
        mesonet_score = result['individual_scores']['mesonet']
        xception_score = result['individual_scores']['xception']
        ensemble_score = result['confidence']
        verdict = result['verdict']
        
        print(f"{name:<20} {mesonet_score:<12.2%} {xception_score:<12.2%} {ensemble_score:<12.2%} {verdict:<10}")
    
    print("-" * 60)
    print("\nConfidence Range Summary:")
    confidences = [await detector.analyze_image(img, return_individual_scores=False) 
                  for _, img in test_images]
    all_scores = [c['confidence'] for c in confidences]
    print(f"Min confidence: {min(all_scores):.2%}")
    print(f"Max confidence: {max(all_scores):.2%}")
    print(f"Avg confidence: {np.mean(all_scores):.2%}")
    print(f"Std deviation: {np.std(all_scores):.2%}")

if __name__ == "__main__":
    asyncio.run(test_varied_predictions())