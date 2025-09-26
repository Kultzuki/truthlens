# Truthlens Backend Setup Instructions

## ✅ What's Already Done

### 1. **Ensemble Model Architecture**
- ✅ MesoNet model service (`app/services/models/mesonet.py`)
- ✅ Xception model service (`app/services/models/xception.py`) 
- ✅ Ensemble detector combining both models (`app/services/ensemble_detector.py`)
- ✅ Weighted averaging with configurable weights (MesoNet: 0.4, Xception: 0.6)

### 2. **Grad-CAM Heatmap Generation (Priority Feature)**
- ✅ Full Grad-CAM implementation (`app/services/heatmap_generator.py`)
- ✅ Generates explainable heatmaps showing which regions influenced detection
- ✅ Base64 encoding for frontend display
- ✅ Multi-scale heatmap support
- ✅ Side-by-side comparison generation

### 3. **Video Processing Optimizations**
- ✅ Frame similarity detection (skip frames with >95% similarity)
- ✅ Parallel model inference using ThreadPoolExecutor
- ✅ Batch processing for videos (processes every 10th frame)
- ✅ Temporal consistency analysis

### 4. **Configuration & Testing**
- ✅ Environment configuration file (`.env.example`)
- ✅ Comprehensive test suite (`test_ensemble.py`)
- ✅ Performance benchmarking included

## 🔧 Manual Setup Required

### 1. **Download ONNX Models**

You need to obtain the pre-trained ONNX models and place them in the `models/` directory:

```bash
# Create models directory if not exists
mkdir models

# Download models (URLs are examples - get actual models)
# Option 1: MesoNet-4 ONNX
wget https://github.com/DariusAf/MesoNet/releases/download/v1.0/mesonet4.onnx -O models/mesonet4.onnx

# Option 2: Xception ONNX
wget https://github.com/onnx/models/raw/main/vision/classification/xception/model/xception.onnx -O models/xception.onnx

# OR convert from TensorFlow/PyTorch models
python convert_to_onnx.py --model mesonet --output models/mesonet4.onnx
python convert_to_onnx.py --model xception --output models/xception.onnx
```

### 2. **Install Dependencies**

```bash
# Activate virtual environment
..\venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source ../venv/bin/activate    # Linux/Mac

# Install required packages
pip install -r requirements.txt
```

### 3. **Set Up Redis (Optional but Recommended)**

For caching support during the hackathon:

```bash
# Windows - Using Docker
docker run -d -p 6379:6379 redis

# OR using Windows Subsystem for Linux (WSL)
sudo apt-get install redis-server
redis-server

# Test connection
redis-cli ping
```

### 4. **Configure Environment Variables**

```bash
# Copy example to .env
cp .env.example .env

# Edit .env with your settings
notepad .env  # or your preferred editor
```

Key variables to update:
- `MESONET_MODEL_PATH`: Path to MesoNet ONNX model
- `XCEPTION_MODEL_PATH`: Path to Xception ONNX model
- `MESONET_WEIGHT`: Weight for MesoNet (default 0.4)
- `XCEPTION_WEIGHT`: Weight for Xception (default 0.6)
- `CONFIDENCE_THRESHOLD`: Detection threshold (default 0.7)
- `REDIS_URL`: Redis connection string (if using)

### 5. **Test the Setup**

```bash
# Run test suite
python test_ensemble.py

# Start the backend server
python -m uvicorn app.main:app --reload --port 8000

# Test API endpoint
curl http://localhost:8000/health
```

### 6. **Prepare Demo Content**

Create a `demo/` folder with test videos and images:

```bash
mkdir demo
# Add sample deepfakes and real videos
# Recommended: 
# - 2-3 deepfake videos (different types)
# - 2-3 real videos
# - Several test images
```

## 🚀 Quick Start Commands

```bash
# Terminal 1 - Backend
cd backend
..\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend 
cd frontend
npm run dev

# Terminal 3 - Redis (if using)
redis-server
```

## 📊 Performance Tips for Hackathon

1. **Pre-warm Models**: The first request is slower due to model loading. Pre-warm by sending a dummy request on startup.

2. **Use Mock Mode**: If models aren't available, the system automatically uses mock predictions for demo purposes.

3. **Optimize for Demo**:
   - Keep videos under 30 seconds
   - Use 720p resolution for best speed/quality balance
   - Enable Redis caching for repeated demos

4. **Heatmap Quality**:
   - The heatmap is the key differentiator
   - Adjust `alpha` in `GradCAMGenerator` for better visibility
   - Use side-by-side comparison for impact

## 🐛 Troubleshooting

### Models Not Loading
- Check file paths in `.env`
- Ensure ONNX models are in correct format
- Verify file permissions

### Slow Performance
- Reduce `MAX_FRAMES_TO_PROCESS` in `.env`
- Enable Redis caching
- Use smaller input resolutions
- Consider using GPU if available

### Memory Issues
- Implement model unloading after idle time
- Process videos in chunks
- Clear cache periodically

## 📝 API Endpoints

- `POST /api/analyze`: Main analysis endpoint
- `GET /health`: Health check
- `GET /api/analyze/formats`: Supported formats

## 🎯 Hackathon Presentation Notes

1. **Lead with the Heatmap**: This is your unique feature - emphasize the explainable AI aspect

2. **Show Real-time Analysis**: Demonstrate on live webcam or uploaded content

3. **Highlight Ensemble Approach**: Explain how combining models improves accuracy

4. **Performance Metrics**: Show the speed optimizations (frame skipping, parallel processing)

5. **Fallback Ready**: System works even without models (uses mock data) - no demo failures!

## 📧 Support

If you encounter issues during setup, check:
1. `test_ensemble.py` output for specific errors
2. Backend logs when running with `--reload`
3. Network tab in browser for API errors