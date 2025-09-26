# Truthlens - AI-Powered Deepfake Detection System

A comprehensive deepfake detection system built for hackathons, featuring ensemble ML models with explainable AI through heatmaps and confidence scores.

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.9+ with pip
- **Git** for version control

### 1. Clone Repository
```powershell
git clone <repository-url>
Set-Location Truthlens
```

### 2. Backend Setup
```powershell
Set-Location backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Setup environment
Copy-Item .env.example .env
# Edit .env with your configuration

# Start backend server
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup
```powershell
Set-Location frontend

# Install dependencies
npm install

# Setup environment
Copy-Item .env.example .env.local
# Edit .env.local with your configuration

# Start development server
npm run dev
```

### 4. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🏗️ Architecture

### Tech Stack
- **Frontend**: Next.js 14, TypeScript, TailwindCSS, shadcn/ui
- **Backend**: FastAPI, Python, Pydantic
- **ML Pipeline**: ONNX Runtime, OpenCV, NumPy
- **Models**: Ensemble of MesoNet + Xception with Grad-CAM

### Project Structure
```
Truthlens/
├── frontend/                 # Next.js frontend
│   ├── src/
│   │   ├── app/             # App Router pages
│   │   ├── components/      # React components
│   │   └── lib/             # Utilities & services
│   ├── .env.example         # Environment template
│   └── package.json
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── routers/         # API endpoints
│   │   ├── services/        # Business logic
│   │   └── models/          # Pydantic models
│   ├── models/              # ML model files (.onnx)
│   ├── .env.example         # Environment template
│   └── requirements.txt
└── README.md               # This file
```

## 🔧 Configuration

### Frontend Environment (.env.local)
```env
# Backend API Configuration
NEXT_PUBLIC_API_BASE=http://localhost:8000

# File Upload Settings
NEXT_PUBLIC_MAX_FILE_SIZE=104857600
NEXT_PUBLIC_ACCEPTED_FORMATS=video/mp4,video/avi,image/jpeg,image/png

# RealityDefender Backup API (Optional)
NEXT_PUBLIC_REALITYDEFENDER_API_KEY=your-api-key-here
NEXT_PUBLIC_REALITYDEFENDER_ENABLED=true
```

### Backend Environment (.env)
```env
# Model Configuration
MESONET_MODEL_PATH=./models/mesonet4.onnx
XCEPTION_MODEL_PATH=./models/xception.onnx
MESONET_WEIGHT=0.4
XCEPTION_WEIGHT=0.6
CONFIDENCE_THRESHOLD=0.7

# Processing Settings
MAX_FRAMES_TO_PROCESS=30
ENABLE_CACHING=true

# Development Settings
DEBUG=true
RELOAD=true
LOG_LEVEL=INFO
```

## 🛡️ Backup API Integration

### RealityDefender Setup

Truthlens includes RealityDefender as a backup detection service for enhanced reliability.

#### 1. Install Package
```powershell
# Already included in package.json
npm install @realitydefender/realitydefender
```

#### 2. Configure API Key
Add your RealityDefender API key to `frontend/.env.local`:
```env
NEXT_PUBLIC_REALITYDEFENDER_API_KEY=your-actual-api-key
NEXT_PUBLIC_REALITYDEFENDER_ENABLED=true
```

#### 3. How It Works
- **Primary**: Truthlens ensemble models (MesoNet + Xception)
- **Fallback**: RealityDefender API automatically triggers if primary fails
- **Seamless**: Users see unified results regardless of which service processes the file

#### 4. Service Features
- Automatic fallback on primary service failure
- Unified result format across both services
- Configurable timeout and error handling
- Support for images and videos

## 📊 API Endpoints

### Core Analysis
- `POST /api/analyze` - Analyze uploaded file or URL
- `POST /api/analyze/batch` - Batch analysis of multiple files
- `GET /health` - Service health check

### Request Format
```bash
# File upload
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@video.mp4"

# URL analysis  
curl -X POST "http://localhost:8000/api/analyze" \
  -F "url=https://example.com/video.mp4"
```

### Response Format
```json
{
  "verdict": "fake|real|unknown",
  "confidence": 0.85,
  "input_type": "video|image",
  "input": "filename.mp4",
  "latency_ms": 2500,
  "signals": [
    {
      "name": "Temporal Inconsistency",
      "score": 0.92,
      "description": "Detected temporal artifacts"
    }
  ],
  "metadata": {
    "processed_frames": 15,
    "total_frames": 150,
    "resolution": "1920x1080",
    "model_version": "ensemble-v1.0"
  },
  "heatmap": "base64_encoded_image"
}
```

## 🧪 Testing

### Frontend Tests
```powershell
Set-Location frontend
npm run test
npm run test:e2e
npm run lint
```

### Backend Tests
```powershell
Set-Location backend
pytest tests/ -v
pytest tests/ --cov=app
black app/ --check
flake8 app/
```

## 🚀 Deployment

### Frontend (Vercel)
```powershell
# Build and deploy
npm run build
vercel --prod
```

### Backend (Railway/Render)
```powershell
# Production server
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker (Optional)
```powershell
# Build and run with Docker Compose
docker-compose up --build
```

## 🔍 Features

### Core Capabilities
- **Multi-format Support**: Images (JPEG, PNG, GIF, WebP) and Videos (MP4, AVI, MOV, WebM)
- **Ensemble Detection**: Combines MesoNet and Xception models for higher accuracy
- **Explainable AI**: Grad-CAM heatmaps show detection focus areas
- **Real-time Processing**: Optimized for fast inference with frame sampling
- **Backup Service**: RealityDefender API fallback for enhanced reliability

### Advanced Features
- **Batch Processing**: Analyze multiple files simultaneously
- **URL Analysis**: Direct analysis from web URLs
- **Caching System**: Avoid reprocessing identical files
- **Progress Tracking**: Real-time upload and processing progress
- **Result Sharing**: Export and share analysis results

## 🛠️ Development

### Adding New Models
1. Convert model to ONNX format
2. Add model service in `backend/app/services/models/`
3. Update ensemble detector configuration
4. Test with sample data

### Customizing UI
- Components use shadcn/ui and TailwindCSS
- Follow atomic design pattern (atoms/molecules/organisms)
- Maintain TypeScript strict mode

### Performance Optimization
- Models are cached in memory after first load
- Video processing uses frame sampling (every 10th frame)
- Results are cached to avoid reprocessing
- Frontend uses React Query for API state management

## 📝 License

This project is built for hackathon purposes. Please ensure compliance with all model licenses and API terms of service.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'feat: add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📞 Support

For issues and questions:
1. Check the [API documentation](http://localhost:8000/docs)
2. Review environment configuration
3. Check logs for detailed error messages
4. Ensure all dependencies are installed correctly

---

**Built with ❤️ for AI-powered media verification**