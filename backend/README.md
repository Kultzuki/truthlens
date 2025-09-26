# Truthlens Backend - Deepfake Detection API

FastAPI backend for the Truthlens deepfake detection system.

## Quick Start

### 1. Install Dependencies

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```powershell
# Copy environment template
Copy-Item .env.example .env

# Edit .env file with your settings
notepad .env
```

### 3. Run Development Server

```powershell
# Start the server
uvicorn app.main:app --reload --port 8000

# Or use the development script
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Core Endpoints

- `POST /api/analyze` - Analyze media file for deepfakes
- `POST /api/analyze/batch` - Batch analysis of multiple files
- `GET /api/analyze/formats` - Get supported file formats
- `GET /health` - Health check endpoint

### Analysis Request

```bash
# File upload
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/video.mp4"

# URL analysis
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "url=https://example.com/video.mp4"
```

### Response Format

```json
{
  "verdict": "fake",
  "confidence": 0.85,
  "input_type": "video",
  "input": "video.mp4",
  "latency_ms": 2500,
  "signals": [
    {
      "name": "Face Authenticity",
      "confidence": 0.85,
      "description": "Overall face authenticity assessment"
    }
  ],
  "frame_analysis": [
    {
      "frame_number": 0,
      "timestamp_ms": 0,
      "confidence": 0.85,
      "verdict": "fake",
      "regions": [
        {
          "x": 50,
          "y": 50,
          "width": 100,
          "height": 100,
          "confidence": 0.85,
          "label": "face"
        }
      ]
    }
  ],
  "metadata": {
    "frames_analyzed": 30,
    "processing_time_ms": 2500,
    "model_version": "xception_v1.0",
    "resolution": "1920x1080"
  },
  "heatmap": "data:image/png;base64,..."
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models/xception_deepfake.onnx` | Path to ONNX model file |
| `CONFIDENCE_THRESHOLD` | `0.7` | Threshold for fake detection |
| `MAX_FRAMES_TO_PROCESS` | `30` | Maximum frames to analyze per video |
| `MAX_FILE_SIZE_MB` | `100` | Maximum upload file size |
| `API_PORT` | `8000` | Server port |
| `DEBUG` | `true` | Enable debug mode |

### Supported File Formats

**Videos**: `.mp4`, `.avi`, `.mov`, `.webm`, `.mkv`
**Images**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`

## Development

### Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   └── analysis.py      # Pydantic models
│   ├── routers/
│   │   └── analyze.py       # API endpoints
│   └── services/
│       ├── deepfake_detector.py  # ML inference
│       └── file_handler.py       # File management
├── tests/                   # Test files
├── uploads/                 # Temporary uploads
├── models/                  # ML model files
├── requirements.txt         # Dependencies
└── .env                     # Configuration
```

### Running Tests

```powershell
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Code Quality

```powershell
# Format code
black app/ --line-length 100

# Lint code
flake8 app/ --max-line-length=100

# Sort imports
isort app/
```

## Deployment

### Production Setup

1. **Update Environment**:
   ```bash
   DEBUG=false
   RELOAD=false
   API_WORKERS=4
   LOG_LEVEL=WARNING
   ```

2. **Install Production Server**:
   ```bash
   pip install gunicorn
   ```

3. **Run Production Server**:
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Railway/Render Deployment

**Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
**Build Command**: `pip install -r requirements.txt`

## Model Integration

### Adding ONNX Model

1. Place your ONNX model in the `models/` directory
2. Update `MODEL_PATH` in `.env`
3. Ensure model input shape is (1, 3, 224, 224) for images

### Model Requirements

- **Input**: RGB images, normalized to [0, 1]
- **Output**: Binary classification probabilities [real, fake]
- **Format**: ONNX for optimal inference speed

## Troubleshooting

### Common Issues

1. **CORS Errors**: Check `CORS_ORIGINS` in `.env`
2. **File Upload Fails**: Verify `MAX_FILE_SIZE_MB` setting
3. **Model Not Found**: Ensure `MODEL_PATH` points to valid ONNX file
4. **Memory Issues**: Reduce `MAX_FRAMES_TO_PROCESS`

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

## Performance Optimization

- Use GPU inference by installing `onnxruntime-gpu`
- Enable Redis caching for repeated analyses
- Adjust `MAX_FRAMES_TO_PROCESS` based on accuracy vs speed needs
- Use batch processing for multiple files

## Security Notes

- Never commit `.env` files to version control
- Use strong `SECRET_KEY` in production
- Implement rate limiting for production deployments
- Validate all file uploads and URLs