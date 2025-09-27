---
type: "always_apply"
---

# Deepfake Detection System Rules

## Project Context
You are helping build a deepfake detection system for a 36-hour hackathon. The system uses AI/ML to detect manipulated videos and images with explainable results through heatmaps and confidence scores.

## Tech Stack Rules

### Frontend (Next.js + TypeScript)
- Always use TypeScript with strict mode enabled
- Use Next.js 14+ with App Router structure
- Style with TailwindCSS and shadcn/ui components
- State management: Use Zustand for global state, React Query for API calls
- File naming: Use PascalCase for components, camelCase for utilities
- Components location: `src/components/` with atomic design (atoms, molecules, organisms)
- API calls: Always use try-catch with proper error handling
- Environment variables: Prefix with NEXT_PUBLIC_ for client-side vars

### Backend (FastAPI + Python)
- Python version: 3.9+ with type hints everywhere
- FastAPI structure: Routers in `app/routers/`, models in `app/models/`
- Always use async/await for endpoints
- Pydantic for request/response validation
- Error handling: Return proper HTTP status codes with detailed messages
- CORS: Configure for localhost:3000 in dev, production URL in prod
- File uploads: Limit to 100MB, accept only video/mp4, video/avi, image/jpeg, image/png
- Use uvicorn for serving with --reload in development

### ML/AI Pipeline
- Model format: ONNX for production deployment (faster inference)
- Batch processing: Process every 10th frame for videos to optimize speed
- Confidence threshold: 0.7 for fake detection
- Models to use: XceptionNet or EfficientNetB0 for primary detection
- Heatmap generation: Use Grad-CAM for explainability
- Frame extraction: Use OpenCV, resize to 224x224 for model input
- Results format: {"confidence": float, "is_fake": bool, "heatmap": base64_string, "frames_analyzed": int}

## Code Patterns

### API Endpoint Structure
```python
@router.post("/analyze")
async def analyze_media(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> AnalysisResponse:
    # Validate file type
    # Extract frames if video
    # Run inference
    # Generate heatmap
    # Return structured response
```

### Frontend API Call Pattern
```typescript
const analyzeMedia = async (file: File): Promise<AnalysisResult> => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error('Analysis failed:', error);
    throw error;
  }
};
```

### Component Structure
```typescript
interface ComponentNameProps {
  // Always define prop types
}

export const ComponentName: React.FC<ComponentNameProps> = ({ props }) => {
  // Hooks at the top
  // Event handlers
  // Render logic
  return <></>;
};
```

## Git Workflow
- Branch naming: feature/[feature-name], fix/[bug-name], chore/[task-name]
- Commit messages: Use conventional commits (feat:, fix:, docs:, style:, refactor:)
- Push every 2 hours during hackathon
- Main branch: Always deployable
- Pull before starting new work

## Deployment Commands

### Frontend Deployment (Vercel)
```bash
# Build command
npm run build

# Install command  
npm install

# Development
npm run dev
```

### Backend Deployment (Railway/Render)
```bash
# Start command
uvicorn app.main:app --host 0.0.0.0 --port $PORT

# Build command
pip install -r requirements.txt

# Development
uvicorn app.main:app --reload --port 8000
```

## Testing Commands
```bash
# Frontend tests
npm run test
npm run test:e2e

# Backend tests  
pytest tests/ -v
pytest tests/ --cov=app

# Linting
npm run lint
black app/ --check
flake8 app/
```

## Environment Variables

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_MAX_FILE_SIZE=104857600
NEXT_PUBLIC_ACCEPTED_FORMATS=video/mp4,video/avi,image/jpeg,image/png
```

### Backend (.env)
```
MODEL_PATH=./models/xception_deepfake.onnx
REDIS_URL=redis://localhost:6379
MAX_FRAMES_TO_PROCESS=30
CONFIDENCE_THRESHOLD=0.7
ENABLE_CACHING=true
```

## Quick Fixes for Common Issues

### CORS Error
Add to FastAPI main.py:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Large File Upload Timeout
Frontend: Increase timeout in fetch
```typescript
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 seconds
```

Backend: Increase limits
```python
app = FastAPI()
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
```

### Model Loading Speed
Cache model in memory:
```python
@lru_cache()
def load_model():
    return onnxruntime.InferenceSession("model.onnx")
```

## Performance Optimizations

### Frontend
- Use React.memo for expensive components
- Implement virtual scrolling for result lists
- Lazy load heavy components
- Use next/image for optimized images
- Implement debouncing for real-time features

### Backend  
- Use Redis for caching repeated analyses
- Implement request queuing for multiple uploads
- Use ThreadPoolExecutor for CPU-bound tasks
- Stream large video processing results

### ML Pipeline
- Quantize models to INT8 for faster inference
- Use batch processing when possible
- Skip similar frames (similarity > 0.95)
- Resize frames before processing
- Use ONNX Runtime for optimized inference

## Debugging Commands

### Check API Health
```bash
curl http://localhost:8000/health
```

### Test File Upload
```bash
curl -X POST -F "file=@test_video.mp4" http://localhost:8000/analyze
```

### Monitor Performance
```bash
# Frontend bundle analysis
npm run analyze

# Backend profiling
python -m cProfile -s cumulative app/main.py
```

### Check GPU Usage (if available)
```bash
nvidia-smi -l 1
```

## Hackathon-Specific Shortcuts

### Quick Mock Data Generator
```typescript
export const mockAnalysisResult = (): AnalysisResult => ({
  confidence: Math.random(),
  is_fake: Math.random() > 0.5,
  heatmap: "base64_placeholder",
  frames_analyzed: Math.floor(Math.random() * 30) + 1,
  processing_time: Math.random() * 5,
});
```

### Fast Deployment Check
```bash
# Verify all services are up
./scripts/health_check.sh

# Quick integration test
npm run test:integration
```

### Emergency Fallbacks
- If model fails: Return mock predictions with warning
- If Redis down: Continue without caching
- If upload fails: Provide sample videos for demo
- If heatmap generation fails: Show confidence only

## Code Quality Rules
- No console.log in production (use proper logging)
- All functions must have docstrings/JSDoc comments
- No hardcoded values (use constants/env vars)
- Handle all error cases explicitly
- Add loading states for all async operations
- Implement proper TypeScript types (no 'any')
- Keep functions under 50 lines
- Extract reusable logic into utilities

## Security Considerations
- Validate file types on both frontend and backend
- Sanitize filenames before processing
- Implement rate limiting (10 requests/minute in demo)
- Don't expose internal errors to users
- Use parameterized queries if database is added
- Validate and sanitize all user inputs

## Demo Preparation
- Always have 3 sample videos ready (1 real, 2 deepfakes)
- Keep browser dev tools closed during demo
- Pre-warm the model before presentation
- Clear cache and test in incognito mode
- Have offline backup running on localhost
- Record screen demo as backup video

## Project Structure
```
deepfake-detector/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── lib/
│   │   ├── hooks/
│   │   └── app/
│   ├── public/
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── routers/
│   │   ├── models/
│   │   ├── services/
│   │   └── main.py
│   ├── tests/
│   └── requirements.txt
├── models/
│   └── xception_deepfake.onnx
├── docker-compose.yml
└── README.md
```

## Remember
- Commit every 2 hours minimum
- Test after every major feature
- Deploy early and often
- Keep the demo simple but impressive
- Focus on working features over complex but broken ones
- The heatmap visualization is your unique selling point
- Always have a fallback plan for each component
