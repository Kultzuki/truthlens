"""
Truthlens Backend - FastAPI Application
AI-Powered Deepfake Detection System
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import os
from contextlib import asynccontextmanager

from app.routers import analyze


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 Truthlens Backend starting up...")
    print("🤖 Loading AI models...")
    # TODO: Initialize ML models here
    yield
    # Shutdown
    print("🛑 Truthlens Backend shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Truthlens API",
    description="AI-Powered Deepfake Detection System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
        "https://truthlens.vercel.app",  # Production frontend (update as needed)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted Host Middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure properly for production
)

# Include routers
app.include_router(analyze.router, prefix="/api", tags=["analysis"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Truthlens API - AI-Powered Deepfake Detection",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "truthlens-backend",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    # Development server configuration
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )