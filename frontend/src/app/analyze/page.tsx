'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Header } from "@/components/layout/header";
import { FileUpload } from "@/components/FileUpload";
import { VideoPreview } from "@/components/VideoPreview";
import { ResultsDisplay, type AnalysisResult } from "@/components/ResultsDisplay";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { useToast } from "@/hooks/use-toast";
import { RealityDefenderService } from "@/lib/realitydefender";

// Temporary: analysis is open to all users (no auth)
const useAuth = () => {
  return { isAuthenticated: true, isLoading: false } as const;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

// Initialize RealityDefender backup service
// Initialize RealityDefender backup service
const realityDefenderService = RealityDefenderService.getInstance();

// Transform the legacy Analysis type to the new AnalysisResult type
const transformAnalysisResult = (legacyAnalysis: any): AnalysisResult => {
  return {
    verdict: legacyAnalysis.verdict === 'real' ? 'real' : 
             legacyAnalysis.verdict === 'fake' ? 'fake' : 'unknown',
    confidence: legacyAnalysis.confidence || 0,
    inputType: legacyAnalysis.inputType === 'image' ? 'image' : 
               legacyAnalysis.inputType === 'video' ? 'video' : 'url',
    input: legacyAnalysis.input || '',
    latencyMs: legacyAnalysis.latencyMs || 0,
    signals: legacyAnalysis.signals?.map((signal: any) => ({
      name: signal.name,
      score: signal.score,
      description: signal.description
    })) || [],
    frameAnalysis: legacyAnalysis.frameAnalysis || legacyAnalysis.frame_analysis,
    metadata: legacyAnalysis.metadata,
    heatmap: legacyAnalysis.heatmap  // Add heatmap from backend
  };
};

export default function AnalyzePage() {
  const { isAuthenticated, isLoading } = useAuth();
  const [showAuthDialog, setShowAuthDialog] = useState(false);
  const router = useRouter();
  const { toast } = useToast();

  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [currentUrl, setCurrentUrl] = useState<string>('');

  // Auth gating disabled; ensure dialog never opens
  useEffect(() => {
    setShowAuthDialog(false);
  }, []);

  const analyzeUrl = async (url: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setCurrentUrl(url);
    setSelectedFile(null);
    setUploadProgress(0);

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `API error ${res.status}`);
      }
      
      const data = await res.json();
      // Backend returns the result directly, not wrapped
      const transformedResult = transformAnalysisResult(data);
      setResult(transformedResult);
      
      toast({ 
        title: "Analysis complete", 
        description: "URL analyzed successfully." 
      });
    } catch (e: any) {
      const msg = e?.message ?? "Failed to analyze URL";
      setError(msg);
      toast({ 
        title: "Analysis failed", 
        description: msg, 
        variant: "destructive" 
      });
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const analyzeFile = async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedFile(file);
    setCurrentUrl('');
    setUploadProgress(0);

    try {
      // Try RealityDefender first (primary detection)
      const isRealityDefenderAvailable = await realityDefenderService.isAvailable();
      if (isRealityDefenderAvailable) {
        try {
          console.log('🔍 Using RealityDefender as primary detection...');
          setUploadProgress(50); // Simulate progress for user feedback
          
          const realityDefenderResult = await realityDefenderService.detect(file);
          
          // Transform RealityDefender SDK result to our format
          const transformedResult: AnalysisResult = {
            verdict: realityDefenderResult.verdict as 'real' | 'fake' | 'unknown',
            confidence: realityDefenderResult.confidence,
            inputType: file.type.startsWith('video/') ? 'video' : 'image',
            input: file.name,
            latencyMs: Math.round((realityDefenderResult.processing_time || 0) * 1000),
            signals: realityDefenderResult.signals || [{
              name: "RealityDefender SDK",
              score: realityDefenderResult.confidence,
              description: "Official Reality Defender deepfake detection service"
            }],
            frameAnalysis: undefined,
            metadata: {
              processedFrames: 1,
              totalFrames: 1,
              resolution: "unknown",
              source: "RealityDefender SDK",
              status: realityDefenderResult.metadata?.status || "completed",
              models: realityDefenderResult.metadata?.models || []
            },
            heatmap: undefined
          };

          setResult(transformedResult);
          
          toast({ 
            title: "Analysis complete", 
            description: `${file.name} analyzed with RealityDefender.` 
          });
          
          return; // Exit successfully with primary result
        } catch (realityDefenderError: any) {
          console.warn("RealityDefender failed, falling back to backend:", realityDefenderError.message);
          
          toast({
            title: "Switching to backup",
            description: "Primary service failed, using backend models...",
          });
        }
      }

      // Fallback to backend analysis
      const form = new FormData();
      form.append("file", file);

      const xhr = new XMLHttpRequest();
      const url = `${API_BASE}/api/analyze`;

      const resPromise = new Promise<any>((resolve, reject) => {
        xhr.upload.onprogress = (evt) => {
          if (evt.lengthComputable) {
            const percent = Math.round((evt.loaded / evt.total) * 100);
            setUploadProgress(percent);
          }
        };
        
        xhr.onreadystatechange = () => {
          if (xhr.readyState === 4) {
            if (xhr.status >= 200 && xhr.status < 300) {
              try {
                const json = JSON.parse(xhr.responseText);
                resolve(json);
              } catch (err) {
                reject(new Error('Invalid response format'));
              }
            } else {
              try {
                const errorData = JSON.parse(xhr.responseText);
                reject(new Error(errorData.detail || `API error ${xhr.status}`));
              } catch {
                reject(new Error(`API error ${xhr.status}`));
              }
            }
          }
        };
        
        xhr.onerror = () => reject(new Error("Network error"));
        xhr.open("POST", url, true);
        xhr.send(form);
      });

      const data = await resPromise;
      // Backend returns the result directly, not wrapped
      const transformedResult = transformAnalysisResult(data);
      setResult(transformedResult);
      
      toast({ 
        title: "Analysis complete", 
        description: `${file.name} analyzed with backend models.` 
      });

    } catch (e: any) {
      const msg = e?.message ?? "Failed to analyze file";
      setError(msg);
      toast({ 
        title: "Analysis failed", 
        description: msg, 
        variant: "destructive" 
      });
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const handleRetry = () => {
    if (selectedFile) {
      analyzeFile(selectedFile);
    } else if (currentUrl) {
      analyzeUrl(currentUrl);
    }
  };

  const handleShare = () => {
    if (result) {
      const shareData = {
        title: 'TruthLens Analysis Result',
        text: `Analysis: ${result.verdict} (${Math.round(result.confidence * 100)}% confidence)`,
        url: window.location.href
      };

      if (navigator.share) {
        navigator.share(shareData).catch(console.error);
      } else {
        // Fallback to copying to clipboard
        navigator.clipboard.writeText(
          `${shareData.title}\n${shareData.text}\n${shareData.url}`
        ).then(() => {
          toast({
            title: "Copied to clipboard",
            description: "Analysis result copied to clipboard"
          });
        });
      }
    }
  };

  const handleDownloadReport = () => {
    if (result) {
      const report = {
        timestamp: new Date().toISOString(),
        verdict: result.verdict,
        confidence: result.confidence,
        inputType: result.inputType,
        input: result.input,
        processingTime: result.latencyMs,
        signals: result.signals,
        metadata: result.metadata
      };

      const blob = new Blob([JSON.stringify(report, null, 2)], {
        type: 'application/json'
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `truthlens-analysis-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast({
        title: "Report downloaded",
        description: "Analysis report saved to your device"
      });
    }
  };

  const handleFrameCapture = (timestamp: number) => {
    toast({
      title: "Frame captured",
      description: `Frame at ${timestamp.toFixed(1)}s captured for analysis`
    });
    // Additional frame capture logic can be added here
  };

  if (isLoading) {
    return (
      <div className="flex flex-col min-h-screen bg-background">
        <Header />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-4">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
            <p>Loading...</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <>
      <div className="flex flex-col min-h-screen bg-background">
        <Header />
        <main className="flex-1 pt-20">
          <div className="container mx-auto px-4 py-8">
            <div className="max-w-7xl mx-auto">
              {/* Page Header */}
              <div className="text-center mb-8">
                <h1 className="text-3xl font-bold tracking-tight mb-2">
                  AI-Powered Deepfake Detection
                </h1>
                <p className="text-muted-foreground max-w-2xl mx-auto">
                  Upload media files or provide URLs to detect AI-generated content with advanced machine learning algorithms.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left Column - Upload and Preview */}
                <div className="space-y-6">
                  <FileUpload
                    onFileSelect={analyzeFile}
                    onUrlSubmit={analyzeUrl}
                    loading={loading}
                    progress={uploadProgress}
                    error={error}
                    maxSize={100} // 100MB for hackathon demo
                  />

                  {(selectedFile || currentUrl) && (
                    <VideoPreview
                      file={selectedFile}
                      url={currentUrl}
                      showControls={true}
                      showMetadata={true}
                      onFrameCapture={handleFrameCapture}
                      analysisRegions={result?.frameAnalysis?.[0]?.regions}
                    />
                  )}
                </div>

                {/* Right Column - Results */}
                <div>
                  <ResultsDisplay
                    result={result}
                    loading={loading}
                    error={error}
                    onRetry={handleRetry}
                    onShare={handleShare}
                    onDownloadReport={handleDownloadReport}
                  />
                </div>
              </div>

              {/* Educational Section */}
              {!loading && !result && (
                <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
                  <div className="space-y-4">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto">
                      <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                    </div>
                    <h3 className="font-semibold">Upload Media</h3>
                    <p className="text-sm text-muted-foreground">
                      Upload images or videos, or provide URLs for analysis
                    </p>
                  </div>

                  <div className="space-y-4">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto">
                      <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                    </div>
                    <h3 className="font-semibold">AI Analysis</h3>
                    <p className="text-sm text-muted-foreground">
                      Advanced algorithms analyze patterns and detect manipulation
                    </p>
                  </div>

                  <div className="space-y-4">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto">
                      <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <h3 className="font-semibold">Get Results</h3>
                    <p className="text-sm text-muted-foreground">
                      Receive confidence scores and detailed analysis reports
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>

      {/* Auth dialog disabled (analysis is free) */}
      <AlertDialog open={false}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle></AlertDialogTitle>
            <AlertDialogDescription></AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter></AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}