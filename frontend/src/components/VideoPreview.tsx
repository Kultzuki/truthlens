'use client';

import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { 
  Play, 
  Pause, 
  Volume2, 
  VolumeX, 
  Maximize2, 
  RotateCcw,
  Download,
  Eye,
  Clock,
  FileVideo,
  Image as ImageIcon
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface VideoPreviewProps {
  file: File | null;
  url?: string;
  className?: string;
  showControls?: boolean;
  showMetadata?: boolean;
  onFrameCapture?: (timestamp: number) => void;
  analysisRegions?: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
    timestamp?: number;
  }>;
}

export function VideoPreview({ 
  file, 
  url,
  className,
  showControls = true,
  showMetadata = true,
  onFrameCapture,
  analysisRegions = []
}: VideoPreviewProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<{
    size: number;
    type: string;
    dimensions?: { width: number; height: number };
    duration?: number;
  } | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Create preview URL when file changes
  useEffect(() => {
    if (file) {
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);
      setMetadata({
        size: file.size,
        type: file.type
      });

      return () => {
        URL.revokeObjectURL(objectUrl);
      };
    } else if (url) {
      setPreviewUrl(url);
    }
  }, [file, url]);

  // Video event handlers
  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
      setMetadata(prev => prev ? {
        ...prev,
        dimensions: {
          width: videoRef.current!.videoWidth,
          height: videoRef.current!.videoHeight
        },
        duration: videoRef.current!.duration
      } : null);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSeek = (value: number[]) => {
    if (videoRef.current) {
      videoRef.current.currentTime = value[0];
      setCurrentTime(value[0]);
    }
  };

  const handleVolumeChange = (value: number[]) => {
    const newVolume = value[0];
    setVolume(newVolume);
    if (videoRef.current) {
      videoRef.current.volume = newVolume;
    }
    setIsMuted(newVolume === 0);
  };

  const toggleMute = () => {
    if (videoRef.current) {
      if (isMuted) {
        videoRef.current.volume = volume;
        setIsMuted(false);
      } else {
        videoRef.current.volume = 0;
        setIsMuted(true);
      }
    }
  };

  const handleFullscreen = () => {
    if (containerRef.current) {
      if (!document.fullscreenElement) {
        containerRef.current.requestFullscreen();
        setIsFullscreen(true);
      } else {
        document.exitFullscreen();
        setIsFullscreen(false);
      }
    }
  };

  const captureCurrentFrame = () => {
    if (videoRef.current && canvasRef.current && onFrameCapture) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      if (ctx) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0);
        
        onFrameCapture(currentTime);
      }
    }
  };

  const downloadFrame = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      if (ctx) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0);
        
        canvas.toBlob((blob) => {
          if (blob) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `frame-${Math.round(currentTime)}s.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
          }
        });
      }
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const isVideo = file?.type.startsWith('video/') || url?.includes('.mp4') || url?.includes('.avi') || url?.includes('.mov') || url?.includes('.webm');

  if (!previewUrl) {
    return (
      <Card className={cn("w-full", className)}>
        <CardContent className="flex items-center justify-center h-64 text-muted-foreground">
          <div className="text-center space-y-2">
            {isVideo ? <FileVideo className="w-12 h-12 mx-auto" /> : <ImageIcon className="w-12 h-12 mx-auto" />}
            <p>No media selected</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn("w-full", className)}>
      {showMetadata && metadata && (
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center justify-between">
            <span>Preview</span>
            <div className="flex space-x-2">
              <Badge variant="secondary">
                {isVideo ? 'Video' : 'Image'}
              </Badge>
              <Badge variant="outline">
                {formatFileSize(metadata.size)}
              </Badge>
            </div>
          </CardTitle>
          {metadata.dimensions && (
            <div className="flex space-x-4 text-sm text-muted-foreground">
              <span>{metadata.dimensions.width} × {metadata.dimensions.height}</span>
              {metadata.duration && (
                <span className="flex items-center space-x-1">
                  <Clock className="w-3 h-3" />
                  <span>{formatTime(metadata.duration)}</span>
                </span>
              )}
            </div>
          )}
        </CardHeader>
      )}
      
      <CardContent className="p-4">
        <div ref={containerRef} className="relative bg-black rounded-lg overflow-hidden">
          {isVideo ? (
            <video
              ref={videoRef}
              src={previewUrl}
              className="w-full h-auto max-h-96 object-contain"
              onLoadedMetadata={handleLoadedMetadata}
              onTimeUpdate={handleTimeUpdate}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              preload="metadata"
            />
          ) : (
            <img
              src={previewUrl}
              alt="Preview"
              className="w-full h-auto max-h-96 object-contain"
            />
          )}

          {/* Analysis Regions Overlay */}
          {analysisRegions.length > 0 && (
            <div className="absolute inset-0 pointer-events-none">
              {analysisRegions.map((region, index) => (
                <div
                  key={index}
                  className="absolute border-2 border-red-500 bg-red-500/20"
                  style={{
                    left: `${region.x}%`,
                    top: `${region.y}%`,
                    width: `${region.width}%`,
                    height: `${region.height}%`,
                  }}
                >
                  <div className="absolute -top-6 left-0 bg-red-500 text-white text-xs px-1 rounded">
                    {Math.round(region.confidence * 100)}%
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Video Controls */}
          {isVideo && showControls && (
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
              {/* Progress Bar */}
              <div className="mb-3">
                <Slider
                  value={[currentTime]}
                  max={duration}
                  step={0.1}
                  onValueChange={handleSeek}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-white mt-1">
                  <span>{formatTime(currentTime)}</span>
                  <span>{formatTime(duration)}</span>
                </div>
              </div>

              {/* Control Buttons */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handlePlayPause}
                    className="text-white hover:bg-white/20"
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </Button>

                  <div className="flex items-center space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={toggleMute}
                      className="text-white hover:bg-white/20"
                    >
                      {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                    </Button>
                    <Slider
                      value={[isMuted ? 0 : volume]}
                      max={1}
                      step={0.1}
                      onValueChange={handleVolumeChange}
                      className="w-20"
                    />
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  {onFrameCapture && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={captureCurrentFrame}
                      className="text-white hover:bg-white/20"
                      title="Capture Frame for Analysis"
                    >
                      <Eye className="w-4 h-4" />
                    </Button>
                  )}

                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={downloadFrame}
                    className="text-white hover:bg-white/20"
                    title="Download Current Frame"
                  >
                    <Download className="w-4 h-4" />
                  </Button>

                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleFullscreen}
                    className="text-white hover:bg-white/20"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} className="hidden" />
      </CardContent>
    </Card>
  );
}
