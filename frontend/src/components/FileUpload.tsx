'use client';

import { useCallback, useState, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Upload, 
  FileVideo, 
  FileImage, 
  X, 
  CheckCircle, 
  AlertCircle,
  Camera,
  Link as LinkIcon 
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  onUrlSubmit: (url: string) => void;
  onWebcamCapture?: () => void;
  loading?: boolean;
  progress?: number;
  error?: string | null;
  maxSize?: number; // in MB
  acceptedTypes?: string[];
}

const defaultAcceptedTypes = [
  'image/jpeg',
  'image/png', 
  'image/gif',
  'image/webp',
  'video/mp4',
  'video/avi',
  'video/mov',
  'video/webm'
];

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export function FileUpload({ 
  onFileSelect, 
  onUrlSubmit,
  onWebcamCapture,
  loading = false,
  progress = 0,
  error = null,
  maxSize = 50, // 50MB default
  acceptedTypes = defaultAcceptedTypes
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback((file: File): string | null => {
    // Check file type
    if (!acceptedTypes.includes(file.type)) {
      return `Invalid file type. Accepted: ${acceptedTypes.map(type => type.split('/')[1]).join(', ')}`;
    }

    // Check file size
    const maxSizeBytes = maxSize * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      return `File too large. Max size: ${maxSize}MB`;
    }

    // Additional security checks
    if (file.name.includes('..') || file.name.includes('/') || file.name.includes('\\')) {
      return 'Invalid file name';
    }

    return null;
  }, [acceptedTypes, maxSize]);

  const validateUrl = useCallback((url: string): string | null => {
    try {
      const urlObj = new URL(url);
      if (!['http:', 'https:'].includes(urlObj.protocol)) {
        return 'Only HTTP and HTTPS URLs are allowed';
      }
      return null;
    } catch {
      return 'Invalid URL format';
    }
  }, []);

  const handleFileSelect = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return;

    const file = files[0];
    const validation = validateFile(file);
    
    if (validation) {
      setValidationError(validation);
      return;
    }

    setValidationError(null);
    setSelectedFile(file);
    onFileSelect(file);
  }, [validateFile, onFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    handleFileSelect(files);
  }, [handleFileSelect]);

  const handleUrlSubmit = useCallback(() => {
    if (!url.trim()) return;
    
    const validation = validateUrl(url.trim());
    if (validation) {
      setValidationError(validation);
      return;
    }

    setValidationError(null);
    onUrlSubmit(url.trim());
  }, [url, validateUrl, onUrlSubmit]);

  const clearFile = useCallback(() => {
    setSelectedFile(null);
    setValidationError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('video/')) {
      return <FileVideo className="w-8 h-8 text-blue-500" />;
    }
    return <FileImage className="w-8 h-8 text-green-500" />;
  };

  return (
    <div className="space-y-6">
      {/* File Upload Zone */}
      <Card className="relative overflow-hidden">
        <CardContent className="p-6">
          <div
            className={cn(
              "relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 cursor-pointer",
              isDragging 
                ? "border-primary bg-primary/5 scale-[1.02]" 
                : "border-border hover:border-primary/50",
              loading && "pointer-events-none opacity-50"
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={acceptedTypes.join(',')}
              onChange={(e) => handleFileSelect(e.target.files)}
              className="hidden"
            />

            {selectedFile ? (
              <div className="space-y-4">
                <div className="flex items-center justify-center space-x-3">
                  {getFileIcon(selectedFile)}
                  <div className="text-left">
                    <p className="font-medium text-sm">{selectedFile.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(selectedFile.size)}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      clearFile();
                    }}
                    className="ml-auto"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>

                {loading && (
                  <div className="space-y-2">
                    <Progress value={progress} className="w-full" />
                    <p className="text-xs text-muted-foreground">
                      {progress > 0 ? `Uploading... ${progress}%` : 'Processing...'}
                    </p>
                  </div>
                )}

                {!loading && (
                  <div className="flex items-center justify-center space-x-2 text-green-600">
                    <CheckCircle className="w-4 h-4" />
                    <span className="text-sm">Ready to analyze</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-center">
                  <Upload className={cn(
                    "w-12 h-12 text-muted-foreground transition-transform duration-200",
                    isDragging && "scale-110 text-primary"
                  )} />
                </div>
                <div className="space-y-2">
                  <p className="text-lg font-medium">Drop files here or click to browse</p>
                  <p className="text-sm text-muted-foreground">
                    Supports images and videos up to {maxSize}MB
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {acceptedTypes.map(type => type.split('/')[1]).join(', ')}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="mt-4 flex flex-wrap gap-2 justify-center">
            <Button
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="flex items-center space-x-2"
            >
              <Upload className="w-4 h-4" />
              <span>Browse Files</span>
            </Button>

            {onWebcamCapture && (
              <Button
                variant="outline"
                onClick={onWebcamCapture}
                disabled={loading}
                className="flex items-center space-x-2"
              >
                <Camera className="w-4 h-4" />
                <span>Use Camera</span>
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* URL Input */}
      <Card>
        <CardContent className="p-6">
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <LinkIcon className="w-5 h-5 text-muted-foreground" />
              <h3 className="font-medium">Or analyze from URL</h3>
            </div>
            
            <div className="flex space-x-2">
              <input
                type="url"
                placeholder="https://example.com/image.jpg"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                className="flex-1 px-3 py-2 border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                disabled={loading}
              />
              <Button
                onClick={handleUrlSubmit}
                disabled={loading || !url.trim()}
                className="px-6"
              >
                Analyze
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error Display */}
      {(error || validationError) && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {error || validationError}
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
