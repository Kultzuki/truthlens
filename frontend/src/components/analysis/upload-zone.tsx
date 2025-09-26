import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { UploadCloud, CheckCircle, AlertCircle, FileText } from "lucide-react";
import { useRef, useState, useCallback } from "react";
import { Progress } from "@/components/ui/progress";

type Props = {
  onAnalyzeUrl?: (url: string) => void;
  onAnalyzeFile?: (file: File, onProgress?: (p: number) => void) => void;
  loading?: boolean;
  uploadedFile?: File | null;
  uploadStatus?: 'idle' | 'uploading' | 'uploaded' | 'error';
};

export function UploadZone({ onAnalyzeUrl, onAnalyzeFile, loading, uploadedFile, uploadStatus = 'idle' }: Props) {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [url, setUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleFiles = useCallback((files: FileList | null) => {
    const f = files?.[0];
    if (f && onAnalyzeFile) {
      setProgress(0); // Reset progress
      onAnalyzeFile(f, setProgress);
    }
  }, [onAnalyzeFile]);

  return (
    <Card className="glassmorphic">
      <CardHeader>
        <CardTitle>Analyze Media</CardTitle>
        <CardDescription>Upload a file or paste a URL to scan for deepfakes.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div
          className={`border-2 border-dashed rounded-lg p-12 flex flex-col items-center justify-center text-center transition-all duration-300 ${
            isDragging 
              ? 'border-primary/60 bg-primary/5 scale-105' 
              : uploadStatus === 'uploaded'
              ? 'border-green-500/60 bg-green-500/5'
              : uploadStatus === 'error'
              ? 'border-red-500/60 bg-red-500/5'
              : 'border-border hover:border-primary/40'
          }`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
        >
          {uploadStatus === 'uploaded' ? (
            <div className="flex flex-col items-center space-y-4 upload-success">
              <CheckCircle className="w-12 h-12 text-green-500 bounce-in pulse-success" />
              <div className="space-y-2 slide-up-fade">
                <p className="text-green-600 font-medium">File uploaded successfully!</p>
                <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                  <FileText className="w-4 h-4" />
                  <span className="truncate max-w-xs">{uploadedFile?.name}</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  {(uploadedFile?.size && (uploadedFile.size / 1024 / 1024).toFixed(2))} MB
                </p>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => fileRef.current?.click()} 
                  className="mt-2 transition-all duration-200 hover:scale-105"
                >
                  Upload New File
                </Button>
              </div>
            </div>
          ) : uploadStatus === 'error' ? (
            <div className="flex flex-col items-center space-y-4 shake-error">
              <AlertCircle className="w-12 h-12 text-red-500 bounce-in" />
              <div className="slide-up-fade">
                <p className="text-red-600 font-medium">Upload failed</p>
                <p className="text-sm text-muted-foreground">Please try again</p>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => fileRef.current?.click()} 
                  className="mt-2 transition-all duration-200 hover:scale-105"
                >
                  Try Again
                </Button>
              </div>
            </div>
          ) : uploadStatus === 'uploading' ? (
            <div className="flex flex-col items-center space-y-4 slide-up-fade">
              <div className="relative">
                <UploadCloud className="w-12 h-12 text-primary animate-pulse" />
                <div className="absolute inset-0 rounded-full border-2 border-primary border-t-transparent animate-spin"></div>
              </div>
              <div className="space-y-2 w-full max-w-md">
                <Progress value={progress} className="progress-fill" />
                <p className="text-sm text-muted-foreground slide-up-fade">
                  Uploading… {progress}%
                </p>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center space-y-4">
              <UploadCloud className={`w-12 h-12 text-muted-foreground transition-transform duration-200 ${isDragging ? 'scale-110' : ''}`} />
              <p className="text-muted-foreground">Drag & drop files here or click to browse</p>
              <div className="mt-4 flex items-center gap-2">
                <input
                  ref={fileRef}
                  type="file"
                  className="hidden"
                  onChange={(e) => handleFiles(e.target.files)}
                />
                <Button 
                  variant="outline" 
                  onClick={() => fileRef.current?.click()} 
                  disabled={loading}
                  className="transition-all duration-200 hover:scale-105"
                >
                  Browse Files
                </Button>
              </div>
            </div>
          )}
        </div>
        <div className="flex items-center space-x-2">
            <div className="flex-grow border-t border-border"></div>
            <span className="text-muted-foreground text-xs">OR</span>
            <div className="flex-grow border-t border-border"></div>
        </div>
        <div className="space-y-2">
            <Input
              placeholder="Paste image or video URL"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="transition-all duration-200 focus:scale-[1.02]"
            />
            <Button 
              className="w-full transition-all duration-200 hover:scale-[1.02] active:scale-95" 
              onClick={() => url && onAnalyzeUrl?.(url)} 
              disabled={loading || !url.trim()}
            >
              {loading ? (
                <div className="flex items-center space-x-2">
                  <div className="h-4 w-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
                  <span>Analyzing…</span>
                </div>
              ) : (
                "Analyze URL"
              )}
            </Button>
        </div>
      </CardContent>
    </Card>
  );
}
