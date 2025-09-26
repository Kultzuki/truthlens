'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Shield, 
  ShieldAlert, 
  ShieldCheck,
  TrendingUp,
  Clock,
  Eye,
  AlertTriangle,
  CheckCircle,
  Info,
  Download,
  Share2,
  Zap,
  Brain,
  Target
} from 'lucide-react';
import { cn } from '@/lib/utils';

export interface AnalysisResult {
  verdict: 'real' | 'fake' | 'unknown';
  confidence: number;
  inputType: 'image' | 'video' | 'url';
  input: string;
  latencyMs: number;
  signals: Array<{
    name: string;
    score: number;
    description?: string;
  }>;
  frameAnalysis?: Array<{
    timestamp: number;
    confidence: number;
    regions: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
      confidence: number;
    }>;
  }>;
  metadata?: {
    processedFrames?: number;
    totalFrames?: number;
    resolution?: string;
    duration?: number;
  };
}

interface ResultsDisplayProps {
  result: AnalysisResult | null;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  onShare?: () => void;
  onDownloadReport?: () => void;
  className?: string;
}

const VerdictBadge = ({ verdict, confidence }: { verdict: string; confidence: number }) => {
  const getVerdictConfig = () => {
    switch (verdict) {
      case 'real':
        return {
          icon: ShieldCheck,
          label: 'Authentic',
          className: 'bg-green-500/10 text-green-700 border-green-500/20',
          glowClass: 'shadow-green-500/20'
        };
      case 'fake':
        return {
          icon: ShieldAlert,
          label: 'Deepfake Detected',
          className: 'bg-red-500/10 text-red-700 border-red-500/20',
          glowClass: 'shadow-red-500/20'
        };
      default:
        return {
          icon: Shield,
          label: 'Inconclusive',
          className: 'bg-yellow-500/10 text-yellow-700 border-yellow-500/20',
          glowClass: 'shadow-yellow-500/20'
        };
    }
  };

  const config = getVerdictConfig();
  const Icon = config.icon;

  return (
    <div className={cn(
      "inline-flex items-center space-x-2 px-4 py-2 rounded-full border-2 shadow-lg",
      config.className,
      config.glowClass,
      "animate-in fade-in-50 duration-500"
    )}>
      <Icon className="w-5 h-5" />
      <span className="font-semibold">{config.label}</span>
      <Badge variant="secondary" className="ml-2">
        {Math.round(confidence * 100)}%
      </Badge>
    </div>
  );
};

const ConfidenceMeter = ({ confidence, verdict }: { confidence: number; verdict: string }) => {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedValue(confidence * 100);
    }, 300);
    return () => clearTimeout(timer);
  }, [confidence]);

  const getColorClass = () => {
    if (verdict === 'fake') return 'bg-red-500';
    if (verdict === 'real') return 'bg-green-500';
    return 'bg-yellow-500';
  };

  const getBackgroundClass = () => {
    if (verdict === 'fake') return 'bg-red-500/20';
    if (verdict === 'real') return 'bg-green-500/20';
    return 'bg-yellow-500/20';
  };

  return (
    <div className="space-y-4">
      <div className="text-center">
        <div className="text-3xl font-bold text-foreground mb-2">
          {Math.round(animatedValue)}%
        </div>
        <div className="text-sm text-muted-foreground">Confidence Score</div>
      </div>
      
      <div className="relative">
        <div className={cn("h-4 rounded-full", getBackgroundClass())}>
          <div 
            className={cn("h-full rounded-full transition-all duration-1000 ease-out", getColorClass())}
            style={{ width: `${animatedValue}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-muted-foreground mt-1">
          <span>0%</span>
          <span>100%</span>
        </div>
      </div>
    </div>
  );
};

const SignalIndicator = ({ signal }: { signal: { name: string; score: number; description?: string } }) => {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedScore(signal.score * 100);
    }, 300);
    return () => clearTimeout(timer);
  }, [signal.score]);

  return (
    <div className="space-y-2 p-3 rounded-lg bg-card border">
      <div className="flex items-center justify-between">
        <span className="font-medium text-sm capitalize">
          {signal.name.replace(/_/g, ' ')}
        </span>
        <Badge variant="outline">
          {Math.round(animatedScore)}%
        </Badge>
      </div>
      
      <div className="relative">
        <div className="h-2 bg-muted rounded-full">
          <div 
            className="h-full bg-primary rounded-full transition-all duration-1000 ease-out"
            style={{ width: `${animatedScore}%` }}
          />
        </div>
      </div>
      
      {signal.description && (
        <p className="text-xs text-muted-foreground">
          {signal.description}
        </p>
      )}
    </div>
  );
};

export function ResultsDisplay({ 
  result, 
  loading = false, 
  error = null, 
  onRetry,
  onShare,
  onDownloadReport,
  className 
}: ResultsDisplayProps) {
  if (loading) {
    return (
      <Card className={cn("w-full", className)}>
        <CardContent className="flex flex-col items-center justify-center h-80 space-y-4">
          <div className="relative">
            <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
            <Brain className="absolute inset-0 m-auto w-6 h-6 text-primary animate-pulse" />
          </div>
          <div className="text-center space-y-2">
            <h3 className="font-semibold">Analyzing Media...</h3>
            <p className="text-sm text-muted-foreground">
              AI is processing frames and detecting patterns
            </p>
          </div>
          <div className="w-full max-w-xs">
            <Progress value={undefined} className="h-2" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={cn("w-full", className)}>
        <CardContent className="flex flex-col items-center justify-center h-80 space-y-4">
          <AlertTriangle className="w-12 h-12 text-destructive" />
          <div className="text-center space-y-2">
            <h3 className="font-semibold text-destructive">Analysis Failed</h3>
            <p className="text-sm text-muted-foreground max-w-md">
              {error}
            </p>
          </div>
          {onRetry && (
            <Button onClick={onRetry} variant="outline">
              Try Again
            </Button>
          )}
        </CardContent>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card className={cn("w-full", className)}>
        <CardContent className="flex flex-col items-center justify-center h-80 space-y-4 text-muted-foreground">
          <Target className="w-12 h-12" />
          <div className="text-center space-y-2">
            <h3 className="font-medium">Ready for Analysis</h3>
            <p className="text-sm">
              Upload a file or provide a URL to start deepfake detection
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <Eye className="w-5 h-5" />
            <span>Analysis Results</span>
          </CardTitle>
          <div className="flex space-x-2">
            {onShare && (
              <Button variant="outline" size="sm" onClick={onShare}>
                <Share2 className="w-4 h-4 mr-2" />
                Share
              </Button>
            )}
            {onDownloadReport && (
              <Button variant="outline" size="sm" onClick={onDownloadReport}>
                <Download className="w-4 h-4 mr-2" />
                Report
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Main Verdict */}
        <div className="text-center space-y-4">
          <VerdictBadge verdict={result.verdict} confidence={result.confidence} />
          <ConfidenceMeter confidence={result.confidence} verdict={result.verdict} />
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 rounded-lg bg-card border">
            <div className="flex items-center justify-center mb-2">
              <Clock className="w-4 h-4 text-muted-foreground" />
            </div>
            <div className="font-semibold">{result.latencyMs}ms</div>
            <div className="text-xs text-muted-foreground">Processing Time</div>
          </div>

          <div className="text-center p-3 rounded-lg bg-card border">
            <div className="flex items-center justify-center mb-2">
              <Zap className="w-4 h-4 text-muted-foreground" />
            </div>
            <div className="font-semibold capitalize">{result.inputType}</div>
            <div className="text-xs text-muted-foreground">Input Type</div>
          </div>

          {result.metadata?.processedFrames && (
            <div className="text-center p-3 rounded-lg bg-card border">
              <div className="flex items-center justify-center mb-2">
                <TrendingUp className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="font-semibold">{result.metadata.processedFrames}</div>
              <div className="text-xs text-muted-foreground">Frames Analyzed</div>
            </div>
          )}

          {result.metadata?.resolution && (
            <div className="text-center p-3 rounded-lg bg-card border">
              <div className="flex items-center justify-center mb-2">
                <Eye className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="font-semibold">{result.metadata.resolution}</div>
              <div className="text-xs text-muted-foreground">Resolution</div>
            </div>
          )}
        </div>

        {/* Detailed Analysis */}
        <Tabs defaultValue="signals" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="signals">Detection Signals</TabsTrigger>
            <TabsTrigger value="timeline" disabled={!result.frameAnalysis}>
              Frame Timeline
            </TabsTrigger>
          </TabsList>

          <TabsContent value="signals" className="space-y-4">
            <div className="grid gap-3">
              {result.signals.map((signal, index) => (
                <SignalIndicator key={index} signal={signal} />
              ))}
            </div>

            {result.signals.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <Info className="w-8 h-8 mx-auto mb-2" />
                <p>No detailed signals available</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="timeline" className="space-y-4">
            {result.frameAnalysis ? (
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {result.frameAnalysis.map((frame, index) => (
                  <div key={index} className="flex items-center justify-between p-2 rounded bg-card border">
                    <div className="flex items-center space-x-3">
                      <Badge variant="outline">
                        {frame.timestamp.toFixed(1)}s
                      </Badge>
                      <div className="text-sm">
                        {frame.regions.length} region(s) detected
                      </div>
                    </div>
                    <Badge 
                      variant={frame.confidence > 0.7 ? "destructive" : frame.confidence > 0.3 ? "secondary" : "default"}
                    >
                      {Math.round(frame.confidence * 100)}%
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Clock className="w-8 h-8 mx-auto mb-2" />
                <p>Frame-by-frame analysis not available</p>
              </div>
            )}
          </TabsContent>
        </Tabs>

        {/* Explanation */}
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            {result.verdict === 'fake' 
              ? "This media appears to contain AI-generated or manipulated content. The analysis detected patterns consistent with deepfake technology."
              : result.verdict === 'real'
              ? "This media appears to be authentic. No significant signs of AI manipulation were detected."
              : "The analysis was inconclusive. This may be due to low quality, unusual content, or sophisticated manipulation techniques."
            }
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
}
