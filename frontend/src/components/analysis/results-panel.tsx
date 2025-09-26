import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Scan, CheckCircle, AlertTriangle, Loader2 } from "lucide-react";

// Legacy Analysis type for backward compatibility
type Analysis = {
  verdict: string;
  confidence: number;
  inputType: string;
  input: string;
  latencyMs: number;
  signals: { name: string; score: number }[];
};

type Props = {
  result?: Analysis | null;
  loading?: boolean;
  error?: string | null;
};

export function ResultsPanel({ result, loading, error }: Props) {
  return (
    <Card className="glassmorphic">
      <CardHeader>
        <CardTitle>Analysis Results</CardTitle>
        <CardDescription>The AI-powered analysis will appear here.</CardDescription>
      </CardHeader>
      <CardContent>
        {!result && !loading && !error && (
          <div className="flex flex-col items-center justify-center h-80 text-center text-muted-foreground space-y-4 animate-in fade-in-50 duration-500">
            <div className="relative">
              <Scan className="w-16 h-16 animate-pulse" />
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-primary/20 rounded-full animate-ping"></div>
            </div>
            <p className="animate-in slide-in-from-bottom-2 duration-700">Awaiting analysis...</p>
            <p className="text-sm animate-in slide-in-from-bottom-2 duration-700 delay-200">Upload a file or paste a URL to get started</p>
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center justify-center h-80 text-center text-muted-foreground space-y-4 animate-in fade-in-50 duration-500">
            <div className="relative">
              <Loader2 className="w-12 h-12 text-primary animate-spin" />
              <div className="absolute inset-0 rounded-full border-2 border-primary/20"></div>
            </div>
            <div className="space-y-2">
              <p className="text-lg font-medium animate-pulse">Analyzing media...</p>
              <p className="text-sm">This may take a few moments</p>
            </div>
            <div className="w-full max-w-xs">
              <div className="h-2 bg-primary/20 rounded-full overflow-hidden">
                <div className="h-full bg-primary rounded-full animate-pulse"></div>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center justify-center h-80 text-center space-y-4 shake-error">
            <AlertTriangle className="w-12 h-12 text-destructive bounce-in" />
            <div className="space-y-2 slide-up-fade">
              <p className="text-lg font-medium text-destructive">Analysis Failed</p>
              <p className="text-sm text-muted-foreground">{error}</p>
            </div>
          </div>
        )}

        {result && !loading && (
          <div className="space-y-4 slide-up-fade">
            <div className="flex items-center space-x-2 mb-4 bounce-in">
              <CheckCircle className="w-5 h-5 text-green-500 pulse-success" />
              <span className="text-sm font-medium text-green-600">Analysis Complete</span>
            </div>
            <div className="flex flex-wrap items-center gap-3 slide-up-fade">
              <span className="px-2 py-1 rounded bg-primary/10 text-primary text-xs bounce-in">{result.inputType}</span>
              <span className="text-sm break-all text-muted-foreground">{result.input}</span>
            </div>
            <div className="grid grid-cols-2 gap-4 slide-up-fade">
              <div className="rounded-md bg-background/50 p-4 transition-all duration-300 hover:bg-background/70 hover:scale-105">
                <div className="text-sm text-muted-foreground">Verdict</div>
                <div className="text-lg font-semibold capitalize">{result.verdict.replace("_"," ")}</div>
              </div>
              <div className="rounded-md bg-background/50 p-4 transition-all duration-300 hover:bg-background/70 hover:scale-105">
                <div className="text-sm text-muted-foreground">Confidence</div>
                <div className="text-lg font-semibold">{Math.round(result.confidence * 100)}%</div>
              </div>
              <div className="rounded-md bg-background/50 p-4 transition-all duration-300 hover:bg-background/70 hover:scale-105">
                <div className="text-sm text-muted-foreground">Latency</div>
                <div className="text-lg font-semibold">{result.latencyMs} ms</div>
              </div>
            </div>
            <div className="slide-up-fade">
              <div className="text-sm text-muted-foreground mb-2">Signals</div>
              <ul className="space-y-2">
                {result.signals.map((s, index) => (
                  <li 
                    key={s.name} 
                    className="flex items-center justify-between rounded bg-background/40 p-2 transition-all duration-300 hover:bg-background/60 hover:scale-105 bounce-in"
                    style={{ animationDelay: `${200 + index * 100}ms` }}
                  >
                    <span className="capitalize">{s.name.replace("_"," ")}</span>
                    <span className="text-sm text-muted-foreground">{Math.round(s.score * 100)}%</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
