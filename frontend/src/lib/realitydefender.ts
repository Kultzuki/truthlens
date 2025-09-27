export interface RealityDefenderResult {
  verdict: 'real' | 'fake' | 'unknown';
  confidence: number;
  processing_time?: number;
  signals?: Array<{
    name: string;
    score: number;
    description: string;
  }>;
  metadata?: {
    status?: string;
    models?: string[];
    [key: string]: any;
  };
}

export class RealityDefenderService {
  private static instance: RealityDefenderService;

  private constructor() {}

  public static getInstance(): RealityDefenderService {
    if (!RealityDefenderService.instance) {
      RealityDefenderService.instance = new RealityDefenderService();
    }
    return RealityDefenderService.instance;
  }

  public async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch('/api/realitydefender', {
        method: 'GET'
      });
      
      if (!response.ok) {
        return false;
      }
      
      const data = await response.json();
      return data.available === true;
    } catch (error) {
      console.warn('Failed to check Reality Defender availability:', error);
      return false;
    }
  }

  public async detect(file: File): Promise<RealityDefenderResult> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/realitydefender', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.details || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (error: any) {
      console.error('Reality Defender detection failed:', error);
      throw new Error(`Reality Defender detection failed: ${error.message}`);
    }
  }

  public async detectBatch(files: File[]): Promise<RealityDefenderResult[]> {
    const results: RealityDefenderResult[] = [];
    
    for (const file of files) {
      try {
        const result = await this.detect(file);
        results.push(result);
      } catch (error) {
        console.error(`Failed to detect file ${file.name}:`, error);
        // Add error result for failed detection
        results.push({
          verdict: 'unknown',
          confidence: 0,
          processing_time: 0,
          signals: [{
            name: "Error",
            score: 0,
            description: `Detection failed: ${error}`
          }],
          metadata: {
            status: "error",
            models: []
          }
        });
      }
    }
    
    return results;
  }
}