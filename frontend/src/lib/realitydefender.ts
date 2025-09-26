export interface RealityDefenderResult {
  confidence: number;
  is_fake: boolean;
  processing_time?: number;
}

export interface RealityDefenderConfig {
  apiKey: string;
  timeout?: number;
  retries?: number;
}

export class RealityDefenderService {
  private static instance: RealityDefenderService | null = null;
  private config: RealityDefenderConfig | null = null;
  private apiUrl: string;

  private constructor() {
    this.apiUrl = '/api/realitydefender';
    this.initialize();
  }

  public static getInstance(): RealityDefenderService {
    if (!RealityDefenderService.instance) {
      RealityDefenderService.instance = new RealityDefenderService();
    }
    return RealityDefenderService.instance;
  }

  private initialize(): void {
    try {
      const apiKey = process.env.NEXT_PUBLIC_REALITYDEFENDER_API_KEY;
      
      if (!apiKey || apiKey === 'your_api_key_here') {
        console.warn('RealityDefender API key not configured');
        return;
      }

      this.config = {
        apiKey,
        timeout: 30000, // 30 seconds
        retries: 2
      };

      console.log('RealityDefender service initialized');
    } catch (error) {
      console.error('Failed to initialize RealityDefender:', error);
    }
  }

  public async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(this.apiUrl, {
        method: 'GET',
      });
      
      if (!response.ok) {
        return false;
      }
      
      const data = await response.json();
      return data.available === true;
    } catch (error) {
      console.error('Failed to check RealityDefender availability:', error);
      return false;
    }
  }

  public async detect(file: File): Promise<RealityDefenderResult> {
    const available = await this.isAvailable();
    if (!available) {
      throw new Error('RealityDefender service not available');
    }

    try {
      // Convert file to base64
      const fileData = await this.fileToBase64(file);
      
      const response = await fetch(this.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_data: fileData,
          file_type: file.type
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'API request failed');
      }

      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Detection failed');
      }
      
      return data.result;
    } catch (error) {
      console.error('RealityDefender detection failed:', error);
      throw new Error('Detection failed');
    }
  }

  public async detectBatch(files: File[]): Promise<RealityDefenderResult[]> {
    const results: RealityDefenderResult[] = [];
    
    for (const file of files) {
      try {
        const result = await this.detect(file);
        results.push(result);
      } catch (error) {
        console.error(`Batch detection failed for file ${file.name}:`, error);
        results.push({
          confidence: 0.5,
          is_fake: false,
          processing_time: 0
        });
      }
    }
    
    return results;
  }

  private async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data URL prefix (e.g., "data:image/jpeg;base64,")
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = error => reject(error);
    });
  }
}