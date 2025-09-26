import { NextRequest, NextResponse } from 'next/server';

// Define interfaces for the API
interface RealityDefenderResult {
  confidence: number;
  is_fake: boolean;
  processing_time?: number;
}

interface RealityDefenderRequest {
  file_url?: string;
  file_data?: string; // base64 encoded file
  file_type?: string;
}

// Mock RealityDefender service for now - replace with actual implementation
class ServerRealityDefenderService {
  private apiKey: string | null;
  
  constructor() {
    this.apiKey = process.env.REALITYDEFENDER_API_KEY || null;
  }

  isAvailable(): boolean {
    return !!this.apiKey && this.apiKey !== 'your_api_key_here';
  }

  async detect(fileData: string, fileType: string): Promise<RealityDefenderResult> {
    if (!this.isAvailable()) {
      throw new Error('RealityDefender API not available');
    }

    try {
      // TODO: Replace with actual RealityDefender API call
      // For now, return mock data to prevent build errors
      const mockResult: RealityDefenderResult = {
        confidence: Math.random() * 0.3 + 0.7, // Random confidence between 0.7-1.0
        is_fake: Math.random() > 0.5,
        processing_time: Math.random() * 2 + 1 // 1-3 seconds
      };

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      return mockResult;
    } catch (error) {
      console.error('RealityDefender API error:', error);
      throw new Error('RealityDefender detection failed');
    }
  }
}

const realityDefenderService = new ServerRealityDefenderService();

export async function POST(request: NextRequest) {
  try {
    if (!realityDefenderService.isAvailable()) {
      return NextResponse.json(
        { error: 'RealityDefender API not configured' },
        { status: 503 }
      );
    }

    const body: RealityDefenderRequest = await request.json();
    
    if (!body.file_data || !body.file_type) {
      return NextResponse.json(
        { error: 'Missing file_data or file_type' },
        { status: 400 }
      );
    }

    const result = await realityDefenderService.detect(body.file_data, body.file_type);
    
    return NextResponse.json({
      success: true,
      result
    });

  } catch (error) {
    console.error('RealityDefender API route error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    available: realityDefenderService.isAvailable(),
    message: realityDefenderService.isAvailable() 
      ? 'RealityDefender API is available'
      : 'RealityDefender API key not configured'
  });
}