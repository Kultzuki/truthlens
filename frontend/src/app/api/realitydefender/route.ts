import { NextRequest, NextResponse } from 'next/server';
import { RealityDefender } from '@realitydefender/realitydefender';

// Initialize Reality Defender client
const getClient = () => {
  const apiKey = process.env.REALITY_DEFENDER_API_KEY;
  if (!apiKey) {
    throw new Error('REALITY_DEFENDER_API_KEY environment variable is required');
  }
  return new RealityDefender({ apiKey });
};

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Convert file to buffer for SDK
    const buffer = Buffer.from(await file.arrayBuffer());
    
    // Initialize Reality Defender client
    const client = getClient();
    
    // Detect using SDK
    const result = await client.detect({
      content: buffer,
      contentType: file.type,
      filename: file.name
    });

    // Transform SDK result to standardized format
    const transformedResult = {
      verdict: result.verdict || 'unknown',
      confidence: result.confidence || 0,
      processing_time: result.processingTime || 0,
      signals: result.signals || [{
        name: "RealityDefender SDK",
        score: result.confidence || 0,
        description: "Official Reality Defender deepfake detection service"
      }],
      metadata: {
        status: result.status || "completed",
        models: result.models || [],
        ...result.metadata
      }
    };

    return NextResponse.json(transformedResult);
  } catch (error: any) {
    console.error('Reality Defender API error:', error);
    
    return NextResponse.json(
      { 
        error: 'Reality Defender detection failed',
        details: error.message 
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    // Check if Reality Defender is available
    const apiKey = process.env.REALITY_DEFENDER_API_KEY;
    
    return NextResponse.json({
      available: !!apiKey,
      service: 'Reality Defender SDK'
    });
  } catch (error) {
    return NextResponse.json({
      available: false,
      error: 'Reality Defender not configured'
    });
  }
}