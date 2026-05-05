import { apiClient } from './apiClient';
import {
  parseAnalyzeImageResponse,
  parseDemoAnalyzeFrameResponse,
  parseDemoFramesResponse,
  parseHealthResponse,
  parsePredictResponse
} from './strawberryParsers';

export async function health(signal) {
  const res = await apiClient.get('/api/health', { signal });
  return parseHealthResponse(res.data);
}

export async function predictImage(formData, signal) {
  const res = await apiClient.post('/api/predict', formData, { signal });
  return parsePredictResponse(res.data);
}

export async function analyzeImage(formData, signal) {
  const res = await apiClient.post('/api/analyze_image', formData, { signal });
  return parseAnalyzeImageResponse(res.data);
}

export async function fetchDemoFrames(limit, signal) {
  const res = await apiClient.get('/api/demo/frames', { params: { limit }, signal });
  return parseDemoFramesResponse(res.data);
}

export async function analyzeDemoFrame(name, signal) {
  const res = await apiClient.get('/api/demo/analyze_frame', { params: { name }, signal });
  return parseDemoAnalyzeFrameResponse(res.data);
}

