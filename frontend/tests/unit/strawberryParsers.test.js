import {
  parseAnalyzeImageResponse,
  parseDemoAnalyzeFrameResponse,
  parseDemoFramesResponse,
  parseHealthResponse,
  parsePredictResponse
} from '../../src/services/strawberryParsers';

describe('strawberryParsers', () => {
  test('parseHealthResponse: ok', () => {
    expect(parseHealthResponse({ status: 'ok' })).toEqual({ status: 'ok' });
  });

  test('parseHealthResponse: invalid throws', () => {
    expect(() => parseHealthResponse({ status: 'nope' })).toThrow();
  });

  test('parsePredictResponse: targets array', () => {
    const data = [
      {
        confidence: 0.9,
        center_2d: [10, 20],
        points_2d: Array.from({ length: 9 }, () => [0, 0]),
        position: { x: 1, y: 2, z: 3 },
        dimensions: { l: 1, w: 2, h: 3 }
      }
    ];
    const targets = parsePredictResponse(data);
    expect(targets).toHaveLength(1);
    expect(targets[0].confidence).toBe(0.9);
    expect(targets[0].center_2d).toEqual([10, 20]);
  });

  test('parsePredictResponse: backend error array throws message', () => {
    expect(() => parsePredictResponse([{ error: 'boom' }])).toThrow('boom');
  });

  test('parseAnalyzeImageResponse: ok', () => {
    const data = {
      result_image: 'data:image/jpeg;base64,abc',
      targets: [
        {
          confidence: 0.7,
          center_2d: [1, 2],
          points_2d: Array.from({ length: 9 }, () => [0, 0]),
          position: { x: 1, y: 2, z: 3 },
          dimensions: { l: 1, w: 2, h: 3 }
        }
      ]
    };
    const res = parseAnalyzeImageResponse(data);
    expect(res.result_image).toContain('data:image');
    expect(res.targets).toHaveLength(1);
  });

  test('parseDemoFramesResponse: ok', () => {
    const frames = parseDemoFramesResponse({
      count: 2,
      frames: [
        { name: 'a.png', url: '/demo-data/a.png' },
        { name: 'b.png', url: '/demo-data/b.png' }
      ]
    });
    expect(frames).toHaveLength(2);
    expect(frames[0].url).toBe('/demo-data/a.png');
  });

  test('parseDemoAnalyzeFrameResponse: ok', () => {
    const res = parseDemoAnalyzeFrameResponse({
      frame: 'a.png',
      mode: 'ground_truth_replay',
      result_image: 'data:image/jpeg;base64,abc',
      targets: [
        {
          confidence: 1,
          center_2d: [1, 2],
          points_2d: Array.from({ length: 9 }, () => [0, 0]),
          position: { x: 1, y: 2, z: 3 },
          dimensions: { l: 1, w: 2, h: 3 }
        }
      ]
    });
    expect(res.frame).toBe('a.png');
    expect(res.targets).toHaveLength(1);
  });
});

