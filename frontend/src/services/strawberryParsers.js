function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function assertString(value, path) {
  if (typeof value !== 'string') throw new Error(`Invalid ${path}`);
  return value;
}

function assertNumber(value, path) {
  if (typeof value !== 'number' || Number.isNaN(value)) throw new Error(`Invalid ${path}`);
  return value;
}

function assertPoint2D(value, path) {
  if (!Array.isArray(value) || value.length !== 2) throw new Error(`Invalid ${path}`);
  return [assertNumber(value[0], `${path}[0]`), assertNumber(value[1], `${path}[1]`)];
}

function assertPosition(value, path) {
  if (!isPlainObject(value)) throw new Error(`Invalid ${path}`);
  return {
    x: assertNumber(value.x, `${path}.x`),
    y: assertNumber(value.y, `${path}.y`),
    z: assertNumber(value.z, `${path}.z`)
  };
}

function assertDimensions(value, path) {
  if (!isPlainObject(value)) throw new Error(`Invalid ${path}`);
  return {
    l: assertNumber(value.l, `${path}.l`),
    w: assertNumber(value.w, `${path}.w`),
    h: assertNumber(value.h, `${path}.h`)
  };
}

function parseTarget(value, path) {
  if (!isPlainObject(value)) throw new Error(`Invalid ${path}`);

  const points2d = value.points_2d;
  if (!Array.isArray(points2d)) throw new Error(`Invalid ${path}.points_2d`);
  const parsedPoints = points2d.map((p, i) => assertPoint2D(p, `${path}.points_2d[${i}]`));

  let axis2d;
  if (value.axis_2d !== undefined) {
    if (!Array.isArray(value.axis_2d)) throw new Error(`Invalid ${path}.axis_2d`);
    axis2d = value.axis_2d.map((p, i) => assertPoint2D(p, `${path}.axis_2d[${i}]`));
  }

  return {
    confidence: assertNumber(value.confidence, `${path}.confidence`),
    center_2d: assertPoint2D(value.center_2d, `${path}.center_2d`),
    points_2d: parsedPoints,
    axis_2d: axis2d,
    position: assertPosition(value.position, `${path}.position`),
    dimensions: assertDimensions(value.dimensions, `${path}.dimensions`)
  };
}

export function parseHealthResponse(data) {
  if (!isPlainObject(data)) throw new Error('Invalid health response');
  if (data.status !== 'ok') throw new Error('Backend not ok');
  return { status: 'ok' };
}

export function parsePredictResponse(data) {
  if (!Array.isArray(data)) throw new Error('Invalid predict response');
  if (data.length > 0 && isPlainObject(data[0]) && typeof data[0].error === 'string') {
    throw new Error(data[0].error);
  }
  return data.map((t, i) => parseTarget(t, `targets[${i}]`));
}

export function parseAnalyzeImageResponse(data) {
  if (!isPlainObject(data)) throw new Error('Invalid analyze response');
  if (typeof data.error === 'string') throw new Error(data.error);
  const targets = Array.isArray(data.targets) ? data.targets : null;
  if (!targets) throw new Error('Invalid analyze response.targets');
  return {
    result_image: assertString(data.result_image, 'result_image'),
    targets: targets.map((t, i) => parseTarget(t, `targets[${i}]`))
  };
}

export function parseDemoFramesResponse(data) {
  if (!isPlainObject(data)) throw new Error('Invalid demo frames response');
  const frames = Array.isArray(data.frames) ? data.frames : null;
  if (!frames) throw new Error('Invalid demo frames response.frames');
  return frames.map((f, i) => {
    if (!isPlainObject(f)) throw new Error(`Invalid frames[${i}]`);
    return {
      name: assertString(f.name, `frames[${i}].name`),
      url: assertString(f.url, `frames[${i}].url`)
    };
  });
}

export function parseDemoAnalyzeFrameResponse(data) {
  if (!isPlainObject(data)) throw new Error('Invalid demo analyze response');
  if (typeof data.error === 'string') throw new Error(data.error);
  const targets = Array.isArray(data.targets) ? data.targets : null;
  if (!targets) throw new Error('Invalid demo analyze response.targets');
  return {
    frame: assertString(data.frame, 'frame'),
    mode: assertString(data.mode, 'mode'),
    result_image: assertString(data.result_image, 'result_image'),
    targets: targets.map((t, i) => parseTarget(t, `targets[${i}]`))
  };
}

