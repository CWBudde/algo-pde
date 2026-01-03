/// <reference lib="webworker" />

// Type declarations for Go WASM
declare global {
  const Go: new () => GoInstance;
  function goInitPlan(nx: number, ny: number, dx: number, dy: number, bcX: number, bcY: number): GoResult;
  function goSolve(planID: string, alpha: number, sx: number, sy: number, srcRadius: number): GoSolveResult;
  function goGetPlanInfo(planID: string): GoPlanInfo;
}

interface GoInstance {
  importObject: WebAssembly.Imports;
  run(instance: WebAssembly.Instance): void;
}

interface GoResult {
  success: boolean;
  error?: string;
  planID?: string;
  nx?: number;
  ny?: number;
  cached?: boolean;
}

interface GoSolveResult {
  success: boolean;
  error?: string;
  field?: Float32Array;
}

interface GoPlanInfo {
  success: boolean;
  error?: string;
  nx?: number;
  ny?: number;
  dx?: number;
  dy?: number;
}

// Worker state
let planID: string | null = null;
let nx = 0;
let ny = 0;
let fields: Float32Array[] = [];
let frequencies: number[] = [];
let dampingFactors: number[] = [];
let lastPingTime = 0;
let isAnimating = false;

// Colormap LUT (blue-white-red diverging)
const colormapLUT = new Uint8Array(256 * 3);
buildDivergingColormap();

function buildDivergingColormap() {
  for (let i = 0; i < 256; i++) {
    const t = i / 255.0;
    const s = 2.0 * t - 1.0; // Map to [-1, 1]

    if (s < 0) {
      // Blue to white
      colormapLUT[i * 3 + 0] = Math.floor(255 * (1 + s)); // R
      colormapLUT[i * 3 + 1] = Math.floor(255 * (1 + s)); // G
      colormapLUT[i * 3 + 2] = 255; // B
    } else {
      // White to red
      colormapLUT[i * 3 + 0] = 255; // R
      colormapLUT[i * 3 + 1] = Math.floor(255 * (1 - s)); // G
      colormapLUT[i * 3 + 2] = Math.floor(255 * (1 - s)); // B
    }
  }
}

function applyColormap(field: Float32Array): Uint8ClampedArray {
  const n = field.length;
  const rgba = new Uint8ClampedArray(n * 4);

  // Compute percentile-based normalization (5th to 95th percentile)
  const sorted = new Float32Array(field);
  sorted.sort();
  const p5 = sorted[Math.floor(n * 0.05)];
  const p95 = sorted[Math.floor(n * 0.95)];
  const range = Math.max(Math.abs(p5), Math.abs(p95), 1e-10);

  // Apply colormap
  for (let i = 0; i < n; i++) {
    const val = field[i];
    const normalized = Math.max(-1, Math.min(1, val / range));
    const idx = Math.floor(((normalized + 1) / 2) * 255);

    rgba[i * 4 + 0] = colormapLUT[idx * 3 + 0];
    rgba[i * 4 + 1] = colormapLUT[idx * 3 + 1];
    rgba[i * 4 + 2] = colormapLUT[idx * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }

  return rgba;
}

function synthesizeFrame(t: number): Float32Array {
  const result = new Float32Array(nx * ny);
  if (fields.length === 0) return result;

  const dt = t - lastPingTime;

  for (let i = 0; i < nx * ny; i++) {
    let sum = 0;
    for (let fi = 0; fi < frequencies.length; fi++) {
      const f = frequencies[fi];
      const gamma = dampingFactors[fi] * f;
      const phase = 2 * Math.PI * f * dt;
      const decay = Math.exp(-gamma * dt);
      sum += fields[fi][i] * Math.cos(phase) * decay;
    }
    result[i] = sum;
  }

  return result;
}

// Message handlers
self.onmessage = async (e: MessageEvent) => {
  const { type, ...data } = e.data;

  try {
    switch (type) {
      case 'init':
        await handleInit(data);
        break;
      case 'ping':
        await handlePing(data);
        break;
      case 'frame':
        handleFrame(data);
        break;
      case 'stop':
        handleStop();
        break;
      default:
        self.postMessage({ type: 'error', message: `Unknown message type: ${type}` });
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: String(error) });
  }
};

async function handleInit(data: { nx: number; ny: number; dx: number; dy: number; bcX: number; bcY: number }) {
  // Load WASM if not already loaded
  if (typeof Go === 'undefined') {
    // Import wasm_exec.js
    importScripts('/wasm_exec.js');

    // Load and initialize WASM module
    const go = new Go();
    const result = await WebAssembly.instantiateStreaming(
      fetch('/acoustics.wasm'),
      go.importObject
    );

    // Run Go runtime in background
    go.run(result.instance);

    // Wait for Go exports to be ready
    await new Promise((resolve) => {
      const check = setInterval(() => {
        if (typeof goInitPlan !== 'undefined') {
          clearInterval(check);
          resolve(undefined);
        }
      }, 10);
    });
  }

  // Initialize plan
  const result = goInitPlan(data.nx, data.ny, data.dx, data.dy, data.bcX, data.bcY);
  if (!result.success) {
    throw new Error(result.error || 'Failed to initialize plan');
  }

  planID = result.planID!;
  nx = result.nx!;
  ny = result.ny!;

  self.postMessage({
    type: 'ready',
    planID,
    nx,
    ny,
  });
}

async function handlePing(data: { sx: number; sy: number; frequencies: number[] }) {
  if (!planID) {
    throw new Error('Plan not initialized. Call init first.');
  }

  const { sx, sy, frequencies: freqs } = data;

  // Store frequencies and compute damping factors
  frequencies = freqs;
  dampingFactors = freqs.map(f => 0.5); // Simple uniform damping

  // Solve for each frequency
  fields = [];
  const c = 343; // Speed of sound (m/s)
  const srcRadius = 3.0; // Grid cells

  for (const f of freqs) {
    const k = (2 * Math.PI * f) / c;
    const alpha = k * k;

    const result = goSolve(planID, alpha, sx, sy, srcRadius);
    if (!result.success) {
      throw new Error(result.error || `Failed to solve for frequency ${f} Hz`);
    }

    fields.push(result.field!);
  }

  lastPingTime = performance.now() / 1000;
  isAnimating = true;

  self.postMessage({
    type: 'computed',
    nFreqs: fields.length,
  });
}

function handleFrame(data: { t: number }) {
  if (!isAnimating || fields.length === 0) {
    return;
  }

  const field = synthesizeFrame(data.t);
  const rgba = applyColormap(field);

  // Transfer ownership of RGBA buffer for zero-copy
  self.postMessage(
    {
      type: 'pixels',
      data: rgba,
      width: nx,
      height: ny,
    },
    [rgba.buffer]
  );
}

function handleStop() {
  isAnimating = false;
  fields = [];
  frequencies = [];
  dampingFactors = [];
}
