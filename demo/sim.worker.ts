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
let isAnimating = false;

// Speed of sound (m/s)
const SPEED_OF_SOUND = 343.0;

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

  // Find min/max for symmetric range
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = field[i];
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }

  // Use symmetric range for diverging colormap
  const range = Math.max(Math.abs(minVal), Math.abs(maxVal), 1e-10);

  // Apply colormap with enhanced contrast
  for (let i = 0; i < n; i++) {
    const val = field[i];
    // Enhanced contrast: apply power function to compress mid-tones
    const normalized = Math.max(-1, Math.min(1, val / range));
    const enhanced = Math.sign(normalized) * Math.pow(Math.abs(normalized), 0.7);
    const idx = Math.floor(((enhanced + 1) / 2) * 255);

    rgba[i * 4 + 0] = colormapLUT[idx * 3 + 0];
    rgba[i * 4 + 1] = colormapLUT[idx * 3 + 1];
    rgba[i * 4 + 2] = colormapLUT[idx * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }

  return rgba;
}

/**
 * Synthesize wave field at given time by combining frequency modes
 * @param t Time in seconds (real time, not scaled)
 * @returns Combined wave field
 */
function synthesizeFrame(t: number): Float32Array {
  const result = new Float32Array(nx * ny);
  if (fields.length === 0) return result;

  // Each frequency mode oscillates as: A_i * cos(2π * f_i * t) * exp(-γ_i * t)
  // where γ_i is the damping coefficient

  for (let i = 0; i < nx * ny; i++) {
    let sum = 0;

    for (let fi = 0; fi < frequencies.length; fi++) {
      const f = frequencies[fi];
      const omega = 2 * Math.PI * f; // Angular frequency (rad/s)

      // Damping: higher frequencies decay faster
      // Using quality factor Q ≈ 10 for room acoustics
      const gamma = omega / 20.0; // Decay rate (1/s)

      const amplitude = fields[fi][i];
      const oscillation = Math.cos(omega * t);
      const decay = Math.exp(-gamma * t);

      sum += amplitude * oscillation * decay;
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
    // Dynamically import wasm_exec.js as a module
    await new Promise<void>((resolve, reject) => {
      // Fetch and evaluate wasm_exec.js in global scope
      fetch('/wasm_exec.js')
        .then(response => response.text())
        .then(script => {
          // Execute in global scope using indirect eval
          (0, eval)(script);
          resolve();
        })
        .catch(reject);
    });

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
        const ready = typeof goInitPlan !== 'undefined' &&
                     typeof goSolve !== 'undefined' &&
                     typeof goGetPlanInfo !== 'undefined';

        if (ready) {
          clearInterval(check);
          resolve(undefined);
        }
      }, 10);
    });
  }

  // Initialize plan
  const result = goInitPlan(data.nx, data.ny, data.dx, data.dy, data.bcX, data.bcY);

  if (!result || typeof result !== 'object') {
    throw new Error(`goInitPlan returned invalid result`);
  }

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

  // Store frequencies
  frequencies = freqs;

  // Solve for each frequency
  fields = [];
  const srcRadius = 3.0; // Grid cells

  for (let i = 0; i < freqs.length; i++) {
    const f = freqs[i];
    const k = (2 * Math.PI * f) / SPEED_OF_SOUND; // Wave number (1/m)
    const alpha = k * k; // Helmholtz parameter

    // Send progress update
    self.postMessage({
      type: 'progress',
      current: i + 1,
      total: freqs.length,
      frequency: f,
    });

    const result = goSolve(planID, alpha, sx, sy, srcRadius);

    if (!result.success) {
      throw new Error(result.error || `Failed to solve for frequency ${f} Hz`);
    }

    if (!result.field) {
      throw new Error(`No field returned for frequency ${f} Hz`);
    }

    fields.push(result.field);
  }

  isAnimating = true;

  self.postMessage({
    type: 'computed',
    nFreqs: fields.length,
    freqRange: [freqs[0], freqs[freqs.length - 1]],
  });
}

function handleFrame(data: { t: number }) {
  if (!isAnimating || fields.length === 0) {
    return;
  }

  // t is already in seconds - use it directly
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
}
