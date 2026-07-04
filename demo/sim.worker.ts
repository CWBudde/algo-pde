/// <reference lib="webworker" />

// Web Worker: loads the Go WASM acoustic-Helmholtz solver and, on each request,
// solves the driven steady-state field for a source at a clicked point and a
// chosen frequency, then posts back a ready-to-blit RGBA image built in Go.

// Type declarations for the Go WASM runtime and exports.
declare global {
  const Go: new () => GoInstance;
  function goSolveAcoustic(
    nx: number,
    ny: number,
    dx: number,
    dy: number,
    bcX: number,
    bcY: number,
    freqHz: number,
    soundSpeed: number,
    eta: number,
    sx: number,
    sy: number,
    srcRadius: number,
  ): GoSolveResult;
}

interface GoInstance {
  importObject: WebAssembly.Imports;
  run(instance: WebAssembly.Instance): void;
}

interface GoSolveResult {
  success: boolean;
  error?: string;
  width?: number;
  height?: number;
  k?: number;
  lambda?: number;
  rgba?: Uint8Array;
}

// Solver configuration (set on init).
let nx = 0;
let ny = 0;
let dx = 0;
let dy = 0;
let bcX = 2;
let bcY = 2;
let wasmReady = false;

// Physical constants / defaults.
const SPEED_OF_SOUND = 343.0; // m/s
const DAMPING_ETA = 0.03; // damping fraction η in α = −k²(1 − iη)
const SRC_RADIUS = 3.0; // Gaussian source radius (grid cells)

// How long to wait for the Go exports to appear before failing.
const READY_TIMEOUT_MS = 15000;
const READY_POLL_MS = 10;

// Asset URLs. The main thread resolves these against the HTML document and
// passes them in `init` (files in demo/public/ land at the dist root, whereas
// this worker bundle lives in dist/assets/, so a URL relative to the worker
// would point at the wrong directory under a subpath deploy). We fall back to a
// worker-relative URL only if the main thread did not supply them.
let wasmUrl = new URL('../acoustics.wasm', import.meta.url).href;
let wasmExecUrl = new URL('../wasm_exec.js', import.meta.url).href;

self.onmessage = async (e: MessageEvent) => {
  const { type, ...data } = e.data;

  try {
    switch (type) {
      case 'init':
        await handleInit(data);
        break;
      case 'solve':
        handleSolve(data);
        break;
      default:
        self.postMessage({ type: 'error', message: `Unknown message type: ${type}` });
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};

async function loadWasm() {
  if (wasmReady) return;

  // Fetch and evaluate the Go runtime shim in the worker's global scope.
  const execResp = await fetch(wasmExecUrl);
  if (!execResp.ok) {
    throw new Error(`Failed to load wasm_exec.js (${execResp.status} ${execResp.statusText}) from ${wasmExecUrl}`);
  }
  const execScript = await execResp.text();
  // Indirect eval to run in global scope so `Go` becomes globally visible.
  (0, eval)(execScript);

  const go = new Go();
  let result: WebAssembly.WebAssemblyInstantiatedSource;
  try {
    result = await WebAssembly.instantiateStreaming(fetch(wasmUrl), go.importObject);
  } catch (streamErr) {
    // Some servers send the wrong MIME type for .wasm; fall back to ArrayBuffer.
    const wasmResp = await fetch(wasmUrl);
    if (!wasmResp.ok) {
      throw new Error(`Failed to load acoustics.wasm (${wasmResp.status} ${wasmResp.statusText}) from ${wasmUrl}: ${String(streamErr)}`);
    }
    const bytes = await wasmResp.arrayBuffer();
    result = await WebAssembly.instantiate(bytes, go.importObject);
  }

  // Run the Go program (blocks forever inside Go, installing the exports).
  go.run(result.instance);

  // Poll for the exports, but bail out (rejecting) if they never appear so a
  // half-failed instantiation surfaces an error instead of hanging forever.
  await new Promise<void>((resolve, reject) => {
    const deadline = Date.now() + READY_TIMEOUT_MS;
    const check = setInterval(() => {
      if (typeof (globalThis as { goReady?: boolean }).goReady !== 'undefined' && typeof goSolveAcoustic !== 'undefined') {
        clearInterval(check);
        wasmReady = true;
        resolve();
      } else if (Date.now() > deadline) {
        clearInterval(check);
        reject(new Error('Timed out waiting for WASM exports (Go runtime failed to initialize)'));
      }
    }, READY_POLL_MS);
  });
}

async function handleInit(data: {
  nx: number;
  ny: number;
  dx: number;
  dy: number;
  bcX: number;
  bcY: number;
  wasmUrl?: string;
  wasmExecUrl?: string;
}) {
  nx = data.nx;
  ny = data.ny;
  dx = data.dx;
  dy = data.dy;
  bcX = data.bcX;
  bcY = data.bcY;
  if (data.wasmUrl) wasmUrl = data.wasmUrl;
  if (data.wasmExecUrl) wasmExecUrl = data.wasmExecUrl;

  await loadWasm();

  self.postMessage({ type: 'ready', nx, ny });
}

function handleSolve(data: { sx: number; sy: number; freqHz: number }) {
  if (!wasmReady) {
    throw new Error('WASM not initialized. Call init first.');
  }

  const result = goSolveAcoustic(nx, ny, dx, dy, bcX, bcY, data.freqHz, SPEED_OF_SOUND, DAMPING_ETA, data.sx, data.sy, SRC_RADIUS);

  if (!result || typeof result !== 'object') {
    throw new Error('goSolveAcoustic returned an invalid result');
  }
  if (!result.success) {
    throw new Error(result.error || 'solve failed');
  }
  if (!result.rgba) {
    throw new Error('no field returned from solver');
  }

  // The Go side already produced RGBA bytes; wrap in a clamped array for the
  // canvas. Transfer the underlying buffer for a zero-copy hand-off.
  const rgba = new Uint8ClampedArray(result.rgba.buffer, result.rgba.byteOffset, result.rgba.byteLength);

  self.postMessage(
    {
      type: 'pixels',
      data: rgba,
      width: result.width,
      height: result.height,
      freqHz: data.freqHz,
      lambda: result.lambda,
    },
    [rgba.buffer],
  );
}
