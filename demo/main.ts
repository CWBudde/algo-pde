// Main UI thread — Acoustic Room Modes demo.
//
// Solves the driven acoustic Helmholtz equation on a rigid-walled rectangular
// room and renders the steady-state pressure field. Click places the driving
// source; the frequency slider sweeps the drive frequency through the room's
// standing-wave modes. A small damping term keeps the field finite on
// resonance.

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d', { alpha: false })!;
const statusEl = document.querySelector('#overlay .status') as HTMLDivElement;
const hintEl = document.querySelector('#overlay .hint') as HTMLDivElement;
const debugEl = document.querySelector('#overlay .debug') as HTMLDivElement;
const controlsEl = document.getElementById('controls') as HTMLDivElement;
const freqSlider = document.getElementById('freqSlider') as HTMLInputElement;
const freqDisplay = document.getElementById('freqDisplay') as HTMLSpanElement;

// Room / grid configuration.
const CONFIG = {
  nx: 256, // grid width  (12.8 m at 0.05 m/cell)
  ny: 192, // grid height ( 9.6 m) — 4:3 room
  dx: 0.05,
  dy: 0.05,
  bcX: 2, // Neumann (rigid walls)
  bcY: 2,
  fMin: 40, // Hz
  fMax: 600, // Hz
  fDefault: 120, // Hz
};

interface AppState {
  worker: Worker | null;
  isReady: boolean;
  imageData: ImageData | null;
  // Last source placement in grid-cell coordinates (null until first click).
  source: { sx: number; sy: number } | null;
  freqHz: number;
}

const state: AppState = {
  worker: null,
  isReady: false,
  imageData: null,
  source: null,
  freqHz: CONFIG.fDefault,
};

function resizeCanvas() {
  canvas.width = CONFIG.nx;
  canvas.height = CONFIG.ny;

  const aspectRatio = CONFIG.nx / CONFIG.ny;
  const windowAspect = window.innerWidth / window.innerHeight;

  let displayWidth: number;
  let displayHeight: number;
  if (windowAspect > aspectRatio) {
    displayHeight = window.innerHeight * 0.9;
    displayWidth = displayHeight * aspectRatio;
  } else {
    displayWidth = window.innerWidth * 0.9;
    displayHeight = displayWidth / aspectRatio;
  }
  canvas.style.width = displayWidth + 'px';
  canvas.style.height = displayHeight + 'px';

  if (!state.imageData) {
    state.imageData = ctx.createImageData(CONFIG.nx, CONFIG.ny);
    clearCanvas();
  }
}

function clearCanvas() {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, CONFIG.nx, CONFIG.ny);
  drawBoundaries();
}

function drawBoundaries() {
  ctx.strokeStyle = '#333333';
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, CONFIG.nx - 2, CONFIG.ny - 2);
}

async function initWorker() {
  state.worker = new Worker(new URL('./sim.worker.ts', import.meta.url), {
    type: 'module',
  });

  state.worker.onmessage = (e) => {
    const { type, ...data } = e.data;
    switch (type) {
      case 'ready':
        handleReady(data);
        break;
      case 'pixels':
        handlePixels(data);
        break;
      case 'error':
        handleError(data);
        break;
    }
  };

  state.worker.onerror = (error) => {
    console.error('Worker error:', error);
    statusEl.textContent = `Worker error: ${error.message}`;
  };

  // Resolve the WASM asset URLs against the HTML document, not the worker
  // bundle. Files in demo/public/ are emitted at the dist root, whereas the
  // worker bundle lives in dist/assets/, so a URL relative to the worker would
  // point at the wrong directory. document.baseURI is the page location (e.g.
  // the GitHub-Pages "/algo-pde/" subpath), so this resolves correctly there.
  const wasmUrl = new URL('acoustics.wasm', document.baseURI).href;
  const wasmExecUrl = new URL('wasm_exec.js', document.baseURI).href;

  state.worker.postMessage({
    type: 'init',
    nx: CONFIG.nx,
    ny: CONFIG.ny,
    dx: CONFIG.dx,
    dy: CONFIG.dy,
    bcX: CONFIG.bcX,
    bcY: CONFIG.bcY,
    wasmUrl,
    wasmExecUrl,
  });
}

function handleReady(data: { nx: number; ny: number }) {
  state.isReady = true;
  const w = (CONFIG.dx * data.nx).toFixed(1);
  const h = (CONFIG.dy * data.ny).toFixed(1);
  statusEl.textContent = `Ready — rigid room ${data.nx}×${data.ny} cells (${w}×${h} m)`;
  hintEl.textContent = 'Click to place the driving source, then sweep the frequency slider';
  controlsEl.classList.add('active');
  updateDebugInfo();
}

function handlePixels(data: {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  freqHz: number;
  lambda: number;
}) {
  if (!state.imageData) return;
  state.imageData.data.set(data.data);
  ctx.putImageData(state.imageData, 0, 0);
  drawBoundaries();

  statusEl.textContent = `Driven at ${data.freqHz.toFixed(0)} Hz — steady-state room response`;
  updateDebugInfo(data.lambda);
}

function handleError(data: { message: string }) {
  console.error('Worker error:', data.message);
  statusEl.textContent = `Error: ${data.message}`;
}

function updateDebugInfo(lambda?: number) {
  const lines = [`f = ${state.freqHz.toFixed(0)} Hz`];
  if (lambda && lambda > 0) {
    lines.push(`λ = ${lambda.toFixed(2)} m`);
  }
  if (state.source) {
    lines.push(`src = (${state.source.sx.toFixed(0)}, ${state.source.sy.toFixed(0)})`);
  }
  debugEl.textContent = lines.join(' | ');
}

// Ask the worker for a fresh steady-state solve at the current source & freq.
function requestSolve() {
  if (!state.isReady || !state.worker || !state.source) return;
  state.worker.postMessage({
    type: 'solve',
    sx: state.source.sx,
    sy: state.source.sy,
    freqHz: state.freqHz,
  });
}

// Event handlers.
canvas.addEventListener('click', (e) => {
  if (!state.isReady || !state.worker) return;

  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;
  // Clamp to [0, n-1]: a click on the right/bottom edge gives x or y == 1,
  // which would otherwise place the source center just outside the grid.
  const sx = Math.min(CONFIG.nx - 1, Math.max(0, x * CONFIG.nx));
  const sy = Math.min(CONFIG.ny - 1, Math.max(0, y * CONFIG.ny));
  state.source = { sx, sy };

  statusEl.textContent = 'Solving…';
  requestSolve();
});

freqSlider.addEventListener('input', () => {
  state.freqHz = parseFloat(freqSlider.value);
  freqDisplay.textContent = state.freqHz.toFixed(0) + ' Hz';
  updateDebugInfo();
  requestSolve();
});

window.addEventListener('resize', resizeCanvas);

// Initialize.
freqSlider.min = String(CONFIG.fMin);
freqSlider.max = String(CONFIG.fMax);
freqSlider.value = String(CONFIG.fDefault);
freqDisplay.textContent = CONFIG.fDefault.toFixed(0) + ' Hz';
resizeCanvas();
initWorker();
