// Main UI thread — Acoustic Room Modes demo (2D room + 3D box).
//
// Solves the driven acoustic Helmholtz equation on a rigid-walled rectangular
// room (2D) or box (3D) and renders the steady-state pressure field. Click
// places the driving source; the frequency slider sweeps the drive frequency
// through the room's standing-wave modes. A small damping term keeps the field
// finite on resonance.
//
// In 2D the solver returns one image per solve. In 3D it returns the whole
// volume as stacked Z-planes in a single solve; the depth slider then scrubs
// through slices client-side with no further solves — only a frequency or
// source change triggers a new solve.

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d', { alpha: false })!;
const statusEl = document.querySelector('#overlay .status') as HTMLDivElement;
const hintEl = document.querySelector('#overlay .hint') as HTMLDivElement;
const debugEl = document.querySelector('#overlay .debug') as HTMLDivElement;
const controlsEl = document.getElementById('controls') as HTMLDivElement;
const freqSlider = document.getElementById('freqSlider') as HTMLInputElement;
const freqDisplay = document.getElementById('freqDisplay') as HTMLSpanElement;
const mode2dBtn = document.getElementById('mode2dBtn') as HTMLButtonElement;
const mode3dBtn = document.getElementById('mode3dBtn') as HTMLButtonElement;
const depthControl = document.getElementById('depthControl') as HTMLDivElement;
const depthSlider = document.getElementById('depthSlider') as HTMLInputElement;
const depthDisplay = document.getElementById('depthDisplay') as HTMLSpanElement;

type Mode = '2d' | '3d';

// 2D room / grid configuration (unchanged from the original demo).
const CONFIG_2D = {
  nx: 256, // grid width  (12.8 m at 0.05 m/cell)
  ny: 192, // grid height ( 9.6 m) — 4:3 room
  dx: 0.05,
  dy: 0.05,
  bcX: 2, // Neumann (rigid walls)
  bcY: 2,
};

// 3D box configuration. Smaller cells-per-axis keep the single-threaded WASM
// solve interactive; all extents factor as 2^a·3^b so the mixed-radix FFT is
// fast. 96×72×48 @ 0.08 m = 7.68 × 5.76 × 3.84 m room.
const CONFIG_3D = {
  nx: 96,
  ny: 72,
  nz: 48,
  dx: 0.08,
  dy: 0.08,
  dz: 0.08,
  bcX: 2,
  bcY: 2,
  bcZ: 2,
};

const FREQ = {
  fMin: 40, // Hz
  fMax: 600, // Hz
  fDefault: 120, // Hz
};

interface AppState {
  worker: Worker | null;
  isReady: boolean;
  mode: Mode;
  imageData: ImageData | null;
  // Last source placement in grid-cell coordinates (null until first click).
  // sz is only meaningful in 3D.
  source: { sx: number; sy: number; sz: number } | null;
  freqHz: number;
  // 3D: the full RGBA volume (nz stacked width×height planes) from the last
  // solve, and the currently displayed Z-slice.
  volume: Uint8ClampedArray | null;
  slice: number;
  // Monotonic id of the most recent solve request. A worker reply is only
  // applied if it still matches; a mode switch or newer request invalidates
  // any solve still in flight.
  reqId: number;
  // The worker is single-threaded and a 3D solve is expensive, so at most one
  // solve runs at a time. `busy` is true while a solve is in flight; `pending`
  // records that a newer solve was requested meanwhile so we can fire it (with
  // the latest source & frequency) once the current one returns, rather than
  // flooding the worker's message queue with every slider step.
  busy: boolean;
  pending: boolean;
}

const state: AppState = {
  worker: null,
  isReady: false,
  mode: '2d',
  imageData: null,
  source: null,
  freqHz: FREQ.fDefault,
  volume: null,
  slice: Math.floor(CONFIG_3D.nz / 2),
  reqId: 0,
  busy: false,
  pending: false,
};

// Active-mode grid dimensions used for canvas sizing and coordinate mapping.
function gridW(): number {
  return state.mode === '2d' ? CONFIG_2D.nx : CONFIG_3D.nx;
}

function gridH(): number {
  return state.mode === '2d' ? CONFIG_2D.ny : CONFIG_3D.ny;
}

function resizeCanvas() {
  const w = gridW();
  const h = gridH();
  canvas.width = w;
  canvas.height = h;

  const aspectRatio = w / h;
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

  // The ImageData is tied to the canvas size, so rebuild it on any resize that
  // changes dimensions (including a 2D⇄3D mode switch).
  state.imageData = ctx.createImageData(w, h);
  clearCanvas();

  // A window resize recreates (and blanks) the ImageData; in 3D re-blit the
  // cached slice so the field doesn't vanish until the depth slider moves.
  if (state.mode === '3d' && state.volume) blitSlice();
}

function clearCanvas() {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, gridW(), gridH());
  drawBoundaries();
}

function drawBoundaries() {
  ctx.strokeStyle = '#333333';
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, gridW() - 2, gridH() - 2);
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
      case 'volume':
        handleVolume(data);
        break;
      case 'error':
        handleError(data);
        break;
    }

    // Any solve reply — even one dropped as stale — means the worker is idle
    // again. Release the in-flight guard and, if a newer solve was requested
    // while it was busy, fire it now (picking up the latest source & freq).
    if (type === 'pixels' || type === 'volume' || type === 'error') {
      state.busy = false;
      if (state.pending) {
        state.pending = false;
        requestSolve();
      }
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

  // init only loads the WASM module; both the 2D and 3D solvers are installed
  // by it, and each solve message carries its own grid parameters.
  state.worker.postMessage({
    type: 'init',
    nx: CONFIG_2D.nx,
    ny: CONFIG_2D.ny,
    dx: CONFIG_2D.dx,
    dy: CONFIG_2D.dy,
    bcX: CONFIG_2D.bcX,
    bcY: CONFIG_2D.bcY,
    wasmUrl,
    wasmExecUrl,
  });
}

function handleReady(_data: { nx: number; ny: number }) {
  state.isReady = true;
  updateStatusReady();
  hintEl.textContent = 'Click to place the driving source, then sweep the frequency slider';
  controlsEl.classList.add('active');
  updateDebugInfo();
}

function updateStatusReady() {
  if (state.mode === '2d') {
    const w = (CONFIG_2D.dx * CONFIG_2D.nx).toFixed(1);
    const h = (CONFIG_2D.dy * CONFIG_2D.ny).toFixed(1);
    statusEl.textContent = `Ready — rigid room ${CONFIG_2D.nx}×${CONFIG_2D.ny} cells (${w}×${h} m)`;
  } else {
    const w = (CONFIG_3D.dx * CONFIG_3D.nx).toFixed(1);
    const h = (CONFIG_3D.dy * CONFIG_3D.ny).toFixed(1);
    const d = (CONFIG_3D.dz * CONFIG_3D.nz).toFixed(1);
    statusEl.textContent = `Ready — rigid box ${CONFIG_3D.nx}×${CONFIG_3D.ny}×${CONFIG_3D.nz} cells (${w}×${h}×${d} m)`;
  }
}

// 2D result: one image, blitted directly.
function handlePixels(data: {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  freqHz: number;
  lambda: number;
  reqId: number;
}) {
  // Drop a reply that a later request or a mode switch has superseded.
  if (data.reqId !== state.reqId) return;
  if (state.mode !== '2d' || !state.imageData) return;
  state.imageData.data.set(data.data);
  ctx.putImageData(state.imageData, 0, 0);
  drawBoundaries();

  statusEl.textContent = `Driven at ${data.freqHz.toFixed(0)} Hz — steady-state room response`;
  updateDebugInfo(data.lambda);
}

// 3D result: the full volume (nz stacked width×height planes). Cache it and
// blit the current slice; the depth slider then re-slices without re-solving.
function handleVolume(data: {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  depth: number;
  freqHz: number;
  lambda: number;
  reqId: number;
}) {
  // Drop a volume that a later request or a mode switch has superseded.
  if (data.reqId !== state.reqId) return;
  if (state.mode !== '3d') return;
  state.volume = data.data;
  blitSlice();

  statusEl.textContent = `Driven at ${data.freqHz.toFixed(0)} Hz — steady-state box response`;
  updateDebugInfo(data.lambda);
}

// Copy one Z-plane out of the cached volume into the canvas. Plane size and
// count are derived from the live ImageData and volume length rather than
// CONFIG_3D, so a mismatched buffer is caught instead of copying wrong offsets.
function blitSlice() {
  if (!state.imageData || !state.volume) return;
  const planeLen = state.imageData.width * state.imageData.height * 4;
  const planes = Math.floor(state.volume.length / planeLen);
  if (planes < 1) return;
  const z = Math.min(planes - 1, Math.max(0, state.slice));
  const start = z * planeLen;
  state.imageData.data.set(state.volume.subarray(start, start + planeLen));
  ctx.putImageData(state.imageData, 0, 0);
  drawBoundaries();
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
    if (state.mode === '2d') {
      lines.push(`src = (${state.source.sx.toFixed(0)}, ${state.source.sy.toFixed(0)})`);
    } else {
      lines.push(`src = (${state.source.sx.toFixed(0)}, ${state.source.sy.toFixed(0)}, ${state.source.sz.toFixed(0)})`);
    }
  }
  if (state.mode === '3d') {
    const zMetres = (state.slice * CONFIG_3D.dz).toFixed(2);
    lines.push(`z = ${state.slice}/${CONFIG_3D.nz - 1} (${zMetres} m)`);
  }
  debugEl.textContent = lines.join(' | ');
}

// Ask the worker for a fresh steady-state solve at the current source & freq.
function requestSolve() {
  if (!state.isReady || !state.worker || !state.source) return;

  statusEl.textContent = 'Solving…';

  // Keep at most one solve in flight. If the worker is still busy (a 3D solve
  // can take a while), just note that a fresh solve is wanted; the reply
  // handler will re-fire requestSolve with the current source & frequency.
  if (state.busy) {
    state.pending = true;
    return;
  }
  state.busy = true;

  const reqId = ++state.reqId;

  if (state.mode === '2d') {
    state.worker.postMessage({
      type: 'solve',
      sx: state.source.sx,
      sy: state.source.sy,
      freqHz: state.freqHz,
      reqId,
    });
    return;
  }

  state.worker.postMessage({
    type: 'solve3d',
    nx: CONFIG_3D.nx,
    ny: CONFIG_3D.ny,
    nz: CONFIG_3D.nz,
    dx: CONFIG_3D.dx,
    dy: CONFIG_3D.dy,
    dz: CONFIG_3D.dz,
    bcX: CONFIG_3D.bcX,
    bcY: CONFIG_3D.bcY,
    bcZ: CONFIG_3D.bcZ,
    sx: state.source.sx,
    sy: state.source.sy,
    sz: state.source.sz,
    freqHz: state.freqHz,
    reqId,
  });
}

function setMode(mode: Mode) {
  if (mode === state.mode) return;
  state.mode = mode;

  // A new geometry invalidates any cached field and the source placement, and
  // bumps reqId so a solve still in flight for the old mode is dropped on reply.
  state.source = null;
  state.volume = null;
  state.reqId++;
  // Drop any coalesced request queued for the old geometry. `busy` is left as
  // is: if a solve is genuinely still running in the worker, its reply clears
  // the guard (and, with source now null, fires no stale re-solve).
  state.pending = false;

  mode2dBtn.classList.toggle('active', mode === '2d');
  mode3dBtn.classList.toggle('active', mode === '3d');
  depthControl.classList.toggle('active', mode === '3d');

  resizeCanvas();
  updateStatusReady();
  hintEl.textContent =
    mode === '3d'
      ? 'Click to place the source in this slice; drag the depth slider to move through Z'
      : 'Click to place the driving source, then sweep the frequency slider';
  updateDebugInfo();
}

// Event handlers.
canvas.addEventListener('click', (e) => {
  if (!state.isReady || !state.worker) return;

  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;
  // Clamp to [0, n-1]: a click on the right/bottom edge gives x or y == 1,
  // which would otherwise place the source center just outside the grid.
  const sx = Math.min(gridW() - 1, Math.max(0, x * gridW()));
  const sy = Math.min(gridH() - 1, Math.max(0, y * gridH()));
  // In 3D the source lands on the currently displayed slice.
  const sz = state.mode === '3d' ? state.slice : 0;
  state.source = { sx, sy, sz };

  requestSolve();
});

freqSlider.addEventListener('input', () => {
  state.freqHz = parseFloat(freqSlider.value);
  freqDisplay.textContent = state.freqHz.toFixed(0) + ' Hz';
  updateDebugInfo();
  requestSolve();
});

// Depth slider only re-slices the cached volume — no solve. If no volume exists
// yet (no source placed), it just updates the readout.
depthSlider.addEventListener('input', () => {
  state.slice = parseInt(depthSlider.value, 10);
  depthDisplay.textContent = (state.slice * CONFIG_3D.dz).toFixed(1) + ' m';
  blitSlice();
  updateDebugInfo();
});

mode2dBtn.addEventListener('click', () => setMode('2d'));
mode3dBtn.addEventListener('click', () => setMode('3d'));

window.addEventListener('resize', resizeCanvas);

// Initialize.
freqSlider.min = String(FREQ.fMin);
freqSlider.max = String(FREQ.fMax);
freqSlider.value = String(FREQ.fDefault);
freqDisplay.textContent = FREQ.fDefault.toFixed(0) + ' Hz';

depthSlider.min = '0';
depthSlider.max = String(CONFIG_3D.nz - 1);
depthSlider.value = String(state.slice);
depthDisplay.textContent = (state.slice * CONFIG_3D.dz).toFixed(1) + ' m';

resizeCanvas();
initWorker();
