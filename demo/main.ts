// Main UI thread — Acoustic Room Modes demo (2D room + 3D box).
//
// Solves the driven acoustic Helmholtz equation on a rigid-walled rectangular
// room (2D) or box (3D) and renders the steady-state pressure field. Click
// places the driving source; the frequency slider sweeps the drive frequency
// through the room's standing-wave modes. A small damping term keeps the field
// finite on resonance.
//
// In 2D the solver returns one image per solve. In 3D it returns the whole
// volume as stacked Z-planes in a single solve. Two 3D views share that one
// solve: "3D slices" scrubs through Z-planes on a 2D canvas (depth slider, no
// re-solve), and "3D volume" ray-marches the whole volume as a rotatable
// translucent glow on a WebGL canvas. Only a frequency or source change
// triggers a new solve; switching between the two 3D views does not.

import { VolumeRenderer } from './volume';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d', { alpha: false })!;
const glCanvas = document.getElementById('glCanvas') as HTMLCanvasElement;
const statusEl = document.querySelector('#overlay .status') as HTMLDivElement;
const hintEl = document.querySelector('#overlay .hint') as HTMLDivElement;
const debugEl = document.querySelector('#overlay .debug') as HTMLDivElement;
const controlsEl = document.getElementById('controls') as HTMLDivElement;
const freqSlider = document.getElementById('freqSlider') as HTMLInputElement;
const freqDisplay = document.getElementById('freqDisplay') as HTMLSpanElement;
const mode2dBtn = document.getElementById('mode2dBtn') as HTMLButtonElement;
const mode3dSliceBtn = document.getElementById('mode3dSliceBtn') as HTMLButtonElement;
const mode3dVolBtn = document.getElementById('mode3dVolBtn') as HTMLButtonElement;
const depthControl = document.getElementById('depthControl') as HTMLDivElement;
const depthSlider = document.getElementById('depthSlider') as HTMLInputElement;
const depthDisplay = document.getElementById('depthDisplay') as HTMLSpanElement;
const densityControl = document.getElementById('densityControl') as HTMLDivElement;
const densitySlider = document.getElementById('densitySlider') as HTMLInputElement;
const densityDisplay = document.getElementById('densityDisplay') as HTMLSpanElement;

// The solve geometry is 2D or 3D; the view chooses how that geometry is shown.
// Both 3D views run the identical volume solve, so `mode` (derived from `view`)
// keys everything solver-related while `view` keys the display.
type Mode = '2d' | '3d';
type View = '2d' | '3d-slice' | '3d-volume';

function modeForView(view: View): Mode {
  return view === '2d' ? '2d' : '3d';
}

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
  view: View;
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
  // 3D volume view: the WebGL ray-march renderer (null if WebGL2 is
  // unavailable), the orbit camera, and the opacity multiplier.
  renderer: VolumeRenderer | null;
  camera: { azimuthDeg: number; elevationDeg: number; distance: number };
  density: number;
  // A single scheduled animation frame; render-on-demand rather than a
  // continuous loop so an idle volume view costs nothing.
  renderPending: boolean;
}

const state: AppState = {
  worker: null,
  isReady: false,
  view: '2d',
  mode: '2d',
  imageData: null,
  source: null,
  freqHz: FREQ.fDefault,
  volume: null,
  slice: Math.floor(CONFIG_3D.nz / 2),
  reqId: 0,
  busy: false,
  pending: false,
  renderer: null,
  camera: { azimuthDeg: -35, elevationDeg: 22, distance: 2.6 },
  density: 1.0,
  renderPending: false,
};

// Opacity gamma for the volume render: < 1 lifts faint lobes so nodes read as
// gaps rather than the whole box washing out. Fixed (not user-facing).
const VOLUME_GAMMA = 0.8;

// Active-mode grid dimensions used for canvas sizing and coordinate mapping.
function gridW(): number {
  return state.mode === '2d' ? CONFIG_2D.nx : CONFIG_3D.nx;
}

function gridH(): number {
  return state.mode === '2d' ? CONFIG_2D.ny : CONFIG_3D.ny;
}

// Grid dimensions for a given mode (not necessarily the active one), used to
// remap the source position when switching modes.
function dimsFor(mode: Mode): { w: number; h: number } {
  return mode === '2d' ? { w: CONFIG_2D.nx, h: CONFIG_2D.ny } : { w: CONFIG_3D.nx, h: CONFIG_3D.ny };
}

// A default 3D source at the box centre (on the current Z-slice). The volume
// view has no click-to-place affordance — the GL canvas drag orbits the camera
// — so entering it with nothing placed seeds this so the box isn't blank.
function centeredSource3D(): { sx: number; sy: number; sz: number } {
  return { sx: (CONFIG_3D.nx - 1) / 2, sy: (CONFIG_3D.ny - 1) / 2, sz: state.slice };
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
    // A hard worker failure never posts an 'error' message, so release the
    // in-flight guard here too; otherwise busy stays set and every later solve
    // just coalesces forever.
    state.busy = false;
    state.pending = false;
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
  // Route the fresh volume to whichever 3D view is active. The volume slider
  // view blits a Z-plane; the volume render uploads the whole box to the GPU.
  if (state.view === '3d-volume') {
    uploadVolume();
    requestRender();
  } else {
    blitSlice();
  }

  statusEl.textContent = `Driven at ${data.freqHz.toFixed(0)} Hz — steady-state box response`;
  updateDebugInfo(data.lambda);
}

// ---- 3D volume render (WebGL) ----------------------------------------------

// Create the WebGL renderer once. If WebGL2 is unavailable the volume view is
// disabled (the button is dimmed) and everything else keeps working.
function initRenderer() {
  try {
    state.renderer = new VolumeRenderer(glCanvas);
  } catch (err) {
    console.warn('3D volume render unavailable:', err);
    state.renderer = null;
    mode3dVolBtn.disabled = true;
    mode3dVolBtn.title = 'WebGL2 not available in this browser';
  }
}

// Size the WebGL drawing buffer to its CSS box (× devicePixelRatio, capped) so
// the render stays crisp without over-allocating on hi-DPI displays.
function resizeGL() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = Math.floor(window.innerWidth * 0.9);
  const cssH = Math.floor(window.innerHeight * 0.9);
  glCanvas.style.width = cssW + 'px';
  glCanvas.style.height = cssH + 'px';
  glCanvas.width = Math.max(1, Math.floor(cssW * dpr));
  glCanvas.height = Math.max(1, Math.floor(cssH * dpr));
  if (state.renderer) state.renderer.resize(glCanvas.width, glCanvas.height);
}

// Push the cached RGBA volume to the GPU as a 3D texture, with the box's
// physical extents so proportions are right.
function uploadVolume() {
  if (!state.renderer || !state.volume) return;
  state.renderer.setVolume(
    state.volume,
    CONFIG_3D.nx,
    CONFIG_3D.ny,
    CONFIG_3D.nz,
    CONFIG_3D.nx * CONFIG_3D.dx,
    CONFIG_3D.ny * CONFIG_3D.dy,
    CONFIG_3D.nz * CONFIG_3D.dz,
  );
}

// Schedule one animation frame. Render-on-demand: a camera drag or a fresh
// volume calls this; an untouched volume view draws nothing further.
function requestRender() {
  if (state.renderPending || state.view !== '3d-volume' || !state.renderer) return;
  state.renderPending = true;
  requestAnimationFrame(() => {
    state.renderPending = false;
    if (state.view !== '3d-volume' || !state.renderer) return;
    state.renderer.render({
      azimuthDeg: state.camera.azimuthDeg,
      elevationDeg: state.camera.elevationDeg,
      distance: state.camera.distance,
      density: state.density,
      gamma: VOLUME_GAMMA,
    });
  });
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
  if (state.view === '3d-slice') {
    const zMetres = (state.slice * CONFIG_3D.dz).toFixed(2);
    lines.push(`z = ${state.slice}/${CONFIG_3D.nz - 1} (${zMetres} m)`);
  } else if (state.view === '3d-volume') {
    lines.push(`density = ${state.density.toFixed(1)}×`);
  }
  debugEl.textContent = lines.join(' | ');
}

// Hint text for the active view.
function updateHint() {
  switch (state.view) {
    case '2d':
      hintEl.textContent = 'Click to place the driving source, then sweep the frequency slider';
      break;
    case '3d-slice':
      hintEl.textContent = 'Click to place the source in this slice; drag the depth slider to move through Z';
      break;
    case '3d-volume':
      hintEl.textContent = 'Drag to orbit the box; density slider fades the field; sweep frequency to change the mode';
      break;
  }
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
    // Bump reqId so the in-flight solve's reply is dropped as stale rather than
    // briefly rendered (with its now-obsolete frequency/lambda) before the
    // coalesced solve replaces it.
    state.reqId++;
    return;
  }
  state.busy = true;

  const reqId = ++state.reqId;

  const message =
    state.mode === '2d'
      ? {
          type: 'solve',
          sx: state.source.sx,
          sy: state.source.sy,
          freqHz: state.freqHz,
          reqId,
        }
      : {
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
        };

  try {
    state.worker.postMessage(message);
  } catch (err) {
    // The worker is gone (terminated/crashed): release the guard so the UI
    // isn't stuck coalescing forever, and surface the failure.
    state.busy = false;
    state.pending = false;
    handleError({ message: err instanceof Error ? err.message : String(err) });
  }
}

function setView(view: View) {
  if (view === state.view) return;
  const prevMode = state.mode;
  const newMode = modeForView(view);
  const geometryChanged = newMode !== prevMode;

  state.view = view;
  state.mode = newMode;

  // Button, control, and canvas visibility for the new view.
  mode2dBtn.classList.toggle('active', view === '2d');
  mode3dSliceBtn.classList.toggle('active', view === '3d-slice');
  mode3dVolBtn.classList.toggle('active', view === '3d-volume');
  depthControl.classList.toggle('active', view === '3d-slice');
  densityControl.classList.toggle('active', view === '3d-volume');
  // The 2D canvas backs the room and the slice viewer; the WebGL canvas backs
  // the volume render. Only one is visible at a time.
  canvas.style.display = view === '3d-volume' ? 'none' : 'block';
  glCanvas.style.display = view === '3d-volume' ? 'block' : 'none';

  // Switching between the two 3D views is pure display: same geometry, same
  // cached volume, no re-solve — just re-target it.
  if (!geometryChanged) {
    if (view === '3d-volume') {
      resizeGL();
      // Seed a default source if none was ever placed, so the volume view is
      // never a blank box with no way to start a solve.
      if (!state.source) state.source = centeredSource3D();
      if (state.volume) {
        uploadVolume();
        requestRender();
      } else {
        // No cached volume (freshly seeded source, or an in-flight/failed
        // solve): kick a solve; handleVolume uploads and renders on reply.
        requestSolve();
      }
    } else {
      // Back to the slice canvas: resizeCanvas rebuilds the ImageData at 3D
      // dimensions and re-blits the cached slice.
      resizeCanvas();
    }
    updateStatusReady();
    updateHint();
    updateDebugInfo();
    return;
  }

  // Geometry change (2D ⇄ 3D). The frequency slider is shared, so the drive
  // frequency carries over untouched. Carry the source over too: remap its
  // position into the new grid (same relative spot in the room face) rather
  // than forcing a fresh click, so the switch immediately re-solves the new
  // geometry at the current source & frequency.
  if (state.source) {
    const from = dimsFor(prevMode);
    const to = dimsFor(newMode);
    // Map cell centres proportionally, then clamp back into [0, n-1].
    const sx = Math.min(to.w - 1, Math.max(0, ((state.source.sx + 0.5) / from.w) * to.w - 0.5));
    const sy = Math.min(to.h - 1, Math.max(0, ((state.source.sy + 0.5) / from.h) * to.h - 0.5));
    // Entering 3D, drop the source onto the current Z-slice; leaving 3D, z is
    // dropped (sz is ignored by the 2D solver).
    const sz = newMode === '3d' ? state.slice : 0;
    state.source = { sx, sy, sz };
  }

  // Entering the volume view with nothing ever placed: seed a default source so
  // it renders a field rather than an empty box (the GL canvas can't place one).
  if (view === '3d-volume' && !state.source) state.source = centeredSource3D();

  // A new geometry invalidates any cached volume, and bumps reqId so a solve
  // still in flight for the old geometry is dropped on reply.
  state.volume = null;
  state.reqId++;
  // Drop any coalesced request queued for the old geometry. `busy` is left as
  // is: if a solve is genuinely still running in the worker, its reply clears
  // the guard, and requestSolve below re-arms one for the new geometry.
  state.pending = false;

  if (view === '3d-volume') resizeGL();
  else resizeCanvas();
  updateStatusReady();
  updateHint();
  updateDebugInfo();

  // Re-solve the new geometry at the carried-over source & frequency. No source
  // yet (nothing ever placed) → requestSolve is a no-op and the status stays
  // "Ready".
  requestSolve();
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

mode2dBtn.addEventListener('click', () => setView('2d'));
mode3dSliceBtn.addEventListener('click', () => setView('3d-slice'));
mode3dVolBtn.addEventListener('click', () => setView('3d-volume'));

// Density slider only fades the volume render — no solve, no re-upload.
densitySlider.addEventListener('input', () => {
  state.density = parseFloat(densitySlider.value);
  densityDisplay.textContent = state.density.toFixed(1) + '×';
  updateDebugInfo();
  requestRender();
});

// Drag on the WebGL canvas orbits the camera. Pointer capture keeps the drag
// alive if the cursor leaves the canvas mid-rotate.
let dragging = false;
let lastX = 0;
let lastY = 0;
glCanvas.addEventListener('pointerdown', (e) => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
  glCanvas.setPointerCapture(e.pointerId);
});
glCanvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;
  state.camera.azimuthDeg -= dx * 0.4;
  // Clamp elevation just shy of the poles so the orbit never gimbal-flips.
  state.camera.elevationDeg = Math.min(89, Math.max(-89, state.camera.elevationDeg + dy * 0.4));
  requestRender();
});
const endDrag = (e: PointerEvent) => {
  if (!dragging) return;
  dragging = false;
  if (glCanvas.hasPointerCapture(e.pointerId)) glCanvas.releasePointerCapture(e.pointerId);
};
glCanvas.addEventListener('pointerup', endDrag);
glCanvas.addEventListener('pointercancel', endDrag);

// Wheel zooms the volume camera (clamped to a sensible range).
glCanvas.addEventListener(
  'wheel',
  (e) => {
    if (state.view !== '3d-volume') return;
    e.preventDefault();
    state.camera.distance = Math.min(6, Math.max(1.3, state.camera.distance * (1 + e.deltaY * 0.001)));
    requestRender();
  },
  { passive: false },
);

window.addEventListener('resize', () => {
  if (state.view === '3d-volume') {
    resizeGL();
    requestRender();
  } else {
    resizeCanvas();
  }
});

// Initialize.
freqSlider.min = String(FREQ.fMin);
freqSlider.max = String(FREQ.fMax);
freqSlider.value = String(FREQ.fDefault);
freqDisplay.textContent = FREQ.fDefault.toFixed(0) + ' Hz';

depthSlider.min = '0';
depthSlider.max = String(CONFIG_3D.nz - 1);
depthSlider.value = String(state.slice);
depthDisplay.textContent = (state.slice * CONFIG_3D.dz).toFixed(1) + ' m';

densitySlider.min = '0.2';
densitySlider.max = '3';
densitySlider.step = '0.1';
densitySlider.value = String(state.density);
densityDisplay.textContent = state.density.toFixed(1) + '×';

resizeCanvas();
initRenderer();
initWorker();
