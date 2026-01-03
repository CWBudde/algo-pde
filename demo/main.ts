// Main UI thread - Acoustic Wave Propagation Demo

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d', { alpha: false })!;
const statusEl = document.querySelector('#overlay .status') as HTMLDivElement;
const fpsEl = document.querySelector('#overlay .fps') as HTMLDivElement;
const hintEl = document.querySelector('#overlay .hint') as HTMLDivElement;
const debugEl = document.querySelector('#overlay .debug') as HTMLDivElement;
const controlsEl = document.getElementById('controls') as HTMLDivElement;
const playPauseBtn = document.getElementById('playPauseBtn') as HTMLButtonElement;
const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
const timeSlider = document.getElementById('timeSlider') as HTMLInputElement;
const timeDisplay = document.getElementById('timeDisplay') as HTMLSpanElement;
const durationDisplay = document.getElementById('durationDisplay') as HTMLSpanElement;

// Configuration
const CONFIG = {
  nx: 256,  // Grid width (12.8m at 0.05m/cell)
  ny: 192,  // Grid height (9.6m) - 4:3 aspect ratio
  dx: 0.05, // 0.05m per cell
  dy: 0.05,
  bcX: 2,   // Neumann (rigid walls)
  bcY: 2,
  nBins: 16,
  fMin: 80,
  fMax: 600,
  duration: 2.0, // Animation duration in seconds (shorter for better responsiveness)
};

// State management
interface AppState {
  worker: Worker | null;
  isReady: boolean;
  isAnimating: boolean;
  isPaused: boolean;
  currentTime: number;
  startRealTime: number; // Performance.now() when animation started
  imageData: ImageData | null;
  animationFrameId: number | null;
  freqRange: [number, number] | null;
}

const state: AppState = {
  worker: null,
  isReady: false,
  isAnimating: false,
  isPaused: false,
  currentTime: 0,
  startRealTime: 0,
  imageData: null,
  animationFrameId: null,
  freqRange: null,
};

// FPS tracking
let frameCount = 0;
let lastFpsUpdate = performance.now();

// Setup canvas
function resizeCanvas() {
  // Set canvas internal resolution to grid size
  canvas.width = CONFIG.nx;
  canvas.height = CONFIG.ny;

  // Calculate display size maintaining aspect ratio
  const aspectRatio = CONFIG.nx / CONFIG.ny;
  const windowAspect = window.innerWidth / window.innerHeight;

  let displayWidth: number;
  let displayHeight: number;

  if (windowAspect > aspectRatio) {
    // Window is wider - fit to height
    displayHeight = window.innerHeight * 0.9; // Leave some margin
    displayWidth = displayHeight * aspectRatio;
  } else {
    // Window is taller - fit to width
    displayWidth = window.innerWidth * 0.9; // Leave some margin
    displayHeight = displayWidth / aspectRatio;
  }

  canvas.style.width = displayWidth + 'px';
  canvas.style.height = displayHeight + 'px';

  state.imageData = ctx.createImageData(CONFIG.nx, CONFIG.ny);

  // Fill with white initially
  clearCanvas();
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

// Generate logarithmically-spaced frequencies
function generateFrequencies(fMin: number, fMax: number, n: number): number[] {
  const logMin = Math.log(fMin);
  const logMax = Math.log(fMax);
  const freqs: number[] = [];

  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const logF = logMin + t * (logMax - logMin);
    freqs.push(Math.exp(logF));
  }

  return freqs;
}

// Initialize worker
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
      case 'progress':
        handleProgress(data);
        break;
      case 'computed':
        handleComputed(data);
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

  // Initialize plan
  state.worker.postMessage({
    type: 'init',
    nx: CONFIG.nx,
    ny: CONFIG.ny,
    dx: CONFIG.dx,
    dy: CONFIG.dy,
    bcX: CONFIG.bcX,
    bcY: CONFIG.bcY,
  });
}

function handleReady(data: { planID: string; nx: number; ny: number }) {
  state.isReady = true;
  statusEl.textContent = `Ready (${data.nx}×${data.ny}, ${CONFIG.dx * data.nx}×${CONFIG.dy * data.ny}m)`;
  hintEl.textContent = 'Click anywhere to create a wave source';
  updateDebugInfo();
}

function handleProgress(data: { current: number; total: number; frequency: number }) {
  const percent = Math.round((data.current / data.total) * 100);
  statusEl.textContent = `Computing ${data.current}/${data.total} (${percent}%) - ${data.frequency.toFixed(0)} Hz`;
}

function handleComputed(data: { nFreqs: number; freqRange: [number, number] }) {
  state.freqRange = data.freqRange;
  statusEl.textContent = `Animating (${data.nFreqs} modes: ${data.freqRange[0].toFixed(0)}-${data.freqRange[1].toFixed(0)} Hz)`;
  startAnimation();
}

function handlePixels(data: { data: Uint8ClampedArray; width: number; height: number }) {
  if (!state.imageData) {
    console.error('No imageData available');
    return;
  }

  state.imageData.data.set(data.data);
  ctx.putImageData(state.imageData, 0, 0);

  // Redraw boundaries on top
  drawBoundaries();

  // Update FPS
  frameCount++;
  const now = performance.now();
  if (now - lastFpsUpdate > 1000) {
    const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
    fpsEl.textContent = `${fps} FPS`;
    frameCount = 0;
    lastFpsUpdate = now;
  }
}

function handleError(data: { message: string }) {
  console.error('Worker error:', data.message);
  statusEl.textContent = `Error: ${data.message}`;
  stopAnimation();
}

// Update debug information
function updateDebugInfo() {
  if (!state.isAnimating) {
    debugEl.textContent = '';
    return;
  }

  const lines = [
    `t = ${state.currentTime.toFixed(3)}s`,
  ];

  if (state.freqRange) {
    lines.push(`f = ${state.freqRange[0].toFixed(0)}-${state.freqRange[1].toFixed(0)} Hz`);
  }

  debugEl.textContent = lines.join(' | ');
}

// Animation control
function startAnimation() {
  if (state.isAnimating) {
    return;
  }

  state.isAnimating = true;
  state.isPaused = false;
  state.currentTime = 0;
  state.startRealTime = performance.now();

  playPauseBtn.textContent = 'Pause';
  controlsEl.classList.add('active');

  function animate() {
    if (!state.isAnimating) {
      return;
    }

    if (!state.isPaused && !isScrubbing) {
      // Calculate elapsed time in seconds
      const elapsed = (performance.now() - state.startRealTime) / 1000;
      state.currentTime = Math.min(elapsed, CONFIG.duration);

      // Update slider and display
      const sliderValue = (state.currentTime / CONFIG.duration) * 1000;
      timeSlider.value = String(Math.round(sliderValue));
      timeDisplay.textContent = state.currentTime.toFixed(2) + 's';

      // Loop animation
      if (state.currentTime >= CONFIG.duration) {
        state.currentTime = 0;
        state.startRealTime = performance.now();
      }
    }

    // Request frame from worker
    requestFrame();

    // Update debug info
    updateDebugInfo();

    state.animationFrameId = requestAnimationFrame(animate);
  }

  animate();
}

function stopAnimation() {
  state.isAnimating = false;
  state.isPaused = false;
  state.currentTime = 0;

  if (state.animationFrameId !== null) {
    cancelAnimationFrame(state.animationFrameId);
    state.animationFrameId = null;
  }

  state.worker?.postMessage({ type: 'stop' });
  controlsEl.classList.remove('active');
  updateDebugInfo();
}

function togglePause() {
  if (!state.isAnimating) {
    return;
  }

  state.isPaused = !state.isPaused;
  playPauseBtn.textContent = state.isPaused ? 'Play' : 'Pause';

  if (!state.isPaused) {
    // Resume: adjust startRealTime to account for current time
    state.startRealTime = performance.now() - state.currentTime * 1000;
  } else {
    // When pausing, immediately update the display
    requestFrame();
    updateDebugInfo();
  }
}

function seekToTime(time: number) {
  state.currentTime = time;
  timeDisplay.textContent = state.currentTime.toFixed(2) + 's';

  // Always update slider to match
  const sliderValue = (state.currentTime / CONFIG.duration) * 1000;
  timeSlider.value = String(Math.round(sliderValue));

  if (!state.isPaused) {
    // Adjust startRealTime if playing
    state.startRealTime = performance.now() - state.currentTime * 1000;
  }

  // Request immediate frame update
  requestFrame();
  updateDebugInfo();
}

function requestFrame() {
  state.worker?.postMessage({ type: 'frame', t: state.currentTime });
}

// Event handlers
canvas.addEventListener('click', (e) => {
  if (!state.isReady || !state.worker) return;

  // Stop current animation
  stopAnimation();

  // Convert click to grid coordinates
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;

  const sx = x * CONFIG.nx;
  const sy = y * CONFIG.ny;

  console.log(`Click at grid (${sx.toFixed(1)}, ${sy.toFixed(1)})`);

  // Generate frequencies
  const frequencies = generateFrequencies(CONFIG.fMin, CONFIG.fMax, CONFIG.nBins);

  // Send ping to worker
  statusEl.textContent = 'Computing...';
  state.worker.postMessage({
    type: 'ping',
    sx,
    sy,
    frequencies,
  });
});

playPauseBtn.addEventListener('click', () => {
  togglePause();
});

resetBtn.addEventListener('click', () => {
  if (state.isAnimating) {
    state.currentTime = 0;
    state.startRealTime = performance.now();
    seekToTime(0);
  }
});

// Track if user is actively scrubbing
let isScrubbing = false;

timeSlider.addEventListener('mousedown', () => {
  if (!state.isAnimating) return;
  isScrubbing = true;

  // Pause animation when user starts scrubbing
  if (!state.isPaused) {
    state.isPaused = true;
    playPauseBtn.textContent = 'Play';
  }
});

timeSlider.addEventListener('input', () => {
  if (!state.isAnimating) return;

  const sliderValue = parseInt(timeSlider.value);
  const time = (sliderValue / 1000) * CONFIG.duration;

  // Update time directly without going through seekToTime to avoid feedback loop
  state.currentTime = time;
  timeDisplay.textContent = state.currentTime.toFixed(2) + 's';

  // Request immediate frame update for scrubbing
  requestFrame();
  updateDebugInfo();
});

timeSlider.addEventListener('mouseup', () => {
  isScrubbing = false;
});

timeSlider.addEventListener('touchstart', () => {
  if (!state.isAnimating) return;
  isScrubbing = true;

  // Pause animation when user starts scrubbing
  if (!state.isPaused) {
    state.isPaused = true;
    playPauseBtn.textContent = 'Play';
  }
});

timeSlider.addEventListener('touchend', () => {
  isScrubbing = false;
});

// Handle window resize
window.addEventListener('resize', resizeCanvas);

// Initialize
resizeCanvas();
initWorker();
durationDisplay.textContent = CONFIG.duration.toFixed(1) + 's';
