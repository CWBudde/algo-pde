// Main UI thread

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d', { alpha: false })!;
const statusEl = document.querySelector('#overlay .status') as HTMLDivElement;
const fpsEl = document.querySelector('#overlay .fps') as HTMLDivElement;
const hintEl = document.querySelector('#overlay .hint') as HTMLDivElement;

// Configuration
const CONFIG = {
  nx: 256,
  ny: 256,
  dx: 0.05, // 12.8m / 256 ≈ 0.05m per cell
  dy: 0.05,
  bcX: 2, // Neumann (rigid walls)
  bcY: 2,
  nBins: 16,
  fMin: 80,
  fMax: 600,
};

// State
let worker: Worker | null = null;
let isReady = false;
let isAnimating = false;
let imageData: ImageData | null = null;
let animationFrameId: number | null = null;

// FPS tracking
let frameCount = 0;
let lastFpsUpdate = performance.now();

// Setup canvas
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = CONFIG.nx;
  canvas.height = CONFIG.ny;
  canvas.style.width = window.innerWidth + 'px';
  canvas.style.height = window.innerHeight + 'px';

  imageData = ctx.createImageData(CONFIG.nx, CONFIG.ny);
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
  worker = new Worker(new URL('./sim.worker.ts', import.meta.url), {
    type: 'module',
  });

  worker.onmessage = (e) => {
    const { type, ...data } = e.data;

    switch (type) {
      case 'ready':
        handleReady(data);
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

  worker.onerror = (error) => {
    console.error('Worker error:', error);
    statusEl.textContent = `Worker error: ${error.message}`;
  };

  // Initialize plan
  worker.postMessage({
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
  console.log('WASM ready:', data);
  isReady = true;
  statusEl.textContent = `Ready (${data.nx}×${data.ny})`;
  hintEl.textContent = 'Click anywhere to create a wave';
}

function handleComputed(data: { nFreqs: number }) {
  console.log(`Computed ${data.nFreqs} frequency modes`);
  statusEl.textContent = `Animating (${data.nFreqs} modes)`;
  startAnimation();
}

function handlePixels(data: { data: Uint8ClampedArray; width: number; height: number }) {
  if (!imageData) return;

  imageData.data.set(data.data);
  ctx.putImageData(imageData, 0, 0);

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

// Animation loop
function startAnimation() {
  if (isAnimating) return;
  isAnimating = true;

  function animate() {
    if (!isAnimating) return;

    const t = performance.now() / 1000;
    worker?.postMessage({ type: 'frame', t });

    animationFrameId = requestAnimationFrame(animate);
  }

  animate();
}

function stopAnimation() {
  isAnimating = false;
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  worker?.postMessage({ type: 'stop' });
}

// Click handler
canvas.addEventListener('click', (e) => {
  if (!isReady || !worker) return;

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
  worker.postMessage({
    type: 'ping',
    sx,
    sy,
    frequencies,
  });
});

// Handle window resize
window.addEventListener('resize', resizeCanvas);

// Initialize
resizeCanvas();
initWorker();
