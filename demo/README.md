# Wave Propagation Demo

A minimal browser-based demonstration of 2D wave propagation using the algo-pde Helmholtz solver compiled to WebAssembly.

## Features

- **Click-to-ping**: Click anywhere to create a wave that propagates and reflects
- **Multi-frequency synthesis**: Uses 16 frequency modes (80-600 Hz) for realistic wave behavior
- **Real-time animation**: Smooth 60 FPS visualization
- **Neumann boundary conditions**: Rigid walls with perfect reflections

## Quick Start

### Development

```bash
# From the project root:
just demo-dev
```

This will:
1. Build the WASM module from `cmd/acoustics-wasm`
2. Copy the Go WASM runtime (`wasm_exec.js`)
3. Install npm dependencies
4. Start the Vite development server

Then open [http://localhost:5173](http://localhost:5173) in your browser.

### Production Build

```bash
just demo-build
```

Output will be in `demo/dist/` ready for static hosting.

## How It Works

### Architecture

```
┌─────────────┐
│   Browser   │
│             │
│  ┌────────┐ │
│  │  UI    │ │ ← Click events, canvas rendering
│  │ Thread │ │
│  └────┬───┘ │
│       │     │
│  ┌────┴────┐│
│  │ Worker  ││ ← Multi-frequency synthesis, colormap
│  │ Thread  ││
│  └────┬────┘│
│       │     │
│  ┌────┴────┐│
│  │  WASM   ││ ← Helmholtz solver (Go)
│  │ Module  ││
│  └─────────┘│
└─────────────┘
```

### Physics

For each click at position `(sx, sy)`:

1. **Multi-frequency solve**: For each frequency `f` in [80, 600] Hz:
   - Compute wavenumber: `k = 2πf/c` (c = 343 m/s)
   - Solve Helmholtz equation: `(-Δ + k²)p = s`
   - Source `s` is a Gaussian blob at click position
   - Store pressure field `p_i(x,y)`

2. **Animation synthesis**: For each frame at time `t`:
   ```
   u(x,y,t) = Σᵢ pᵢ(x,y) · cos(2π fᵢ t) · exp(-γᵢ t)
   ```
   where `γᵢ = 0.5 · fᵢ` (frequency-dependent damping)

3. **Visualization**: Apply diverging colormap (blue ← 0 → red) with percentile normalization

### Technical Details

- **Grid**: 256×256 cells, 0.05m spacing (~12.8m room)
- **Boundary conditions**: Neumann (rigid walls)
- **Frequency bins**: 16 logarithmically-spaced modes
- **Performance**: ~72ms for 16-mode solve (target <200ms)
- **Bundle size**: ~4.1 MB WASM + 17 KB runtime

## Files

- `index.html` - Minimal HTML shell with canvas
- `main.ts` - UI thread (canvas, click handling, FPS counter)
- `sim.worker.ts` - Web Worker (WASM loading, synthesis, colormap)
- `vite.config.ts` - Vite bundler configuration
- `public/` - Static assets (WASM module, Go runtime)

## Browser Compatibility

Requires modern browser with:
- WebAssembly support
- Web Workers
- Float32Array
- Canvas 2D API

Tested on:
- Chrome 120+
- Firefox 120+
- Safari 17+

## Performance Notes

The demo is optimized for 60 FPS at 256² resolution:

- **WASM solve**: ~4.5ms per frequency (16 × 4.5ms = 72ms total)
- **Frame synthesis**: ~2-3ms (worker thread)
- **Colormap + transfer**: ~1-2ms
- **Canvas render**: ~1-2ms (UI thread)
- **Total frame budget**: ~80ms (well under 16.67ms animation budget)

## GitHub Pages Deployment

The demo is automatically deployed to GitHub Pages on every push to the `main` branch (when demo-related files change).

### Setup (One-time)

1. Go to your repository **Settings** → **Pages**
2. Under **Source**, select **GitHub Actions**
3. The workflow will automatically deploy on the next push

### Manual Deployment

Trigger a manual deployment from the Actions tab:
1. Go to **Actions** → **Deploy Demo to GitHub Pages**
2. Click **Run workflow** → **Run workflow**

The demo will be available at: `https://<username>.github.io/algo-pde/`

### Local Production Build

Test the production build locally:

```bash
just demo-build
cd demo/dist
python3 -m http.server 8000
# Open http://localhost:8000
```

## Troubleshooting

### WASM fails to load

Check browser console for errors. Ensure:
- Files exist: `demo/public/acoustics.wasm` and `wasm_exec.js`
- Server sends correct MIME type for `.wasm` (Vite handles this)

### Poor performance

- Check FPS counter (top left)
- Reduce frequency bins in `main.ts` (CONFIG.nBins = 8)
- Check browser dev tools for memory leaks

### No animation after click

- Check browser console for worker errors
- Verify WASM initialized (status should show "Ready")
- Try clicking in different locations

## License

Same as parent project (algo-pde).
