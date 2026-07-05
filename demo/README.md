# Acoustic Room Modes Demo

A browser demonstration of the **driven acoustic Helmholtz equation** on a
rigid-walled rectangular room, using the algo-pde complex Helmholtz solver
compiled to WebAssembly.

Click to place a harmonic source, then sweep the drive frequency to watch the
room's standing-wave (modal) pressure pattern form and shift. A small complex
damping term keeps the field finite when the drive frequency lands on a room
mode.

## What it shows

- **Driven Helmholtz, not wave propagation.** For a source at angular frequency
  ω and sound speed c (343 m/s), with wavenumber `k = ω/c`, we solve the
  steady-state response `∇²p + k²p = -s`, i.e. `(−k² − Δ)p = s`.
- **Room modes.** Rigid (Neumann) walls give a discrete set of standing-wave
  modes. As you sweep frequency, the field peaks and reshapes as `k²` sweeps
  past each modal eigenvalue.
- **Complex damping.** The solve uses `alpha = −k²(1 − iη)` (`η ≈ 0.03`). The
  imaginary shift keeps `|α + λ|` bounded away from zero, so a mode driven at
  resonance yields a finite field instead of blowing up.
- **Click-to-place-source**: a narrow Gaussian source at the clicked point.
- **2D room or 3D box.** The **2D room / 3D slices / 3D volume** toggle switches
  between a 256×192 rectangular room and a 96×72×48 box, both with rigid
  (Neumann) walls. The same complex Helmholtz solver runs at `dim = 2` or
  `dim = 3`. The two 3D views share one solve and differ only in how the volume
  is drawn.

### The 3D case

In 3D the solver returns the **entire volume** in a single solve (stacked
`nz` planes of RGBA, normalized once over the whole box so slices share a
colour scale). Two views present that one volume:

- **3D slices** — a **Z-slice** slider scrubs through the volume entirely
  client-side; moving it re-blits a cached plane with no extra solve. Clicking
  places the source on the currently displayed slice.
- **3D volume** — a WebGL2 ray-marched render of the whole box as a rotatable
  translucent glow. Antinodes glow red/blue while nodes stay transparent, so the
  full standing-wave shape reads at once. Drag to orbit, scroll to zoom, and use
  the **density** slider to fade the field. The render needs no extra solver
  data: the diverging colour map is invertible, so a voxel's amplitude is
  `1 − min(r,g,b)` and its sign is the red/blue side. The cached RGBA is
  transcoded to a signed scalar field, uploaded to the GPU as a single-channel
  3D texture, and the colour map is re-applied in the shader — so `LINEAR`
  filtering interpolates the field (transparent through a node) rather than the
  encoded colour (which would blend adjacent lobes to opaque purple). It falls
  back gracefully (the button is disabled) if WebGL2 is unavailable.

Only a change of frequency or source position triggers a new solve; switching
between the two 3D views does not.

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
│  │  UI    │ │ ← click (source), frequency slider, canvas blit
│  │ Thread │ │
│  └────┬───┘ │
│       │     │
│  ┌────┴────┐│
│  │ Worker  ││ ← loads WASM, calls the solver, posts RGBA frames
│  │ Thread  ││
│  └────┬────┘│
│       │     │
│  ┌────┴────┐│
│  │  WASM   ││ ← complex Helmholtz solve + colormap (Go)
│  │ Module  ││
│  └─────────┘│
└─────────────┘
```

### Solve pipeline

For each click / frequency change:

1. **Assemble the drive.** `k = 2πf / c`, `alpha = complex(−k², k²·η)`.
2. **Build the source.** A narrow Gaussian bump at the clicked grid cell.
3. **Solve.** `NewComplexHelmholtzPlan` + `SolveComplex` return the complex
   steady-state field `p(x,y)`. Plans are cached by
   `(nx, ny, dx, dy, bc, alpha)` and reused across clicks at the same
   frequency — the transforms and eigenvalues are the expensive part.
4. **Colormap in Go.** `Re(p)` is mapped to a symmetric blue-white-red diverging
   colormap and returned as an RGBA byte buffer via `js.CopyBytesToJS` (one bulk
   copy, no per-element writes).
5. **Blit.** The worker transfers the buffer to the UI thread, which draws it to
   the canvas.

There is no time-domain animation: each frame is the steady-state field for the
current source and frequency.

### Technical Details

- **Grid**: 256×192 cells, 0.05 m spacing (12.8 m × 9.6 m room)
- **Boundary conditions**: Neumann (rigid walls)
- **Frequency range**: 40–600 Hz (slider)
- **Damping**: `η = 0.03` complex shift

## Files

- `index.html` — HTML shell with canvas and frequency slider
- `main.ts` — UI thread (canvas, click handling, frequency control)
- `sim.worker.ts` — Web Worker (WASM loading, solve calls, buffer transfer)
- `vite.config.ts` — Vite bundler configuration (`base: './'`)
- `public/` — static assets (WASM module, Go runtime) emitted at the dist root

## Deployment notes

The demo is built with Vite `base: './'` so it works under a subpath such as
`https://<username>.github.io/algo-pde/`. WASM asset URLs are resolved against
the HTML document (`document.baseURI`) in `main.ts` and passed to the worker, so
they resolve correctly under the Pages subpath (the worker bundle lives in
`assets/` while the WASM files land at the dist root). GitHub Pages deployment is
handled by `.github/workflows/deploy-demo.yml`.

### Local subpath test

```bash
just demo-build
mkdir -p /tmp/pages && ln -s "$PWD/dist" /tmp/pages/algo-pde
npx --yes http-server /tmp/pages -p 8099
# open http://localhost:8099/algo-pde/
```

## Browser Compatibility

Requires a modern browser with WebAssembly, Web Workers, and the Canvas 2D API
(Chrome 120+, Firefox 120+, Safari 17+).

## Troubleshooting

### WASM fails to load

Check the browser console. Ensure `demo/public/acoustics.wasm` and
`wasm_exec.js` exist (run `just wasm`) and that the server sends
`Content-Type: application/wasm` for `.wasm` (Vite handles this in dev).

### No field after click

- Verify the status line shows "Ready …"; if it shows a WASM error the runtime
  failed to initialize (the worker rejects after a 15 s readiness timeout).
- Try a different frequency — some frequencies sit between modes and produce a
  low-amplitude field.

## License

MIT (same as the parent project).
