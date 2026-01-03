# algo-pde Implementation Plan

A fast spectral Poisson/Helmholtz solver library for Go, built on top of `algo-fft`.

---

## Phase 1: Grid Package (`grid/`) ✅ COMPLETE

- Shape/stride types, indexing helpers, and row-major stride computation.
- Iterators for lines and planes, plus strided copy utilities.
- Unit tests covering indexing and iterator behavior.

---

## Phase 2: Real-to-Real & Line Transforms (`r2r/`) ✅

- [x] **DST/DCT Core**: DST-I/II and DCT-I/II implemented via FFT embedding with full normalization and correctness tests.
- [x] **Plan API**: `DSTPlan`, `DCTPlan`, and `FFTPlan` for allocation-conscious, axis-wise transforms on N-D grids.
- [x] **Multi-D Support**: `ForwardLines`/`InverseLines` for all transform types with 2D/3D unit tests.
- [x] **Performance**: Verified O(N log N) scaling and optimized buffer management.

---

## Phase 3: Finite Difference Operators (`fd/`) - PARTIAL ✅

### 3.1 Laplacian eigenvalues

- [x] Implement `EigenvaluesPeriodic(n int, h float64) []float64`
  - Formula: λ_m = (2 - 2\*cos(2πm/N)) / h²
- [x] Implement `EigenvaluesDirichlet(n int, h float64) []float64`
  - Formula: λ_m = (2 - 2\*cos(πm/(N+1))) / h²
- [x] Implement `EigenvaluesNeumann(n int, h float64) []float64`
  - Formula: λ_m = (2 - 2\*cos(πm/N)) / h²
- [x] Write tests comparing eigenvalues to explicit matrix eigendecomposition
- [x] Document eigenvalue formulas and grid conventions

### 3.2 Laplacian stencil (for testing/validation)

- [x] Implement `Apply1D(dst, src []float64, h float64, bc BCType)`
- [x] Implement `Apply2D(dst, src []float64, shape Shape, h [2]float64, bc [2]BCType)`
- [x] Implement `Apply3D(...)`
- [x] Write tests verifying Δu matches expected for known u

---

## Phase 4: Periodic Poisson Solver (`poisson/`)

### 4.1 1D Periodic Solver

- [x] Implement `Plan1DPeriodic` struct
- [x] Implement `NewPlan1DPeriodic(nx int, hx float64, opts ...Option) (*Plan1DPeriodic, error)`
- [x] Pre-compute eigenvalues in plan creation
- [x] Pre-allocate FFT plan from algo-fft
- [x] Pre-allocate work buffers
- [x] Implement `Plan.Solve(dst, rhs []float64) error`
- [x] Implement `Plan.SolveInPlace(buf []float64) error`
- [x] Implement zero-mode (mean) handling:
  - [x] Option: RequireMeanZeroRHS (error if not)
  - [x] Option: SubtractMean (auto-subtract before solve)
  - [x] Option: SetSolutionMean (set u's mean to specified value)
- [x] Write unit tests with manufactured solutions
- [x] Write benchmark tests

### 4.2 2D Periodic Solver

- [x] Implement `Plan2DPeriodic` struct
- [x] Implement `NewPlan2DPeriodic(nx, ny int, hx, hy float64, opts ...Option) (*Plan2DPeriodic, error)`
- [x] Use algo-fft's 2D real FFT plans where possible
- [x] Implement eigenvalue division in spectral space
- [x] Implement zero-mode handling (same options as 1D)
- [x] Write manufactured solution tests:
  - [x] u = sin(2πx/Lx) \* sin(2πy/Ly)
  - [x] u = cos(2πx/Lx) \* cos(4πy/Ly) (mixed frequencies)
- [x] Write convergence tests (error vs grid spacing)
- [x] Benchmark: 64², 128², 256², 512², 1024²

### 4.3 3D Periodic Solver

- [x] Implement `Plan3DPeriodic` struct
- [x] Implement `NewPlan3DPeriodic(nx, ny, nz int, hx, hy, hz float64, opts ...Option)`
- [x] Use algo-fft's 3D real FFT plans
- [x] Implement eigenvalue division
- [x] Implement zero-mode handling
- [x] Write manufactured solution tests
- [x] Benchmark: 32³, 64³, 128³

### 4.4 ND Periodic Solver (generic)

- [x] Implement `PlanNDPeriodic` for arbitrary dimensions
- [x] Implement `NewPlanNDPeriodic(shape Shape, h []float64, opts ...Option)`
- [x] Write tests for 4D case (stress test)

---

## Phase 5: Dirichlet/Neumann Poisson Solver

### 5.1 Unified Plan type with per-axis BC

- [x] Implement main `Plan` struct:
  ```go
  type Plan struct {
      dim      int
      n        [3]int
      h        [3]float64
      bc       [3]BCType
      eig      [3][]float64
      tr       [3]AxisTransform
      work     workspace
      opts     Options
  }
  ```
- [x] Implement `NewPlan(dim int, n []int, h []float64, bc []BCType, opts ...Option) (*Plan, error)`
- [x] Select appropriate eigenvalue formula per axis based on BC
- [x] Select appropriate transform (FFT/DST/DCT) per axis based on BC
- [x] Validate BC combinations (document restrictions if any)

### 5.2 1D Dirichlet Solver

- [x] Wire DST transform for Dirichlet BC
- [x] Compute Dirichlet eigenvalues
- [x] Write manufactured solution tests:
  - [x] u = sin(πx/L) (fundamental mode)
  - [x] u = sin(2πx/L) \* sin(πx/L) (combination)
- [x] Verify boundary values are exactly zero

### 5.3 1D Neumann Solver

- [x] Wire DCT transform for Neumann BC
- [x] Compute Neumann eigenvalues
- [x] Handle nullspace (constant mode has zero eigenvalue)
- [x] Write manufactured solution tests:
  - [x] u = cos(πx/L)
  - [x] u = cos(2πx/L)
- [x] Verify derivative at boundaries is zero (finite difference check)

### 5.4 2D Mixed BC Solver

- [x] Test Dirichlet-Dirichlet (both axes)
- [x] Test Neumann-Neumann (both axes)
- [x] Test Periodic-Dirichlet (mixed)
- [x] Test Dirichlet-Neumann (mixed)
- [x] Write manufactured solution tests for each combination
- [x] Document which combinations have nullspace issues

### 5.5 3D Mixed BC Solver

- [x] Implement 3D with arbitrary BC per axis
- [x] Test all 27 combinations (3³) or representative subset
- [x] Benchmark against periodic-only solver

---

## Phase 6: Inhomogeneous Boundary Conditions

### 6.1 Boundary value data structures

- [x] Define `BoundaryFace` enum (XLow, XHigh, YLow, YHigh, ZLow, ZHigh)
- [x] Define `BoundaryData` struct:
  ```go
  type BoundaryData struct {
      Face   BoundaryFace
      Type   BCType  // Dirichlet or Neumann
      Values []float64  // boundary values (shape of face)
  }
  ```
- [x] Define `BoundaryConditions` as collection of `BoundaryData`

### 6.2 RHS modification for inhomogeneous Dirichlet

- [x] Implement RHS contribution from boundary values
- [x] For each boundary cell: `rhs[i] -= u_boundary / h²`
- [x] Write tests with non-zero Dirichlet values
- [x] Verify solution matches boundary values at edges

### 6.3 RHS modification for inhomogeneous Neumann

- [x] Implement ghost point elimination or modified stencil
- [x] For each boundary cell: adjust RHS based on derivative condition
- [x] Write tests with non-zero Neumann values
- [x] Verify derivative at boundary matches specified value

### 6.4 Unified inhomogeneous API

- [x] Implement `Plan.SolveWithBC(dst, rhs []float64, bc BoundaryConditions) error`
- [x] Write comprehensive tests for 2D and 3D
- [x] Add examples to examples/ directory

---

## Phase 7: Helmholtz Solver Extension

### 7.1 Helmholtz operator: (α - Δ)u = f

- [x] Extend `Plan` with `alpha` parameter (shift)
- [x] Modify eigenvalue division: divide by `α + λ` instead of just `λ`
- [x] Implement `NewHelmholtzPlan(...)` constructor
- [x] Handle α = 0 case (reduces to Poisson with nullspace)
- [x] Write tests for positive α (well-posed problem)
- [x] Write tests for negative α (potential resonance issues - document)

### 7.2 Screened Poisson / reaction-diffusion steady state

- [x] Document use case: `u - νΔu = f` (implicit diffusion step)
- [x] Add example for diffusion time-stepping
- [x] Benchmark against iterative methods for comparison

---

## Phase 8: Performance Optimization

### 8.1 Zero-allocation solve path

- [x] Audit `Solve()` for heap allocations using benchmarks
- [x] Ensure all scratch memory is pre-allocated in Plan
- [x] Add `Plan.WorkBytes()` method for memory introspection
- [x] Write allocation benchmarks: `go test -bench=. -benchmem`
- [x] Target: 0 allocs/op for Solve with pre-made plan

### 8.2 Parallelism support

- [ ] Pass through `Options{Workers: n}` to algo-fft plans
- [x] Pass through `Options{Workers: n}` to algo-fft plans
- [x] Parallelize eigenvalue division loop (if beneficial)
- [x] Parallelize line-wise DST/DCT transforms
- [x] Benchmark single-threaded vs multi-threaded
- [x] Document scaling characteristics

### 8.3 SIMD considerations

- [ ] Profile hot paths (eigenvalue division likely)
- [ ] Consider SIMD for eigenvalue division if beneficial
- [ ] Document any architecture-specific optimizations

### 8.4 Plan caching / reuse

- [ ] Document plan reuse patterns in README
- [ ] Consider `sync.Pool` for temporary buffers if needed
- [ ] Ensure thread-safety for concurrent Solve() calls on same plan

---

## Phase 9: Validation & Testing

### 9.1 Manufactured solution test suite

- [x] Create `testdata/` with analytic test cases
- [x] 1D periodic: u = sin(2πx/L)
- [x] 1D Dirichlet: u = sin(πx/L)
- [x] 1D Neumann: u = cos(πx/L) + x (linear + cosine)
- [x] 2D periodic: u = sin(2πx/Lx) \* sin(2πy/Ly)
- [x] 2D Dirichlet: u = sin(πx/Lx) \* sin(πy/Ly)
- [x] 2D Neumann: u = cos(πx/Lx) \* cos(πy/Ly)
- [x] 2D mixed: combinations of above
- [x] 3D cases for each BC type

### 9.2 Convergence tests

- [x] Implement `TestConvergence_*` tests
- [x] Verify O(h²) error convergence for 2nd-order FD
- [ ] Plot convergence (log-log) in documentation

### 9.3 Reference solver comparison

- [x] Implement naive dense solver for small grids (8x8, 16x16)
- [x] Compare spectral solution against direct solve
- [ ] Maximum error should be O(machine epsilon \* condition number)

### 9.4 Fuzzing

- [x] Add fuzz tests for robustness
- [x] Fuzz input sizes, values, BC combinations
- [x] Ensure no panics on edge cases

---

## Phase 10: Documentation & Examples

### 10.1 Package documentation

- [x] Write comprehensive doc.go for each package
- [ ] Document eigenvalue formulas with LaTeX/math
- [ ] Document memory layout conventions
- [ ] Document BC conventions and grid alignment

### 10.2 README.md

- [x] Project overview and motivation
- [x] Installation instructions
- [x] Quick start example
- [x] Performance characteristics
- [x] Comparison with alternatives

### 10.3 Examples

- [x] `examples/periodic1d/` - basic 1D periodic solve
- [x] `examples/periodic2d/` - 2D periodic solve with visualization
- [x] `examples/dirichlet2d/` - 2D Dirichlet problem
- [x] `examples/neumann2d/` - 2D Neumann problem
- [x] `examples/mixed2d/` - 2D mixed BC problem
- [x] `examples/helmholtz/` - Helmholtz equation
- [x] `examples/diffusion/` - implicit diffusion time-stepping

### 10.4 Benchmarks documentation

- [ ] Create BENCHMARKS.md with performance data
- [ ] Compare against gonum sparse solvers (if applicable)
- [ ] Document scaling: problem size vs time

---

## Phase 11: Future Extensions (Out of Scope for MVP)

### 11.1 Projection for incompressible flow

- [ ] Design API for pressure projection
- [ ] Implement divergence computation
- [ ] Implement gradient computation
- [ ] Write Navier-Stokes projection example

### 11.2 Variable coefficients

- [ ] Research preconditioned iterative methods
- [ ] Consider spectral method as preconditioner

### 11.3 Non-rectangular domains

- [ ] Research immersed boundary methods
- [ ] Consider mask-based approaches

---

## Phase 12: Simple WebAssembly Wave Demo ✅ COMPLETE

A minimal, shippable browser demo showcasing wave propagation using the Helmholtz solver. No UI controls - just click to ping.

### 12.1 Minimal UI Specification

**Interface:**

- [x] Fullscreen canvas
- [x] Top-left tiny text overlay: FPS + "Click to ping"
- [x] Interactions:
  - [x] **Click:** set source position; restart animation
  - [ ] **(Optional) R:** cycle resolution 128²/256²/512² - NOT IN MVP
  - [ ] **(Optional) B:** cycle boundary preset (Rigid / Open / Periodic) - NOT IN MVP
- [x] No forms, no sliders

### 12.2 Simulation Approach

**Approach A (Implemented): Multi-frequency synthesis animation**

- [x] Pick a small set of frequencies (16 bins, 80-600 Hz)
- [x] For each f_i:
  - [x] Solve steady field p_i(x,y) for source at click position
- [x] During animation frame t, render:
  ```
  u(x,y,t) = Σ_i w_i * p_i(x,y) * cos(2π f_i t) * exp(-γ_i t)
  ```
- [x] Add damping via exp(-γ_i t) with γ_i = 0.5 * f_i
- [x] Result: expanding/rippling patterns that reflect and decay

### 12.3 Core Pipeline

**On startup:**

- [x] Set grid: nx=256, ny=256
- [x] Set BC preset (rigid: Neumann on both axes)
- [x] Build and cache Helmholtz plan(s) for grid/BC

**On click(x,y):**

- [x] Convert click to grid indices (sx, sy)
- [x] Build source blob s(x,y) (Gaussian with radius=3 cells)
- [x] For each frequency bin f_i:
  - [x] Compute k_i = 2π f_i / c
  - [x] Set Helmholtz parameter (alpha = k_i²)
  - [x] Solve → store p_i (Float32Array)
- [x] Send ack to UI: "ready to animate"

**Each animation frame:**

- [x] Worker computes frame field u(x,y,t) and sends pixels
- [x] Worker returns Uint8ClampedArray rgba (already colormapped)
- [x] UI blits ImageData to canvas

### 12.4 Web/WASM Structure

**Files:**

- [x] `demo/index.html` - canvas + overlay
- [x] `demo/main.ts` - UI + click handling
- [x] `demo/sim.worker.ts` - loads wasm + does all compute
- [x] `cmd/acoustics-wasm/main.go` - exports functions
- [x] `demo/package.json` - Vite + TypeScript config
- [x] `demo/README.md` - Documentation

**Worker message protocol:**

- [x] `init {nx, ny, dx, dy, bcX, bcY}`
- [x] `ping {sx, sy, frequencies}`
- [x] `frame {t}`
- [x] Worker → UI: `pixels {rgba, width, height}` or `error`

### 12.5 WASM Exports (Minimal API)

**Three core entrypoints:**

- [x] `goInitPlan(nx, ny, dx, dy, bcX, bcY) -> {planID, nx, ny}`
- [x] `goSolve(planID, alpha, sx, sy, srcRadius) -> {field: Float32Array}`
- [x] `goGetPlanInfo(planID) -> {nx, ny, dx, dy}`

Everything else (synthesis, colormap, animation loop) lives in Worker JS/TS.

### 12.6 Visual Design

- [x] Render signed field with diverging colormap (blue-white-red)
- [x] Auto-scale using percentile clamp (5th–95th percentile)
- [ ] Add faint outline box showing room boundary - NOT IN MVP
- [ ] (Optional) Draw small dot at last click position - NOT IN MVP

### 12.7 Performance Knobs (No UI)

**NOT IN MVP - Future enhancements**

- [ ] **R:** cycle resolution (128²/256²/512²)
- [ ] **B:** cycle boundary conditions
- [ ] **F:** number of frequency bins (16/32/64)
- [ ] **D:** damping strength

### 12.8 Milestones

- [x] **Milestone 1:** Canvas + Worker + WASM loads ✅
- [x] **Milestone 2:** Multi-frequency solve on click ✅
- [x] **Milestone 3:** Multi-frequency synthesis animation ✅
- [x] **Milestone 4:** Diverging colormap with percentile normalization ✅
- [x] **Milestone 5:** FPS counter and performance optimization ✅

**Implementation Complete!** 🎉

Run with: `just demo-dev` and open http://localhost:5173

**Bundle size:** 4.1 MB WASM + 17 KB runtime
**Performance:** ~72ms for 16-mode solve, 60 FPS animation
**Files:** See `demo/README.md` for full documentation

---

## Phase 13: Full-Featured WebAssembly Acoustic Room Demo

A comprehensive browser-based interactive demo showcasing the Helmholtz solver for 2D acoustic room simulation with full controls and audio output.

### 13.1 Product Specification (MVP User Journey)

**What users can do:**

1. Pick a **room preset** (Rectangular, Lx×Ly)
2. Drag **Source (🔊)** and **Mic (🎤)** points on the room
3. Scrub a **frequency slider**
4. See:
   - a **pressure field heatmap** (amplitude; optional animated phase)
   - a **mic response plot** (amplitude vs frequency)
5. Click **Play** to auralize using an impulse response generated from the sweep

**Controls (MVP):**

- Room: width/height (meters), grid resolution (e.g. 128²/256²/512²)
- BC per edge (or per axis):
  - **Rigid** (Neumann)
  - **Open** (Dirichlet)
  - **Periodic** (for "waveguide loop" fun)
- Medium:
  - speed of sound `c` (default 343 m/s)
  - damping/loss knob (needed for nice, stable resonances)
- Source:
  - type: monopole (point-ish blob)
  - gain
- Mic:
  - readouts: SPL-ish (relative), phase (optional), transfer magnitude

**Nice-to-have:**

- [ ] "**Mode explorer**": show resonant patterns + highlight peaks
- [ ] "**Quality while dragging**": low-res preview then refine
- [ ] Shareable URLs encoding parameters

**Design constraint:**

- ✅ Rectangular rooms with clean BCs → perfect fit for separable spectral solver
- ⚠️ Arbitrary internal obstacles/polygons → **not directly supported** (MVP uses rectangular rooms only; obstacles can come later via iterative wrapper)

### 13.2 Architecture & Data Flow

**Runtime components:**

- [ ] **UI thread (React/TS)**
  - [ ] Canvas/WebGL rendering
  - [ ] Controls + plots
  - [ ] Audio playback
- [ ] **Simulation Worker**
  - [ ] Owns the WASM instance (runs hot without blocking UI)
  - [ ] Caching of plans/results
- [ ] **Go WASM module**
  - [ ] Wraps `algo-pde` plans
  - [ ] Performs field solve + sweep sampling
  - [ ] Returns arrays/buffers to worker

**Data flow:**

- UI → Worker: params, source/mic positions, "solve" requests
- Worker → WASM: numeric solve/sweep
- Worker → UI: field buffer, response arrays, progress updates
- UI → WebAudio: IR buffer (AudioBuffer) + dry audio input → ConvolverNode

### 13.3 Repository Structure

**Monorepo layout:**

- [ ] `/cmd/acoustics-wasm/` - Go `main` for wasm exports
- [ ] `/demo/` - Vite + React + TS
  - [ ] `/src/worker/sim.worker.ts`
  - [ ] `/src/wasm/loader.ts`
  - [ ] `/src/render/fieldRenderer.ts`
  - [ ] `/src/audio/auralizer.ts`
  - [ ] `/public/wasm/` - built artifacts: `acoustics.wasm`, `wasm_exec.js`

### 13.4 WASM Build & Integration

**Build strategy:**

- [ ] Compile with `GOOS=js GOARCH=wasm go build -o acoustics.wasm`
- [ ] Bundle/copy `wasm_exec.js` from Go's distribution

**Vite integration:**

- [ ] Treat `acoustics.wasm` as static asset
- [ ] Load in worker via `fetch()` + `WebAssembly.instantiateStreaming`
- [ ] (Optional later) Use Vite plugin for cleaner bundling

### 13.5 Go WASM API Design

**Exported functions (MVP):**

- [ ] `init() -> version/info`
- [ ] `createPlan2D(nx, ny, dx, dy, bcX, bcY, nullspaceMode) -> planHandle`
  - [ ] Implement plan caching by key: `(nx,ny,dx,dy,bcX,bcY)`
- [ ] `solveField(planHandle, k, damping, sourceX, sourceY, sourceWidthCells) -> Float32Array field`
  - [ ] Returns amplitude field (or signed pressure)
- [ ] `sampleAt(planHandle, lastFieldOrSolveResult, micX, micY) -> float`
- [ ] `sweepResponse(planHandle, fMin, fMax, nBins, damping, source…, mic…) -> Float32Array mags`
  - [ ] Return log-spaced frequencies

**Memory strategy:**

- [ ] Return raw `[]float32` to JS as `Uint8Array/Float32Array` (copied buffer)
- [ ] (Optional perf) Implement pinned "output buffer" in wasm for zero-copy reads

### 13.6 Acoustic Physics Mapping

**Helmholtz form:**

- [ ] Use stable shifted form: `(-Δ + k²) p = s` (screened Poisson)
- [ ] Map: `k = 2π f / c`, set `alpha = k²`
- [ ] Add damping as frequency-dependent smoothing or effective loss parameter

**Source injection:**

- [ ] Implement small Gaussian-ish blob over a few cells
- [ ] Normalize energy so loudness doesn't explode with resolution changes

**Field visualization:**

- [ ] Output `field = p(x,y)` (signed) or `abs(p)` (amplitude)
- [ ] (Optional) Animation: `frame(t) = p(x,y) * cos(2π f t)`

### 13.7 Frontend Implementation (React + Canvas/WebGL)

**UI components:**

- [ ] **RoomCanvas**
  - [ ] Draw border/BC icons
  - [ ] Draggable source/mic handles
  - [ ] Mouse events → normalized coordinates
- [ ] **ControlPanel**
  - [ ] Sliders + dropdowns
  - [ ] Preset selector
  - [ ] "Compute" + "auto update" toggle
- [ ] **Plots**
  - [ ] Mic magnitude vs freq (line plot)
  - [ ] (Optional) Show peaks / modal markers

**Rendering approach:**

- [ ] MVP: **Canvas2D ImageData** (fast enough for 256², OK for 512²)
- [ ] Later: **WebGL2 texture** upload for smoother scaling and faster colormaps

**Rendering pipeline:**

- [ ] Receive `Float32Array field`
- [ ] Compute min/max (robust: percentile clamp)
- [ ] Map to pixels via colormap LUT
- [ ] Paint to canvas (nearest-neighbor scaling)

**Interaction performance:**

- [ ] While dragging source/mic: run low-res plan (128²) quickly
- [ ] On release / idle: recompute at chosen quality (256²/512²)

### 13.8 Audio Auralization

**Tier A (MVP): Magnitude-only → minimum-phase IR**

- [ ] Sweep frequencies, get magnitude response at mic: `|H(f)|`
- [ ] Assume minimum phase and reconstruct phase from log-magnitude (Hilbert / cepstrum method)
- [ ] Build complex spectrum `H(f)` and IFFT → impulse response
- [ ] Create `AudioBuffer`, feed into `ConvolverNode.buffer`
- [ ] Prepare buffer off main thread to avoid jank

**Tier B (Upgrade): True complex response**

- [ ] Add "complex Helmholtz" path by representing complex values as two real fields (Re/Im)
- [ ] Do same transforms twice with complex division in spectral space
- [ ] Yields true phase and cleaner impulse responses

### 13.9 Testing & Validation

**Numerical validation:**

- [ ] Add validation suite runnable in Node (Go wasm under Node)
- [ ] Test cases:
  - [ ] Known analytic modes in a rectangle (compare eigenvalues / patterns)
  - [ ] Symmetry tests (move source and check mirrored field)
  - [ ] BC sanity (Dirichlet edges should go ~0 at boundaries)

**Web integration tests:**

- [ ] Worker start/stop, re-init correctness
- [ ] Determinism (same params → same result)
- [ ] Memory stability (repeat 100 solves, ensure no growth)

**Audio tests:**

- [ ] IR normalization and clipping safety
- [ ] Convolver on/off A/B toggles
- [ ] Export IR as WAV (simple PCM writer)

### 13.10 Deployment

- [ ] Build demo as static site (Vite)
- [ ] Host on **GitHub Pages** (or any static host)
- [ ] Ensure correct MIME types for `.wasm`

### 13.11 Milestones

**Milestone 1 — "Field viewer"**

- [ ] WASM loads in worker
- [ ] Rectangular room + BCs
- [ ] Source/mic drag
- [ ] Frequency slider updates heatmap

**Milestone 2 — "Response + plots"**

- [ ] Sweep response magnitude at mic
- [ ] Plot magnitude vs frequency
- [ ] Presets + quality modes + caching

**Milestone 3 — "Audio MVP"**

- [ ] Convert magnitude response → IR (minimum-phase)
- [ ] WebAudio convolver pipeline with A/B dry/wet
- [ ] Export IR

**Milestone 4 — "Polish + performance"**

- [ ] Progressive refinement while dragging
- [ ] WebGL renderer (optional)
- [ ] URL sharing + preset gallery
- [ ] Perf telemetry overlay (FPS, solve ms, allocations)

**Milestone 5 — "Physics upgrade (optional)"**

- [ ] Complex response for true phase
- [ ] Better damping model
- [ ] Mode explorer

### 13.12 Risk Mitigation

**Risk: Users want obstacles (pillars, L-shaped rooms)**

- Mitigation: Ship MVP as "Room Modes Lab (Rectangles)" with strong educational framing
- Later: Add obstacles via iterative method (penalty / immersed boundary) using spectral solver as fast inner solve

**Risk: Indefinite Helmholtz causes instability**

- Mitigation: Use stable shifted form for MVP, add damping, clamp peaks, present as "steady-state field"

**Risk: WASM ↔ JS overhead / UI jank**

- Mitigation: Keep WASM inside Worker; return transferable buffers; limit recomputes during drag

---

## Implementation Order Summary

**MVP (Phases 0-4):** ~2-3 weeks of focused work

1. Phase 0: Setup ✅ COMPLETE
2. Phase 1: Grid package ✅ COMPLETE (core functionality)
3. Phase 2: R2R transforms ✅ PARTIAL (DST-I/DCT-I done, line-wise pending)
4. Phase 3: FD operators ✅ PARTIAL (eigenvalues done)
5. Phase 4: Periodic Poisson (1 week)

**Feature Complete (Phases 5-7):** ~2-3 weeks additional 6. Phase 5: Dirichlet/Neumann (1 week) 7. Phase 6: Inhomogeneous BC (3-4 days) 8. Phase 7: Helmholtz (2-3 days)

**Polish (Phases 8-10):** ~1-2 weeks 9. Phase 8: Optimization (ongoing) 10. Phase 9: Validation (ongoing, parallel with implementation) 11. Phase 10: Documentation (ongoing)

---

## Dependencies

- `github.com/MeKo-Christian/algo-fft` - FFT plans and execution
- Standard library only otherwise (no external deps)

## Testing Strategy

- Unit tests for every exported function
- Table-driven tests for BC combinations
- Manufactured solution tests for correctness
- Benchmark tests for performance regression
- Fuzz tests for robustness

## Success Criteria

1. **Correctness:** All manufactured solution tests pass with error < 1e-10 (for double precision, h → 0)
2. **Performance:** O(N log N) complexity verified empirically
3. **Memory:** Zero allocations in Solve() with pre-made plan
4. **API:** Clean, plan-based API consistent with algo-fft style
5. **Documentation:** All public APIs documented with examples
