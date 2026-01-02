## 1) What you can lean on from `algo-fft` (important for the architecture)

Even without reading the code contents, the repo structure indicates it already contains:

- **2D/3D/ND FFT planning** (`plan_2d.go`, `plan_3d.go`, `plan_nd.go`) ([GitHub][3])
- **Real FFT planning** (`plan_real.go`, `plan_real_2d.go`, `plan_real_3d.go`) ([GitHub][3])
- **Bluestein support** (`plan_bluestein.go`) for “awkward sizes” ([GitHub][3])
- **Batch / strided / executor-related pieces** (`plan_batch.go`, `plan_strided.go`, `executor.go`) ([GitHub][3])
- Already has **convolution/correlation helpers** (`convolve.go`, `convolve_real.go`, `correlate.go`) ([GitHub][3])

That’s excellent: your Poisson solver can be mostly “thin math + orchestration”, delegating transforms to those plans/executors.

---

## 2) Core math choice (sets you up for physical boundaries)

To support **physical boundaries** efficiently, the cleanest “fast Poisson solver” path is:

### Use the **finite-difference Laplacian** on a regular grid

Then the operator is diagonalized by:

- **Periodic** → FFT
- **Dirichlet (u=0 at boundary)** → DST (sine transform)
- **Neumann (∂u/∂n=0 at boundary)** → DCT (cosine transform)
- **Mixed** → use a different transform per axis

This gives you a direct solver via **separation of variables**:

1. Transform RHS (f) into spectral space (axis-by-axis).
2. Divide each coefficient by the summed eigenvalues.
3. Inverse-transform back.

Complexity: **O(N log N)**, memory: **O(N)**.

This approach is “general purpose” for **1D/2D/3D** on rectangles/boxes and is exactly what you want to grow into later PDE solvers.

---

## 3) Library shape: keep it simple, composable, and plan-centric

### Proposed module layout

Create a repo like `algo-pde` or `algo-poisson` with packages:

- `poisson/` – Poisson solvers, plans, boundary handling
- `r2r/` – real-to-real transforms layer (DST/DCT wrappers implemented via `algo-fft`)
- `grid/` – indexing, shapes, strides, small helpers
- `fd/` – finite difference operators (optional but useful for tests + PDE building blocks)

### The key type: a reusable “plan”

Like FFTW, you want plan creation separate from solve calls.

```go
type BCType int
const (
  Periodic BCType = iota
  Dirichlet
  Neumann
)

type AxisBC struct {
  Left, Right BCType // later allow mixed per-side; start with “same on both ends”
}

type Plan struct {
  Dim int              // 1,2,3
  N   [3]int           // nx,ny,nz (unused entries = 1)
  H   [3]float64       // dx,dy,dz
  BC  [3]AxisBC        // per-axis boundary

  // precomputed eigenvalues per axis and combined denominator workspace
  eig [3][]float64

  // transform operators for forward/inverse, built from algo-fft plans
  tr  transformStack

  // scratch buffers (owned by plan) to avoid allocations
  work workspace

  // options: normalization, threading, constraints for nullspace, etc.
  opt Options
}
```

---

## 4) “Transform stack” design (this is the whole game)

You need **axis-wise transforms** for each axis depending on BC:

### 4.1 Periodic axis: FFT / RFFT

- If the overall solve is real-valued (common), prefer **real FFT plans** to cut work roughly in half.
- `algo-fft` appears to have real 2D/3D planning which is perfect. ([GitHub][3])

### 4.2 Dirichlet axis: DST via FFT

Implement DST by embedding into an FFT with an **odd extension**.

- You can implement DST-I / DST-II (pick one consistent with your FD grid convention).
- The wrapper should:
  - map length N real line to an “extended length” M (often `2*(N+1)` or `2*N`)
  - call `algo-fft` on that extended buffer
  - extract sine coefficients

### 4.3 Neumann axis: DCT via FFT

Implement DCT via **even extension** similarly.

### 4.4 Mixed BCs (per-axis)

In 2D/3D, you’ll do:

- transform along X lines, then Y lines, then Z lines
- each axis uses its own operator (FFT/DST/DCT)

**Implementation strategy:** build an internal interface like:

```go
type AxisTransform interface {
  ForwardLines(dst, src []float64, shape Shape, axis int)
  InverseLines(dst, src []float64, shape Shape, axis int)
  // maybe also Complex variants if you choose FFT complex domain in the middle
}
```

Then your `Plan` composes `[3]AxisTransform` (one per axis). That way, adding a new BC type is “just” another axis transform.

---

## 5) Eigenvalues + denominator handling (fast, stable, correct)

For a **standard 2nd-order FD Laplacian**, the 1D eigenvalues are:

- **Periodic** (m = 0..N-1):
  [
  \lambda_m = \frac{2 - 2\cos(2\pi m/N)}{h^2}
  ]
- **Dirichlet** (m = 1..N):
  [
  \lambda_m = \frac{2 - 2\cos(\pi m/(N+1))}{h^2}
  ]
- **Neumann** (m = 0..N-1):
  [
  \lambda_m = \frac{2 - 2\cos(\pi m/N)}{h^2}
  ]
  (note: m=0 is 0 → nullspace mode)

In multi-D:
[
\Lambda(i,j,k)=\lambda_x(i)+\lambda_y(j)+\lambda_z(k)
]

### Practical plan

- Precompute `eigX`, `eigY`, `eigZ` once in `NewPlan(...)`.
- During solve, when you’re in spectral space:
  - divide coefficient-wise by `eigX[i] + eigY[j] (+ eigZ[k])`

- Avoid building a full 3D denominator array unless it’s faster for your memory/CPU tradeoff. Start “sum on the fly”.

### Nullspace constraints (critical!)

- Periodic and Neumann have **a zero eigenvalue** in the constant mode.
- Enforce one of:
  1. require RHS has mean zero and set the zero-mode of solution to 0, or
  2. pin one value (or mean) with an explicit constraint option

This must be explicit in API; otherwise users get NaNs or huge drift.

---

## 6) The solve pipeline (periodic first, then extend)

### Phase A: Periodic Poisson (fast to get right)

**Solve(f) → u**

1. Validate sizes, strides, nils.
2. Copy/scale RHS into plan buffer (optional).
3. Forward transform (RFFT/FFT) using `algo-fft` ND/2D/3D plan. ([GitHub][3])
4. Divide by eigenvalue sum, handle zero-mode.
5. Inverse transform.
6. Normalize (depending on `algo-fft` conventions).
7. Return.

Deliverables:

- `poisson.NewPlanPeriodic1D/2D/3D(...)`
- `Plan.Solve(dst, rhs)` and `Plan.SolveInPlace(buf)` variants
- Benchmarks + manufactured-solution tests.

### Phase B: Add Dirichlet / Neumann via DST/DCT wrappers

Now replace the periodic axis transform with the DST/DCT axis transform where requested.

Deliverables:

- `NewPlan(...)` that accepts per-axis BCs
- axis transforms: `dstTransform`, `dctTransform`, `fftTransform`
- tests per BC type.

### Phase C: Mixed boundaries + inhomogeneous boundaries

- Start with homogeneous BCs (zero Dirichlet, zero Neumann).
- Then support nonzero boundary data by **RHS modification** (standard FD trick):
  - For Dirichlet: add boundary contributions to the nearest interior cells
  - For Neumann: incorporate ghost-point elimination or modified stencil

Deliverables:

- `BoundaryValues` optional argument with callbacks or arrays for each face
- `Plan.SolveWithBC(dst, rhs, bcValues)`

---

## 7) Data model: avoid Go-level overhead, be explicit about layout

Make the primary API accept **flat slices** with shape:

- 1D: `[]float64` length `nx`
- 2D: row-major `[]float64` length `nx*ny`
- 3D: row-major `[]float64` length `nx*ny*nz`

Add a small `Shape/Stride` type internally so you can support:

- contiguous buffers (fast path)
- strided views later (nice-to-have)

Since `algo-fft` already hints at “strided” planning (`plan_strided.go`) ([GitHub][3]), keep your architecture ready to exploit that later.

---

## 8) Performance strategy (where the big wins are)

### 8.1 Plan caching

- Planning + twiddle generation is expensive; Poisson is typically “same grid, many solves”.
- Provide a plan that is safe for repeated `Solve` calls.

### 8.2 Zero allocations in Solve

- Own scratch buffers inside the plan (`work`).
- Expose `Plan.WorkBytes()` for introspection if you like.
- Offer `Solve(dst, rhs)` (no alloc) and a convenience `SolveNew(rhs)` (alloc).

### 8.3 Parallelism

You likely want parallelism in the transform layer, not in the denominator division.
Since `algo-fft` has an `executor.go` and batch planning hooks ([GitHub][3]), design your plan to:

- accept `Options{Workers:int}` or `Options{Executor:...}` (whatever fits `algo-fft`)
- reuse that executor for line-wise DST/DCT too

---

## 9) Validation strategy (you’ll catch 90% of bugs here)

### Unit tests

- Eigenvalue formulas per BC and small N.
- Round-trip tests for your DST/DCT wrappers: inverse(forward(x)) ≈ x.

### Manufactured solution tests (the gold standard)

Pick an analytic `u(x,y,z)` consistent with BCs, compute discrete RHS using your FD stencil, solve, compare `u`.

Examples:

- Periodic: `u = sin(2πx/Lx) sin(2πy/Ly)`
- Dirichlet: `u = sin(πx/Lx) sin(πy/Ly)`
- Neumann: `u = cos(πx/Lx) cos(πy/Ly)` (careful with compatibility)

### Cross-check against a slow reference

For small sizes (e.g., 8³, 16²):

- build the sparse Laplacian and solve with Gaussian elimination / simple CG (internal reference)
- compare solution vectors

---

## 10) Roadmap: from Poisson to PDE solvers (what you originally wanted)

Once Poisson is solid, you can build:

1. **Diffusion / heat equation**:

- implicit step `(I - νΔtΔ)u_{n+1} = u_n` → same diagonalization trick (divide by `1 + νΔtΛ`)

2. **Projection step for incompressible flow (Navier–Stokes)**:

- solve Poisson for pressure from divergence → huge practical use-case

3. **Helmholtz**:

- `(α - Δ)u = f` → again just modify denominator

All of those reuse the exact same transform stack and eigenvalues.

---

## 11) A realistic phased implementation plan (so you don’t drown)

### Milestone 1 (fast win): Periodic 1D/2D/3D

- Use `algo-fft` ND or dedicated 2D/3D plans ([GitHub][3])
- Real-valued API, real FFT where possible ([GitHub][3])
- Mean-zero handling

### Milestone 2: Homogeneous Dirichlet + Neumann

- Implement DST/DCT wrappers using `algo-fft` under the hood
- Mixed per-axis in 2D/3D

### Milestone 3: Inhomogeneous BCs

- RHS modification utilities
- Good docs and examples

### Milestone 4: Helmholtz / diffusion / projection helpers

- same plan, different denominators

---

If you want, next I can propose a **concrete public API** (function names, option patterns, error model, plan caching story, and a minimal example program for each BC) while keeping it aligned with how `algo-fft` is organized (plans/executor/strides) based on the repo’s structure.

[1]: https://pkg.go.dev/github.com/mjibson/go-dsp/fft?utm_source=chatgpt.com "fft package - github.com/mjibson/go-dsp/fft - Go Packages"
[2]: https://github.com/cpmech/gosl?utm_source=chatgpt.com "GitHub - cpmech/gosl: Linear algebra, eigenvalues, FFT, Bessel ..."
[3]: https://github.com/MeKo-Christian/algo-fft/ "GitHub - MeKo-Christian/algo-fft: WORK IN PROGRESS"
