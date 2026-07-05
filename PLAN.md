# algo-pde Roadmap

Fast spectral Poisson/Helmholtz solver library for Go, built on `algo-fft`.

This plan was reset in July 2026 after a deep adversarial review. The initial
build-out (grid, r2r transforms, fd operators, periodic/Dirichlet/Neumann/mixed
solvers, inhomogeneous BC, Helmholtz, WASM demo) is done and lives in git
history. What follows is only what is **ahead of us**, ordered by priority.

The review's verdict, for context: the numerical core (transform kernels,
eigenvalue formulas, boundary-lifting algebra) is verified correct. What is
broken is the contract layer around it — documentation claims that are false,
tests that structurally cannot fail, options that silently no-op, and a demo
that solves the wrong equation. No rewrite; targeted repair.

---

## Phase A: Fix False Contracts (highest priority)

The library currently makes three headline promises — concurrent-safe plans,
O(N log N), zero allocations — and all three are false. Fix or retract each.

### A.1 Concurrency: make the doc.go claim true (or delete it)

`doc.go` claims "Plans are safe for concurrent Solve() calls". Every plan type
funnels each call through shared plan-owned workspaces with no synchronization
(`poisson/periodic_1d.go:77`, `periodic_2d.go:128`, `periodic_3d.go:140`,
`periodic_nd.go:142`, `plan.go:153`). `PlanNDPeriodic` additionally mutates
plan-level index state (`eigIndices`, `axisIdx`) per call.

- [x] Decide the contract: per-call workspace via `sync.Pool`, or document
      "one goroutine per plan" and delete the doc.go claim. (Recommended:
      `sync.Pool` of workspaces — keeps the zero-alloc steady state.)
      → Implemented pooled per-solve workspaces everywhere; pool entries are
      built with full constructors rather than `Clone`. Both upstream Clone
      defects (nil `stridedScratch` in `Plan.Clone`; `PlanReal2D/3D.Clone`
      sharing the stateful row/width plan) are fixed in algo-fft v0.6.13,
      which this repo now depends on (module path moved to
      `github.com/cwbudde/algo-fft`). Switching pool entries to `Clone` to
      share twiddle tables is an optional follow-up optimization.
- [x] Remove per-call plan mutation in `periodic_nd.go` (make Solve stateless
      w.r.t. the plan).
- [x] Remove the lazy `p.work.Real` reallocation in `plan_bc.go:29-31`
      (dead or racy; the buffer is already pre-allocated in `plan.go:116`).
- [x] Fix `FFTPlan.TransformLines` — concurrent callers on the same plan share
      `plans[0]`/`scratchA[0]` (`fft_plan.go:96`).
- [x] Add a concurrent-Solve race test (N goroutines, one shared plan, verify
      results match serial) for every plan type; run under `just test-race`.

### A.2 Performance: kill the O(N²) inverse transforms

`DST2Plan.Inverse` and `DCT2Plan.Inverse` are naive O(N²) triple loops
recomputing `math.Sin`/`math.Cos` per element (`r2r/dst.go:263`,
`r2r/dct.go:258` — the TODO admits it). Measured 513–650× slower than Forward
at n=1024; every Neumann axis in the Poisson solver pays this per line. This
falsifies the O(N log N) claim.

- [x] Implement DST-III/DCT-III via FFT embedding (same technique as the
      forwards), using the plan's existing FFT and buffers.
      → `DST2Plan.Inverse`/`DCT2Plan.Inverse` now pack the weighted
      coefficients into a single **real** 2N-point FFT (cosine part
      even-symmetric, sine part odd-symmetric around the midpoint) and read
      `Re(FFT)+Im(FFT)`. Real FFT input is deliberate: algo-fft's mixed-radix
      kernel returns wrong results for 2N ∈ {40,80,160,320,200,400}
      (n ∈ {20,40,80,160,100,200}) — see `fftSoundSizes` in
      `r2r/inverse_fft_test.go`; that upstream defect breaks the (real-only)
      Forward identically, so the inverse is correct wherever Forward is.
      `TestD{S,C}T2PlanInverse_MatchesNaive` pins the new path against the old
      O(N²) formula across all FFT-sound sizes.
- [x] Eliminate the per-call allocation in the aliased `Inverse(buf, buf)`
      path. src is now fully consumed into the plan's FFT buffer before dst is
      written, so aliasing needs no scratch copy; `TestInverseAllocFree`
      asserts 0 allocs/call.
- [x] Add `ForwardLines`/`InverseLines` to `DST2Plan`/`DCT2Plan` (`r2r/lines.go`)
      and collapse the two ~84-line strided blocks in
      `poisson/axis_transform.go` into one generic `realAxisTransform[P]` over
      a `realLinePlan` interface (DST-I and DCT-II share it).
- [x] Benchmark Neumann vs Dirichlet vs periodic solves at 256²/512²/1024²
      (`poisson/bc_scaling_bench_test.go`). Neumann (DCT-II inverse) went from
      236 ms → 4.0 ms at 256² and 1756 ms → 18.8 ms at 512² (~59× / ~93×) and
      now scales linearithmically like periodic instead of O(N²). (The
      Dirichlet path is slow only at these sizes because 2(n+1) is a
      Bluestein-heavy prime factorization — an orthogonal FFT-size concern.)

### A.3 Correctness traps that return garbage with err == nil

- [x] Helmholtz resonance: replace exact `denom == 0` (`plan.go:244`) with a
      relative-tolerance check (`|denom| < eps * |alpha|`-scale); return
      `ErrResonant` for near-resonance instead of a ~1e16-amplified field.
      → `applyEigenvalues` now gates on `|denom| <= resonanceRelTol*scale`,
      where `scale` is the sum of the term magnitudes (`|alpha| + Σ|eig|`).
      That ratio is exactly the catastrophic-cancellation conditioning of the
      divide, so it flags near-resonance (amplification > ~1e9) while never
      tripping on Poisson (all terms non-negative ⇒ `|denom| == scale`).
      `TestHelmholtz_NearResonantReturnsErrResonant` pins the 1-ulp case.
- [x] Validate `alpha` and `h` for NaN/Inf at plan creation (`plan.go:34,79` —
      `h <= 0` does not reject NaN).
      → New `validSpacing`/`validAlpha` (`validation.go`) reject NaN/±Inf; wired
      into every constructor. Non-finite alpha returns the new `ErrInvalidAlpha`.
- [x] Fix the nullspace mean gate: `meanTol = 1e-12` (`plan.go:144`) rejects
      analytically compatible inhomogeneous Neumann problems whose lifted RHS
      mean is O(h²) quadrature error (~1e-4 at n=100). Scale the tolerance
      with the discretization (or with the lifting magnitude), and use
      pairwise/Kahan summation in `meanAndMaxAbs` so the gate is stable at
      512³. Today every test dodges this via `WithSubtractMean()` — the
      default path is untested and unusable.
      → `meanRelTol(minExtent)` scales the relative tolerance as
      `max(8/minN², 1e-12)`, tightening as the grid refines (matching the
      O(h²)=O(1/n²) shrink of the quadrature error) yet still separating a
      compatible RHS from a genuinely inconsistent O(1) mean. `meanAndMaxAbs`
      now uses Neumaier compensated summation so the gate is stable at 512³.
      `TestNeumann_DefaultZeroMode{Accepts,Rejects}*` cover both directions on
      the default path.
- [x] `WithRealFFT(true)` silently downgrades the whole solve to float32
      buffers (`periodic_2d.go:128`). Either document the ~1e-6 accuracy
      contract loudly in the option, or rename it (`WithFloat32`), and add a
      test comparing float32 vs float64 paths on the same input.
      → Added `WithFloat32` alias whose name states the precision trade, loudly
      documented the ~1e-6 contract on `WithRealFFT`/`UseRealFFT`, and exposed
      `Plan2D/3DPeriodic.UsedRealFFT()` so callers can confirm which path ran.
      `TestWithFloat32_SinglePrecisionContract` checks the float32 result stays
      within the contract yet visibly differs from the float64 path.
- [x] With `NullspaceError`, periodic plans construct fine but every Solve
      fails (`periodic_1d.go:63`). Fail at plan creation instead.
      → All periodic constructors (and the core `Plan` whenever `hasNullspace()`)
      reject `NullspaceError` at construction; the dead per-Solve checks are gone.
      `TestNullspaceError_RejectedAtConstruction` verifies each constructor and
      that a Dirichlet `Plan` still builds.

### A.4 Document the grid conventions (cheap, prevents the worst failure mode)

Dirichlet axes are vertex-centered (nodes at (i+1)h, length (n+1)h, DST-I);
Neumann axes are cell-centered (nodes at (i+½)h, length nh, DCT-II). This is
nowhere documented — users sampling f/g at the wrong points converge smoothly
to the wrong answer. `bc.go:15` documents Neumann as outward-normal ∂u/∂n = g
but the implementation uses the positive-axis derivative (sign-flipped at low
faces).

- [x] Write the conventions section in `poisson/doc.go`: node placement per
      BC, domain length per BC, Neumann sign convention, mixed-axis
      implications. Include an ASCII diagram.
      → New `# Grid Conventions` section documents the per-axis node rules
      (Periodic x_i=i·h, L=n·h; Dirichlet x_i=(i+1)·h, L=(n+1)·h, DST-I;
      Neumann x_i=(i+½)·h, L=n·h, DCT-II) with an ASCII diagram of all three
      layouts, the "sample f/g at these nodes or converge to the wrong answer"
      warning, and the mixed-axis length/offset implication.
- [x] Fix the `bc.go` Neumann sign doc to match the implementation (or flip
      the implementation to outward-normal and migrate — decide once, now).
      → Decided: document the code as-is (positive-axis derivative). The
      `Neumann` const comment and the doc.go conventions section now state that
      the supplied g is ∂u/∂x_axis (g=−∂u/∂n at low faces, g=+∂u/∂n at high),
      matching `boundary_neumann.go` and every existing test/example. No
      behavior change.
- [x] Document eigenvalue formulas and memory layout (carried over from the
      old plan; still open).
      → New `# Eigenvalues and Memory Layout` section in `poisson/doc.go`
      lists the three per-BC eigenvalue formulas, the sum rule
      α + λ_x(i)+λ_y(j)+λ_z(k), and the row-major spectral layout
      (idx = i·ny·nz + j·nz + k, axis 0 slowest).

---

## Phase B: Rebuild Test Trust

The suite probes modes 1–3 only; the general-RHS case — the point of a Poisson
solver — is essentially unverified. A corrupted eigenvalue above index 3 would
ship green today.

- [x] **Random-RHS residual checks** (the single highest-leverage item): for
      every BC combination in 1D/2D/3D, solve a random RHS and assert
      `fd.Apply*(solution) ≈ rhs` to tight tolerance. This pins the entire
      spectrum, not just low modes.
      → `poisson/random_residual_test.go` exercises all 3 BCs in 1D, all 9
      pairs in 2D, and all 27 triples in 3D (distinct extents/spacings per
      axis to catch axis-swapped indexing). Each test solves a deterministic
      random RHS and asserts the reapplied `fd.Apply{1,2,3}D` reproduces it to
      1e-9 relative. Nullspace cases (all-Neumann/Periodic) project out the RHS
      mean up front and solve with `WithSubtractMean`, since only the mean-zero
      RHS is in the operator's range. A mutation that corrupts the highest
      Dirichlet eigenvalue (index n, far above the modes the manufactured
      sine/cosine tests probe) fails this check at 2e-3 while every manufactured
      test still passes — confirming it pins the whole spectrum, not just low
      modes.
- [x] Feed the dense Gaussian-elimination reference
      (`reference_solver_test.go`) **random** RHS, and extend it beyond 2D
      homogeneous Dirichlet: Neumann, periodic, mixed, anisotropic h,
      Helmholtz, inhomogeneous BC.
      → `TestReferenceSolve2D_Random` + `buildDenseOperator2D` compare the
      spectral solver against a dense Gaussian-elimination reference on a
      deterministic random RHS for 2D Neumann, periodic, Dirichlet×Neumann,
      anisotropic-h Dirichlet, and Helmholtz(α>0). Singular (Neumann/periodic)
      cases project to zero-mean and compare up to an additive constant.
- [x] Give the fuzz test a property: currently `_ = plan.Solve(dst, rhs)`
      (`fuzz_test.go:58`) asserts nothing. Check err handling, finite output,
      and (for valid inputs) the residual. Un-tie `nz` from `nx` (line 27).
      → Fuzz target now takes an independent `nz`, clamps dims/spacings, and on
      a successful solve asserts finite output and a reapplied-operator residual
      (mean-projected for nullspace BCs) within tolerance.
- [x] Delete or fix vacuous assertions:
      `(u[0]-u[0])/h` in `neumann_1d_test.go:131` (identically zero),
      the `t.Logf`-only `TestDCTPlan_KnownValues` (`dct_test.go:117`),
      the `math.Sin(0)` checks in `dirichlet_1d_test.go:29,62`.
      → `checkNeumannDerivative` now computes a real one-sided FD of the boundary
      derivative; `dirichlet_1d_test.go` extrapolates the solver output to the
      vertex-centered boundary; `TestDCTPlan_KnownValues` asserts the DCT-I
      amplitude spike (`N-1`) at the driven mode (the old `(N-1)/2` expectation
      was wrong, which is why it never asserted).
- [x] Tighten the periodic convergence test: `0.6×` per halving accepts
      first-order schemes (`periodic_2d_test.go:160`); require rate ≥ 1.8 as
      `convergence_test.go` already does — and add strict-order tests for
      Neumann, periodic, mixed, and 3D.
      → Periodic test uses the strict log-log `checkConvergenceRates` (≥1.8);
      added `TestConvergence2D_Neumann`, `_Mixed_DirichletNeumann`, and
      `TestConvergence3D_Dirichlet`.
- [x] Replace the circular eigenvalue tests (`fd/eigenvalues_test.go` re-types
      the identical formula inline) with actual small-matrix
      eigendecomposition or brute-force stencil checks on random vectors.
      → `fd/eigenvalues_verify_test.go` `TestEigenvaluesMatchStencil` applies the
      real `Apply1D` stencil to each BC's analytic eigenvector and asserts
      `A·v_k = λ_k·v_k` — no restatement of the closed-form λ formula.
- [x] Helmholtz gaps: solve tests for negative non-resonant α, a
      near-resonant α test (1 ulp off — must return ErrResonant, not
      garbage), Neumann/periodic BC with α > 0.
      → Negative-non-resonant and near-resonant already existed in
      `correctness_a3_test.go`; added α>0 accuracy tests for Neumann and Periodic
      BC (1D/2D) in `helmholtz_test.go`.
- [x] Cover the 0%-coverage surface: `Plan.SolveInPlace`,
      `Plan2DPeriodic.SolveInPlace`, `Plan3DPeriodic.SolveInPlace`,
      `WithNullspace`, `WithWorkers`, `WithInPlace`.
      → `coverage_gaps_test.go` adds `TestWithInPlace_MatchesDefault` (also
      exercises `Plan.SolveInPlace`), `TestWithNullspace_Functional`, and
      dedicated `Plan2D/3DPeriodic.SolveInPlace` correctness tests; `WithWorkers`
      was already covered by `concurrency_test.go`.
- [x] Asymmetric Neumann data through `SolveWithBC` (currently only constant,
      symmetric faces — `inhom_api_test.go:50,134`; swapped/mirrored faces
      would pass).
      → `TestPlan2D_SolveWithBC_AsymmetricNeumannFaces` uses yHigh=−yLow and
      x-varying face data.
- [x] Add a naive-DFT reference test for DST-I/DCT-I amplitudes (only type-II
      has one), and a test pinning the Hermitian/Nyquist bin in the real-FFT
      path.
      → `dst1Reference`/`dct1Reference` + `TestDSTPlan_Reference`/
      `TestDCTPlan_Reference` (direct O(N²) type-I DFT), and
      `TestPlan2DPeriodic_RealFFT_NyquistBin` pins the Nyquist mode in the
      real-FFT path.

---

## Phase C: API Hardening — no more silent failure

House style today: bad input → silent no-op, silent garbage, or panic. Pick
one contract (errors) and enforce it.

- [x] `fd.Apply1D/2D/3D`: size mismatch is a silent no-op leaving stale data
      in dst (`fd/laplacian.go:13,82,165`) — return an error. Unknown BCType
      silently becomes zeros in `Eigenvalues` and Dirichlet in `Apply2D/3D` —
      error in both.
- [x] `r2r` lines API: panics on short buffers, bad axis, zero-extent shapes
      (`r2r/lines.go:61-87`) — validate and return `ErrSizeMismatch` /
      `ErrInvalidAxis`.
- [x] `grid`: `LineIterator`/`PlaneIterator` yield phantom lines for shapes
      with a zero extent (`grid/grid.go:149-206`); validate shape and axis at
      construction. Reject negative extents in `NewShape*`. (Zero-extent
      iterators return no phantom lines/planes — covered by
      `TestLineIteratorZeroExtentYieldsNoLines` /
      `TestPlaneIteratorZeroExtentYieldsNoPlanes`. `NewShape1D/2D/3D` now panic
      on a negative extent via `mustNonNegative`; zero remains allowed — see
      `TestNewShapeRejectsNegativeExtent`.)
- [x] `grid.Shape.Dim()` infers dimension from trailing extents, so 64×64×1
      reports 2D and `SolveWithBC` rejects its Y faces — store the declared
      dimension instead of guessing. (`grid.Shape` is now a value-semantics
      struct `{dims [3]int; ndim int}`; `Dim()` returns the declared `ndim`, so
      `NewShape3D(64, 64, 1).Dim() == 3`. `SolveWithBC` already gated on the
      plan's `p.dim`, so its Z/Y faces were never actually rejected; regression
      locked by `TestSolveWithBC3DSmallNzAcceptsZFaces` and `TestShape_Dim*`.)
- [x] Options that silently no-op — make each either work or error:
      `WithWorkers` on `PlanNDPeriodic` (ignored entirely),
      `WithSolutionMean` on plans without nullspace (`plan.go:174`),
      `WithRealFFT` on the BC-plan path, `WithInPlace` on periodic plans.
- [x] `SolveWithBC`: reject duplicate faces (two `XLow` entries silently
      double the contribution); don't corrupt the caller's rhs on a mid-loop
      error in the `InPlace` path (`plan_bc.go:36-62`).
- [x] `NormOrtho` DCT-I is not orthonormal (missing √2 endpoint weights) —
      fix or document; Parseval-based uses are silently wrong today.
- [x] `NormalizationFactor` is a dead API whose documented semantics are
      wrong by O(N) (`r2r/dct.go:293`, `transform.go:21`) — delete it or fix
      doc + implementation to agree.
- [x] Replace `log` calls on the real-FFT fallback (`periodic_2d.go:54`) with
      an inspectable plan property (`p.UsedRealFFT() bool`) or an option-level
      error.
- [x] Fix misleading fd docs: formulas are for the **negative** Laplacian;
      `fd/doc.go` says "Laplacian".

---

## Phase D: Structural Debt

- [x] **Un-duplicate the eigenvalue formulas.** Extracted leaf package `bc/`
      (BCType + the single copy of the eigenvalue formulas). `poisson.BCType`
      is now an alias of `bc.BCType`; `poisson/eigenvalues_bc.go` and
      `eigenvalues_periodic.go` are deleted, `fd/eigenvalues.go` is deleted, and
      `fd` no longer imports `poisson` (cycle broken). The `(2-2cos)/h²`
      formula now lives only in `bc/eigenvalues.go`.
- [x] One `Shape` type. `grid.Shape` (fixed `[3]int`) and `poisson`'s
      `Shape []int` (`shape_nd.go`) coexist; `parallel.go`'s helpers hardcode
      3 axes and would silently miscount for >3D.
      (→ DONE — `poisson.Shape` and `shape_nd.go` are deleted; a single
      slice-backed `grid.Shape struct { dims []int }` now serves every package.
      `Dim()` returns `len(dims)`; `N(axis)` returns `dims[axis]` or **1** for
      `axis >= len(dims)` (preserving the trailing-axis-is-1 semantics the ≤3D
      indexing helpers rely on); `Size()` is the product; `Dims()` exposes the
      backing slice read-only for the N-D `periodic_nd` iteration; the ≤3D
      helpers (`RowMajorStride`, `Index3D`, `FromIndex3D`) and the
      `Line`/`PlaneIterator`s were rewritten to drive off `N(axis)`.
      Constructors `NewShape1D/2D/3D` still panic on a negative extent
      (`mustNonNegative`), and a new `NewShapeND(dims ...int)` / `NewShapeN([]int)`
      builds the N-D path. `NewPlanNDPeriodic` now takes a `grid.Shape` and
      `PlanNDPeriodic` still solves >3D (4D tests unchanged). Zero-alloc
      mitigation: a slice-backed Shape allocates on construction, so the general
      `Plan` builds its shape ONCE at construction and stores it in `Plan.shp`;
      `Plan.shape()` returns the cached value, keeping the 2D/3D Solve path at
      its prior allocs/op (2D-Dirichlet workers=1 stays 5 allocs/op — a per-Solve
      rebuild would make it 6). `PlanNDPeriodic` already held its shape built
      once. Shape is no longer comparable (slice field); nothing compared it with
      `==` or used it as a map key, so no call sites changed on that account.)
- [x] Deduplicate the pow2/non-pow2 strided-transform logic copy-pasted
      between `periodic_nd.go:263` and `fft_plan.go:120-157`.
      → Resolved by the Phase A concurrency refactor: the single
      `fftTransformLine` helper (`fft_plan.go`) now owns the pow2
      (`TransformStrided`) vs. non-pow2 (gather → `Forward`/`Inverse` →
      scatter) branching, and both `PlanNDPeriodic.transformAxis` and
      `FFTPlan.TransformLines` call it. The periodic 2D/3D solvers route through
      `TransformLines`, so there is exactly one copy of the strided logic.
- [x] Delete dead code: `isZeroMode` (`plan.go:258`), the unreachable
      lazy-grow in `plan_bc.go`. (Done: `AxisBC`/`NewAxisBC`,
      `Index1D`/`FromIndex1D`, and `fd.HasZeroEigenvalue` — the last now lives
      only as `bc.HasZeroEigenvalue`/`BCType.HasNullspace`.)
      → `isZeroMode` and the `plan_bc.go` lazy-grow are both gone; the
      per-mode divide checks `denom == 0` inline.
- [x] Parallel layer polish: lazy per-worker FFT plan allocation in
      `NewFFTPlanWithWorkers` (currently eager GOMAXPROCS × plans+scratch).
      (Done: `parallelFor` now cancels remaining workers on the first error via
      a per-call `context.WithCancel`, still returning the first error;
      `periodic_2d.go` now partitions the spectral divide over the larger of the
      two axes. The 1D-threshold-gate claim was inaccurate — 1D already clamps
      workers to the task count.)
      → Lazy allocation landed too: `newFFTWorkerPool` seeds a single worker
      into a `residentPool` of capacity `workers`; further workers are built
      on demand in `get()` only under real concurrency, so plan construction
      no longer eagerly allocates GOMAXPROCS × (plan + scratch).
- [x] Fix `sizeStr` benchmark labels (`fd/eigenvalues_test.go` — used
      `string(rune(...))`, broke for n ≥ 10240; now `strconv`-based) and
      size-brittle absolute tolerances in `fd/laplacian_test.go` (now scaled to
      the expected field magnitude via `eigTol`).

---

## Phase E: Demo Repair (or removal)

The shipped demo is broken end-to-end and actively misrepresents the library.
Either fix all of the below or pull it from the README until fixed.

- [x] **Wrong equation:** worker passed `alpha = +k²`, solving screened Poisson
      `(k² − Δ)p = s` — monotone decaying blobs, no room modes.
      → The WASM `SolveAcoustic` now solves the driven acoustic Helmholtz
      `(−k² − Δ)p = s` via `NewComplexHelmholtzPlan`/`SolveComplex` with
      `alpha = complex(−k², k²·η)` (= `−k²(1 − iη)`, `η = 0.03`). The imaginary
      shift bounds `|α + λ| ≥ |Im α| > 0`, so modes driven at resonance stay
      finite. The demo is re-branded honestly as an "Acoustic Room Modes" lab
      (single driving frequency, standing-wave field; no bogus time synthesis).
- [x] **Broken deploy:** root-absolute `fetch('/wasm_exec.js')` /
      `fetch('/acoustics.wasm')` 404 under the Pages subpath + Vite `base: './'`.
      → `main.ts` now resolves the asset URLs against `document.baseURI` and
      passes them to the worker (the worker bundle lives in `dist/assets/` but
      the WASM files land at the dist root, so a worker-relative `./` URL was
      wrong too). Verified loading under `http://localhost:8099/algo-pde/` with
      no 404s (headless Chromium / Playwright). `deploy-demo.yml` now uses
      `go-version-file: go.mod` instead of the dead `1.23` pin.
- [x] **Dead plan cache:** `InitPlan` cached a plan that was never used; `Solve`
      built a fresh plan per call (16 constructions per click).
      → Plans are now cached by `(nx,ny,dx,dy,bc,alpha)` in a plain map (single-
      threaded WASM — noted in a comment) and reused across clicks at the same
      frequency.
- [x] Replace the ~49k `jsArray.SetIndex` calls per solve with `js.CopyBytesToJS`.
      → Go builds the RGBA byte buffer (diverging colormap on `Re(p)`) and copies
      it into a `Uint8Array` with a single `js.CopyBytesToJS`.
- [x] Add timeout/error path to the WASM readiness poll (hung forever on a
      half-failed instantiation).
      → The poll now rejects after a 15 s deadline, and fetch failures for
      `wasm_exec.js`/`acoustics.wasm` throw with the status; errors are posted to
      `main.ts` and shown in the status line.
- [x] Fix README/demo-README claims (placeholder link, grid, damping, timing,
      "reflections").
      → `demo/README.md` and the root `README.md` were rewritten for the accurate
      acoustic room-modes description; the live-demo link points at
      `https://cwbudde.github.io/algo-pde/` and the stale synthesis/FPS/timing
      claims are gone.
- [x] Modern build constraint (`//go:build js && wasm`), `Release()` strategy /
      comment for `js.FuncOf`, note the single-threaded `planCache` assumption.
      → `main.go` starts with `//go:build js && wasm`; the callback lives for the
      program lifetime (main blocks forever), documented with why no `Release()`
      is needed; the plan-cache comment notes the single-threaded assumption.
- [x] **3D volumetric render.** The 3D box previously only offered a movable
      Z-slice. Added a **3D volume** view: a WebGL2 ray-marched render of the
      whole box as a rotatable translucent glow (antinodes opaque, nodes clear),
      with drag-to-orbit, scroll-to-zoom, and a density slider.
      → New `demo/volume.ts` (`VolumeRenderer`): front-to-back ray-marches the
      volume with premultiplied alpha, plus a faint wireframe box for reference.
      No solver/WASM change was needed — the diverging colour map is invertible
      (per-voxel amplitude `1 − min(r,g,b)`, sign from the red/blue side), so the
      cached RGBA is transcoded on the JS side to a signed scalar `R8`
      `TEXTURE_3D` and the colour map is re-applied in the shader. Filtering the
      signed field (not the encoded colour) is what keeps nodal zero-crossings
      transparent under `LINEAR` interpolation instead of blending to opaque
      purple. `main.ts` grew a three-way **2D room / 3D slices / 3D volume**
      toggle; both 3D views share the single volume solve, so switching between
      them never re-solves. Degrades gracefully (button disabled) without WebGL2.
      Verified end-to-end in headless Chromium (software WebGL): the box solves,
      renders the glowing modal field, and orbits on drag.

---

## Phase F: Hygiene & Release Readiness

- [x] **LICENSE file.** README says "TBD" while giving `go get` instructions;
      legally nobody may use the library today. Pick one (MIT/Apache-2.0).
      → Added the MIT license (`LICENSE`, © 2026 Christian Budde) and pointed
      the README License section at it.
- [x] `gofmt` the tree: 7 of 8 examples + the WASM main currently fail plain
      gofmt despite a formatting CI job. Then figure out why CI didn't catch
      it (format workflow scope).
      → The whole tree is now clean under both `gofmt -l .` and `gofumpt -l .`
      (empty output).
- [x] Remove root clutter: `check_fft.go` debug script, `coverage.out`,
      `poisson_cov2.out`, `poisson_newtests.out`, empty `internal/` dir,
      `goal.md` (superseded by this file).
      → Deleted `check_fft.go` and `goal.md`; the coverage `*.out` files and the
      `internal/` dir were already gone.
- [x] Commit or revert the local `.gitignore`/`.golangci.toml` drift; resolve
      the `demo/package-lock.json` tracked-but-gitignored contradiction (CI's
      `npm ci` depends on the lockfile).
      → Removed the `demo/package-lock.json` line from `.gitignore` (the file is
      tracked and CI's `npm ci` + cache-dependency-path need it); left all other
      entries intact. `.golangci.toml` reviewed and found sound — no change.
- [x] Align CI Go version with go.mod (CI pins 1.23, go.mod demands 1.25 —
      the pin is dead config via GOTOOLCHAIN).
      → `test-unit`/`test-lint` now use `go-version-file: go.mod` like
      `test-format` already did. The old `1.23` pin made `GOTOOLCHAIN=auto`
      download the minimal go1.25 toolchain module, which lacks the `covdata`
      tool and failed `go test -coverprofile` on the example main packages
      (all tests passed, but the job still exited 1). (`deploy-demo` still
      pins 1.23 — left for the Phase E demo work.)
- [x] Fix the two examples whose plan is never used (`examples/neumann2d`,
      `examples/periodic1d` — staticcheck SA4006); add a correctness check to
      `examples/helmholtz` (currently prints a number with no verification).
      → Removed the dead first `NewPlan`/`NewPlan1DPeriodic` from both examples,
      keeping the single `WithSubtractMean` plan. `examples/helmholtz` now
      reapplies the screened-Poisson operator (`alpha*u + fd.Apply2D(u) - rhs`,
      since `fd.Apply2D` is the negative Laplacian) and prints the max relative
      residual (~3.4e-14 measured).
- [x] Work through the substantive `golangci-lint` findings (exhaustive
      switch in `plan_bc.go:38`, dupl, staticcheck); decide policy on the
      style ones (varnamelen etc.) and configure the linter accordingly.
      → `golangci-lint run --timeout=5m` (v2.12.2) reports `0 issues` (only the
      benign `gomodguard is deprecated` warning remains). Style-linter policy is
      captured in `.golangci.toml`.
- [x] BENCHMARKS.md with honest numbers per BC type (carried over; blocked on
      A.2 — current Neumann numbers would be embarrassing).
      → Added `BENCHMARKS.md` with measured ns/op and allocs/op per BC type
      (Periodic/Dirichlet/Neumann) at 256²/512²/1024², a single-machine caveat,
      and notes on the now-linearithmic Neumann path and the Bluestein FFT-size
      penalty on the Dirichlet path.
- [x] Convergence log-log plots in docs (carried over).
      → `cmd/convergence-plot` runs the manufactured-solution studies for
      Dirichlet/Neumann/Periodic 2D and emits a self-contained log-log SVG
      (`docs/convergence-2d.svg`, no external plotting dependency) with a
      slope-2 reference guide; all three BCs measure order ≈ 2.00. Wired into
      the README's new Accuracy section and regenerated via
      `go generate ./docs/...` (`docs/doc.go`).

---

## Phase G: Future Extensions (unchanged ambitions, after the above)

### G.1 Full acoustic room demo ("Room Modes Lab")

The old Phase-13 vision — React UI, draggable source/mic, frequency sweep, mic
response plot, minimum-phase IR auralization via WebAudio ConvolverNode,
GitHub Pages deploy. Rectangular rooms only (separable spectral solver).
Prerequisites: Phase E fixed, Phase A.3 (negative α) decided. Details in git
history (`PLAN.md` @ 3acff0c, Phase 13).

### G.2 Solver features

- [x] True complex Helmholtz (Re/Im as two real solves) → correct phase,
      damping via complex shift.
      → `NewComplexHelmholtzPlan` + `SolveComplex(dst []complex128, rhs []float64)`
      solve `(α−Δ)u = f` for complex α. The existing transform pipeline is linear,
      so the real/imag parts of each mode ride through independently and a single
      complex per-mode divide gives the complex solution for every BC (no new
      transform). `imag(α) != 0` acts as a damping shift (never resonant); real α
      keeps the `ErrResonant` guard. Zero extra per-solve allocation vs the real
      path. `poisson/complex_helmholtz.go` + `_test.go` (residual for all BCs,
      real-path agreement, damping-vs-resonance, validation, alloc parity).
- [ ] Robin / per-face asymmetric BCs (the `AxisBC` promise, currently dead
      code — implement or drop).
- [x] Pressure projection API for incompressible flow (divergence, gradient,
      Navier–Stokes projection example).
      → `poisson/projection.go` adds `ProjectionPlan2D`/`ProjectionPlan3D`:
      `Project` performs the Helmholtz–Hodge decomposition `u = u* − ∇φ` with
      `Δφ = ∇·u*`, and `Divergence` exposes the discrete operator it drives to
      zero. The gradient uses forward differences and the divergence backward
      differences on the collocated periodic grid, so their composition D∘G is
      exactly the second-order periodic Laplacian the internal spectral plan
      inverts (the collocated analogue of MAC staggering). The projected field
      is therefore divergence-free to solver round-off, not just to truncation
      order — `examples/projection` drops max|div| from ~3e2 to ~2e-13, and
      `projection_test.go` pins divergence-free output, pure-gradient removal,
      idempotency, validation, and concurrent-Project agreement (under -race).
      The plan runs the inner solve with `WithSubtractMean` (the periodic
      divergence telescopes to an analytically zero mean) and reuses per-call
      scratch from a pool so `Project` stays allocation-free and concurrency-safe.
- [ ] Variable coefficients: spectral solve as preconditioner for an
      iterative method.
- [ ] Non-rectangular domains via immersed-boundary / masking, using the
      spectral solver as the fast inner solve.

### G.3 Performance

- [ ] SIMD for the eigenvalue-division loop (profile first).
- [ ] Real-input FFT (or compact DCT algorithms) instead of full complex128
      FFT of the 2(N±1) extension — currently ~4× redundant work — without
      the float32 downgrade of the current `WithRealFFT`.

---

## Success Criteria (revised)

1. **Honest contracts:** every claim in doc.go/README is enforced by a test
   (concurrency, complexity, allocations).
2. **Spectrum-complete correctness:** random-RHS residual tests pass for
   every BC combination in 1D/2D/3D; dense-reference comparison on random RHS.
3. **No silent failure:** invalid input → error; never a no-op, panic in an
   error-returning API, or garbage with err == nil.
4. **One source of truth** for eigenvalue formulas and shapes.
5. **Shippable repo:** LICENSE, gofmt-clean, working demo (or none), CI that
   actually gates what it claims to gate.
