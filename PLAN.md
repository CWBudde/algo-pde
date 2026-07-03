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

- [ ] Write the conventions section in `poisson/doc.go`: node placement per
      BC, domain length per BC, Neumann sign convention, mixed-axis
      implications. Include an ASCII diagram.
- [ ] Fix the `bc.go` Neumann sign doc to match the implementation (or flip
      the implementation to outward-normal and migrate — decide once, now).
- [ ] Document eigenvalue formulas and memory layout (carried over from the
      old plan; still open).

---

## Phase B: Rebuild Test Trust

The suite probes modes 1–3 only; the general-RHS case — the point of a Poisson
solver — is essentially unverified. A corrupted eigenvalue above index 3 would
ship green today.

- [ ] **Random-RHS residual checks** (the single highest-leverage item): for
      every BC combination in 1D/2D/3D, solve a random RHS and assert
      `fd.Apply*(solution) ≈ rhs` to tight tolerance. This pins the entire
      spectrum, not just low modes.
- [ ] Feed the dense Gaussian-elimination reference
      (`reference_solver_test.go`) **random** RHS, and extend it beyond 2D
      homogeneous Dirichlet: Neumann, periodic, mixed, anisotropic h,
      Helmholtz, inhomogeneous BC.
- [ ] Give the fuzz test a property: currently `_ = plan.Solve(dst, rhs)`
      (`fuzz_test.go:58`) asserts nothing. Check err handling, finite output,
      and (for valid inputs) the residual. Un-tie `nz` from `nx` (line 27).
- [ ] Delete or fix vacuous assertions:
      `(u[0]-u[0])/h` in `neumann_1d_test.go:131` (identically zero),
      the `t.Logf`-only `TestDCTPlan_KnownValues` (`dct_test.go:117`),
      the `math.Sin(0)` checks in `dirichlet_1d_test.go:29,62`.
- [ ] Tighten the periodic convergence test: `0.6×` per halving accepts
      first-order schemes (`periodic_2d_test.go:160`); require rate ≥ 1.8 as
      `convergence_test.go` already does — and add strict-order tests for
      Neumann, periodic, mixed, and 3D.
- [ ] Replace the circular eigenvalue tests (`fd/eigenvalues_test.go` re-types
      the identical formula inline) with actual small-matrix
      eigendecomposition or brute-force stencil checks on random vectors.
- [ ] Helmholtz gaps: solve tests for negative non-resonant α, a
      near-resonant α test (1 ulp off — must return ErrResonant, not
      garbage), Neumann/periodic BC with α > 0.
- [ ] Cover the 0%-coverage surface: `Plan.SolveInPlace`,
      `Plan2DPeriodic.SolveInPlace`, `Plan3DPeriodic.SolveInPlace`,
      `WithNullspace`, `WithWorkers`, `WithInPlace`.
- [ ] Asymmetric Neumann data through `SolveWithBC` (currently only constant,
      symmetric faces — `inhom_api_test.go:50,134`; swapped/mirrored faces
      would pass).
- [ ] Add a naive-DFT reference test for DST-I/DCT-I amplitudes (only type-II
      has one), and a test pinning the Hermitian/Nyquist bin in the real-FFT
      path.

---

## Phase C: API Hardening — no more silent failure

House style today: bad input → silent no-op, silent garbage, or panic. Pick
one contract (errors) and enforce it.

- [ ] `fd.Apply1D/2D/3D`: size mismatch is a silent no-op leaving stale data
      in dst (`fd/laplacian.go:13,82,165`) — return an error. Unknown BCType
      silently becomes zeros in `Eigenvalues` and Dirichlet in `Apply2D/3D` —
      error in both.
- [ ] `r2r` lines API: panics on short buffers, bad axis, zero-extent shapes
      (`r2r/lines.go:61-87`) — validate and return `ErrSizeMismatch` /
      `ErrInvalidAxis`.
- [ ] `grid`: `LineIterator`/`PlaneIterator` yield phantom lines for shapes
      with a zero extent (`grid/grid.go:149-206`); validate shape and axis at
      construction. Reject negative extents in `NewShape*`.
- [ ] `grid.Shape.Dim()` infers dimension from trailing extents, so 64×64×1
      reports 2D and `SolveWithBC` rejects its Y faces — store the declared
      dimension instead of guessing.
- [ ] Options that silently no-op — make each either work or error:
      `WithWorkers` on `PlanNDPeriodic` (ignored entirely),
      `WithSolutionMean` on plans without nullspace (`plan.go:174`),
      `WithRealFFT` on the BC-plan path, `WithInPlace` on periodic plans.
- [ ] `SolveWithBC`: reject duplicate faces (two `XLow` entries silently
      double the contribution); don't corrupt the caller's rhs on a mid-loop
      error in the `InPlace` path (`plan_bc.go:36-62`).
- [ ] `NormOrtho` DCT-I is not orthonormal (missing √2 endpoint weights) —
      fix or document; Parseval-based uses are silently wrong today.
- [ ] `NormalizationFactor` is a dead API whose documented semantics are
      wrong by O(N) (`r2r/dct.go:293`, `transform.go:21`) — delete it or fix
      doc + implementation to agree.
- [ ] Replace `log` calls on the real-FFT fallback (`periodic_2d.go:54`) with
      an inspectable plan property (`p.UsedRealFFT() bool`) or an option-level
      error.
- [ ] Fix misleading fd docs: formulas are for the **negative** Laplacian;
      `fd/doc.go` says "Laplacian".

---

## Phase D: Structural Debt

- [ ] **Un-duplicate the eigenvalue formulas.** `fd` imports `poisson` for
      `BCType`, so `poisson` carries verbatim copies
      (`eigenvalues_bc.go`, `eigenvalues_periodic.go`) of `fd/eigenvalues.go`.
      Extract a leaf package (e.g. `bc/`: BCType + eigenvalue formulas) that
      both import. This is the most likely source of a future silent
      numerical divergence.
- [ ] One `Shape` type. `grid.Shape` (fixed `[3]int`) and `poisson`'s
      `Shape []int` (`shape_nd.go`) coexist; `parallel.go`'s helpers hardcode
      3 axes and would silently miscount for >3D.
- [ ] Deduplicate the pow2/non-pow2 strided-transform logic copy-pasted
      between `periodic_nd.go:263` and `fft_plan.go:120-157`.
- [ ] Delete dead code: `isZeroMode` (`plan.go:258`), `AxisBC`/`NewAxisBC`
      (`bc.go:41`), `Index1D`/`FromIndex1D` (`grid/grid.go:57,77`), the
      unreachable lazy-grow in `plan_bc.go`, `fd.HasZeroEigenvalue` (verbatim
      duplicate of `BCType.HasNullspace`).
- [ ] Parallel layer polish: propagate/cancel on first worker error instead
      of dropping the rest (`parallel.go:57`); threshold gate so 1D solves
      don't spawn GOMAXPROCS goroutines for a pointwise division
      (`periodic_1d.go:85`); partition over the largest dimension, not always
      nx (`periodic_2d.go:136`); lazy per-worker FFT plan allocation in
      `NewFFTPlanWithWorkers` (currently eager GOMAXPROCS × plans+scratch).
- [ ] Fix `sizeStr` benchmark labels (`fd/eigenvalues_test.go:190` — breaks
      for n ≥ 10240) and size-brittle absolute tolerances in
      `fd/laplacian_test.go` (use relative error).

---

## Phase E: Demo Repair (or removal)

The shipped demo is broken end-to-end and actively misrepresents the library.
Either fix all of the below or pull it from the README until fixed.

- [ ] **Wrong equation:** worker passes `alpha = +k²` (`demo/sim.worker.ts:250`),
      solving screened Poisson `(k² − Δ)p = s` — monotone decaying blobs, no
      waves, no room modes. Acoustic Helmholtz needs `alpha = −k²` — which
      requires Phase A.3's near-resonance handling plus a damping strategy
      (complex shift via two real solves, or stay with the screened form and
      re-brand the demo honestly as a "room modes / decay-length lab").
- [ ] **Broken deploy:** root-absolute `fetch('/wasm_exec.js')` /
      `fetch('/acoustics.wasm')` (`sim.worker.ts:175,188`) 404 under the
      GitHub-Pages subpath + Vite `base: './'`. Use relative URLs.
- [ ] **Dead plan cache:** `InitPlan` caches a plan that is never used;
      `Solve` builds a fresh plan per call (`cmd/acoustics-wasm/main.go:80`
      vs `:131`) — 16 plan constructions per click. Cache by
      `(nx,ny,dx,dy,bc,alpha-independent)` and reuse.
- [ ] Replace the ~49k `jsArray.SetIndex` calls per solve with
      `js.CopyBytesToJS` over a `Uint8Array` view (`main.go:158` — the
      comment claiming this is impossible is wrong).
- [ ] Add timeout/error path to the WASM readiness poll
      (`sim.worker.ts:196` — hangs forever on half-failed instantiation).
- [ ] Fix README/demo-README claims: `yourusername` placeholder link, grid
      256×192 not 256×256, damping γ = ω/20 not 0.5·f, "80ms well under
      16.67ms" arithmetic, "reflections" claim.
- [ ] Modern build constraint in `cmd/acoustics-wasm/main.go:1`
      (`//go:build js && wasm`), `Release()` strategy or comment for
      `js.FuncOf` callbacks, note the single-threaded assumption on
      `planCache`.

---

## Phase F: Hygiene & Release Readiness

- [ ] **LICENSE file.** README says "TBD" while giving `go get` instructions;
      legally nobody may use the library today. Pick one (MIT/Apache-2.0).
- [ ] `gofmt` the tree: 7 of 8 examples + the WASM main currently fail plain
      gofmt despite a formatting CI job. Then figure out why CI didn't catch
      it (format workflow scope).
- [ ] Remove root clutter: `check_fft.go` debug script, `coverage.out`,
      `poisson_cov2.out`, `poisson_newtests.out`, empty `internal/` dir,
      `goal.md` (superseded by this file).
- [ ] Commit or revert the local `.gitignore`/`.golangci.toml` drift; resolve
      the `demo/package-lock.json` tracked-but-gitignored contradiction (CI's
      `npm ci` depends on the lockfile).
- [x] Align CI Go version with go.mod (CI pins 1.23, go.mod demands 1.25 —
      the pin is dead config via GOTOOLCHAIN).
      → `test-unit`/`test-lint` now use `go-version-file: go.mod` like
      `test-format` already did. The old `1.23` pin made `GOTOOLCHAIN=auto`
      download the minimal go1.25 toolchain module, which lacks the `covdata`
      tool and failed `go test -coverprofile` on the example main packages
      (all tests passed, but the job still exited 1). (`deploy-demo` still
      pins 1.23 — left for the Phase E demo work.)
- [ ] Fix the two examples whose plan is never used (`examples/neumann2d`,
      `examples/periodic1d` — staticcheck SA4006); add a correctness check to
      `examples/helmholtz` (currently prints a number with no verification).
- [ ] Work through the substantive `golangci-lint` findings (exhaustive
      switch in `plan_bc.go:38`, dupl, staticcheck); decide policy on the
      style ones (varnamelen etc.) and configure the linter accordingly.
- [ ] BENCHMARKS.md with honest numbers per BC type (carried over; blocked on
      A.2 — current Neumann numbers would be embarrassing).
- [ ] Convergence log-log plots in docs (carried over).

---

## Phase G: Future Extensions (unchanged ambitions, after the above)

### G.1 Full acoustic room demo ("Room Modes Lab")

The old Phase-13 vision — React UI, draggable source/mic, frequency sweep, mic
response plot, minimum-phase IR auralization via WebAudio ConvolverNode,
GitHub Pages deploy. Rectangular rooms only (separable spectral solver).
Prerequisites: Phase E fixed, Phase A.3 (negative α) decided. Details in git
history (`PLAN.md` @ 3acff0c, Phase 13).

### G.2 Solver features

- [ ] True complex Helmholtz (Re/Im as two real solves) → correct phase,
      damping via complex shift.
- [ ] Robin / per-face asymmetric BCs (the `AxisBC` promise, currently dead
      code — implement or drop).
- [ ] Pressure projection API for incompressible flow (divergence, gradient,
      Navier–Stokes projection example).
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
