//go:build js && wasm

// Command acoustics-wasm is the WebAssembly backend for the browser demo. It
// solves the driven acoustic Helmholtz equation on a rectangular "room" with
// rigid (Neumann) walls and returns a ready-to-blit RGBA image of the
// steady-state pressure field.
//
// Physics. For a harmonic drive at angular frequency ω = 2πf and sound speed c,
// the wavenumber is k = ω/c and the (undamped) Helmholtz equation is
//
//	∇²p + k²p = -s   ⇔   (−k² − Δ)p = s.
//
// In this library's plan form (α − Δ)u = f that means α = −k². The library's
// discrete −Δ has non-negative eigenvalues λ, so the per-mode denominator is
// (−k² + λ); it vanishes whenever a room mode's eigenvalue matches k² — a
// resonance where the undamped field is infinite. To keep |p| finite we add a
// small imaginary (damping) shift and use the complex solver:
//
//	alpha = complex(-k*k, k*k*eta)   //  = −k²(1 − iη),  η a small damping fraction
//
// Any nonzero imaginary part bounds |α + λ| ≥ |Im α| > 0, so the divide never
// blows up near a mode. We use a positive imaginary part (the e^{+iωt}
// convention); the rendered magnitude is independent of that sign — only its
// nonzero-ness matters for finiteness.
package main

import (
	"fmt"
	"math"
	"syscall/js"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

// planKey identifies a cached complex-Helmholtz plan. A plan bakes in its grid,
// boundary conditions and α (which depends on the driving frequency), so any
// change to those requires a fresh plan; repeated clicks at the same frequency
// reuse one.
type planKey struct {
	nx, ny   int
	dx, dy   float64
	bcX, bcY int
	// alpha is quantized into the key via its bit patterns so equal α values map
	// to the same entry.
	alphaReBits, alphaImBits uint64
}

// planKey3D is the 3D analogue of planKey; it identifies a cached complex-
// Helmholtz volume plan.
type planKey3D struct {
	nx, ny, nz    int
	dx, dy, dz    float64
	bcX, bcY, bcZ int
	alphaReBits   uint64
	alphaImBits   uint64
}

// planCache memoizes plans across Solve calls. WASM here is single-threaded (Go
// scheduling inside one JS event loop, no SharedArrayBuffer / web-worker
// concurrency touches this map), so a plain map without a mutex is safe.
//
// A frequency sweep produces a distinct alpha (hence a distinct plan) per step,
// so the cache is bounded: once it reaches maxCachedPlans it is dropped and
// rebuilt, keeping memory flat instead of accumulating hundreds of large plans.
const maxCachedPlans = 32

var planCache = make(map[planKey]*poisson.Plan)

// planCache3D is the 3D counterpart to planCache, bounded the same way.
var planCache3D = make(map[planKey3D]*poisson.Plan)

func main() {
	// The callbacks below live for the entire lifetime of the program: main
	// blocks on the channel forever and never returns, so the js.Func values are
	// never eligible for release. There is therefore no Release() to call — if
	// this ever became a short-lived instance, each js.FuncOf would need a
	// matching Release() to avoid leaking the Go-side callback.
	js.Global().Set("goSolveAcoustic", js.FuncOf(SolveAcoustic))
	js.Global().Set("goSolveAcoustic3D", js.FuncOf(SolveAcoustic3D))

	// Signal to the JS side that exports are installed.
	js.Global().Set("goReady", js.ValueOf(true))

	<-make(chan struct{})
}

// SolveAcoustic solves the driven acoustic Helmholtz problem and returns an
// RGBA image of the steady-state field.
//
// Args (all numbers):
//
//	nx, ny        grid dimensions (cells)
//	dx, dy        cell spacing (metres)
//	bcX, bcY      boundary codes: 0=Periodic, 1=Dirichlet, 2=Neumann
//	freqHz        driving frequency (Hz)
//	soundSpeed    speed of sound (m/s)
//	eta           damping fraction (small positive, e.g. 0.03)
//	sx, sy        source centre in grid-cell coordinates
//	srcRadius     Gaussian source radius (cells)
//
// Returns { success, error?, width, height, k, lambda, rgba } where rgba is a
// Uint8Array of length width*height*4 filled via js.CopyBytesToJS (one bulk
// copy — not per-element SetIndex).
func SolveAcoustic(_ js.Value, args []js.Value) interface{} {
	if len(args) != 12 {
		return jsError("SolveAcoustic requires 12 arguments: nx, ny, dx, dy, bcX, bcY, freqHz, soundSpeed, eta, sx, sy, srcRadius")
	}

	nx := args[0].Int()
	ny := args[1].Int()
	dx := args[2].Float()
	dy := args[3].Float()
	bcX := args[4].Int()
	bcY := args[5].Int()
	freqHz := args[6].Float()
	soundSpeed := args[7].Float()
	eta := args[8].Float()
	sx := args[9].Float()
	sy := args[10].Float()
	srcRadius := args[11].Float()

	if nx < 1 || ny < 1 {
		return jsError("grid dimensions must be positive")
	}
	if dx <= 0 || dy <= 0 {
		return jsError("grid spacing must be positive")
	}
	if bcX < 0 || bcX > 2 || bcY < 0 || bcY > 2 {
		return jsError("boundary conditions must be 0 (Periodic), 1 (Dirichlet), or 2 (Neumann)")
	}
	if soundSpeed <= 0 {
		return jsError("sound speed must be positive")
	}
	if eta <= 0 {
		// A strictly positive damping fraction is required to keep |p| finite at
		// resonance; without it a driven mode diverges (ErrResonant).
		return jsError("damping fraction eta must be positive")
	}

	// Acoustic Helmholtz: k = ω/c, alpha = −k²(1 − iη) = complex(−k², k²·η).
	omega := 2 * math.Pi * freqHz
	k := omega / soundSpeed
	alpha := complex(-k*k, k*k*eta)

	plan, err := getPlan(nx, ny, dx, dy, bcX, bcY, alpha)
	if err != nil {
		return jsError(fmt.Sprintf("failed to create plan: %v", err))
	}

	rhs := buildGaussianSource(nx, ny, sx, sy, srcRadius)

	u := make([]complex128, nx*ny)
	if err := plan.SolveComplex(u, rhs); err != nil {
		return jsError(fmt.Sprintf("solve failed: %v", err))
	}

	rgba := fieldToRGBA(u)

	// One bulk copy of the byte buffer into a JS Uint8Array.
	jsBuf := js.Global().Get("Uint8Array").New(len(rgba))
	js.CopyBytesToJS(jsBuf, rgba)

	return jsSuccess(map[string]interface{}{
		"width":  nx,
		"height": ny,
		"k":      k,
		"lambda": twoPiOver(k), // acoustic wavelength (m); 0 at DC
		"rgba":   jsBuf,
	})
}

// getPlan returns a cached plan for the given parameters, building (and caching)
// one on first use.
func getPlan(nx, ny int, dx, dy float64, bcX, bcY int, alpha complex128) (*poisson.Plan, error) {
	key := planKey{
		nx:          nx,
		ny:          ny,
		dx:          dx,
		dy:          dy,
		bcX:         bcX,
		bcY:         bcY,
		alphaReBits: math.Float64bits(real(alpha)),
		alphaImBits: math.Float64bits(imag(alpha)),
	}

	if plan, ok := planCache[key]; ok {
		return plan, nil
	}

	// Axis order must match the field's memory layout. The source and the
	// rendered image are row-major with x contiguous (index = y*nx+x), so y is
	// the outer/slow axis and x the inner/contiguous one. The solver reads dim 0
	// as the outer axis and the last dim as contiguous, so the plan is built as
	// {y, x}: extents {ny, nx}, spacings {dy, dx}, boundaries {bcY, bcX}.
	// Passing {nx, ny} instead transposes the operator relative to the buffer;
	// with nx != ny the resulting stride mismatch aliases the FFT and renders as
	// a tiled, striped field rather than the true room response.
	plan, err := poisson.NewComplexHelmholtzPlan(
		2,
		[]int{ny, nx},
		[]float64{dy, dx},
		[]poisson.BCType{poisson.BCType(bcY), poisson.BCType(bcX)},
		alpha,
	)
	if err != nil {
		return nil, err
	}

	if len(planCache) >= maxCachedPlans {
		planCache = make(map[planKey]*poisson.Plan, maxCachedPlans)
	}
	planCache[key] = plan
	return plan, nil
}

// getPlan3D is the 3D counterpart to getPlan.
func getPlan3D(nx, ny, nz int, dx, dy, dz float64, bcX, bcY, bcZ int, alpha complex128) (*poisson.Plan, error) {
	key := planKey3D{
		nx:          nx,
		ny:          ny,
		nz:          nz,
		dx:          dx,
		dy:          dy,
		dz:          dz,
		bcX:         bcX,
		bcY:         bcY,
		bcZ:         bcZ,
		alphaReBits: math.Float64bits(real(alpha)),
		alphaImBits: math.Float64bits(imag(alpha)),
	}

	if plan, ok := planCache3D[key]; ok {
		return plan, nil
	}

	// Axis order must match the volume's memory layout. The source and rendered
	// planes are laid out with x contiguous and z the slowest axis
	// (index = z*ny*nx + y*nx + x), so the plan is built as {z, y, x}: extents
	// {nz, ny, nx}, spacings {dz, dy, dx}, boundaries {bcZ, bcY, bcX}. Any other
	// order transposes the operator relative to the buffer and aliases the FFT.
	plan, err := poisson.NewComplexHelmholtzPlan(
		3,
		[]int{nz, ny, nx},
		[]float64{dz, dy, dx},
		[]poisson.BCType{poisson.BCType(bcZ), poisson.BCType(bcY), poisson.BCType(bcX)},
		alpha,
	)
	if err != nil {
		return nil, err
	}

	if len(planCache3D) >= maxCachedPlans {
		planCache3D = make(map[planKey3D]*poisson.Plan, maxCachedPlans)
	}
	planCache3D[key] = plan
	return plan, nil
}

// SolveAcoustic3D is the volumetric analogue of SolveAcoustic: it solves the
// driven acoustic Helmholtz problem on a rectangular box and returns an RGBA
// image of the entire volume, stored plane-by-plane so the caller can slice it
// cheaply (the browser demo shows one movable Z-slice without re-solving).
//
// Args (all numbers):
//
//	nx, ny, nz        grid dimensions (cells)
//	dx, dy, dz        cell spacing (metres)
//	bcX, bcY, bcZ     boundary codes: 0=Periodic, 1=Dirichlet, 2=Neumann
//	freqHz            driving frequency (Hz)
//	soundSpeed        speed of sound (m/s)
//	eta               damping fraction (small positive, e.g. 0.03)
//	sx, sy, sz        source centre in grid-cell coordinates
//	srcRadius         Gaussian source radius (cells)
//
// Returns { success, error?, width, height, depth, k, lambda, rgba } where rgba
// is a Uint8Array of length width*height*depth*4. Plane z occupies the byte
// range [z*width*height*4, (z+1)*width*height*4); each plane is a row-major
// width×height image identical in layout to the 2D result. The colour scale is
// normalized once over the whole volume, so slices share a consistent mapping.
func SolveAcoustic3D(_ js.Value, args []js.Value) interface{} {
	if len(args) != 16 {
		return jsError("SolveAcoustic3D requires 16 arguments: nx, ny, nz, dx, dy, dz, bcX, bcY, bcZ, freqHz, soundSpeed, eta, sx, sy, sz, srcRadius")
	}

	nx := args[0].Int()
	ny := args[1].Int()
	nz := args[2].Int()
	dx := args[3].Float()
	dy := args[4].Float()
	dz := args[5].Float()
	bcX := args[6].Int()
	bcY := args[7].Int()
	bcZ := args[8].Int()
	freqHz := args[9].Float()
	soundSpeed := args[10].Float()
	eta := args[11].Float()
	sx := args[12].Float()
	sy := args[13].Float()
	sz := args[14].Float()
	srcRadius := args[15].Float()

	if nx < 1 || ny < 1 || nz < 1 {
		return jsError("grid dimensions must be positive")
	}
	if dx <= 0 || dy <= 0 || dz <= 0 {
		return jsError("grid spacing must be positive")
	}
	if bcX < 0 || bcX > 2 || bcY < 0 || bcY > 2 || bcZ < 0 || bcZ > 2 {
		return jsError("boundary conditions must be 0 (Periodic), 1 (Dirichlet), or 2 (Neumann)")
	}
	if soundSpeed <= 0 {
		return jsError("sound speed must be positive")
	}
	if eta <= 0 {
		return jsError("damping fraction eta must be positive")
	}

	omega := 2 * math.Pi * freqHz
	k := omega / soundSpeed
	alpha := complex(-k*k, k*k*eta)

	plan, err := getPlan3D(nx, ny, nz, dx, dy, dz, bcX, bcY, bcZ, alpha)
	if err != nil {
		return jsError(fmt.Sprintf("failed to create plan: %v", err))
	}

	rhs := buildGaussianSource3D(nx, ny, nz, sx, sy, sz, srcRadius)

	u := make([]complex128, nx*ny*nz)
	if err := plan.SolveComplex(u, rhs); err != nil {
		return jsError(fmt.Sprintf("solve failed: %v", err))
	}

	// fieldToRGBA maps every element to an RGBA quad, so for the flat volume it
	// yields nz contiguous width×height planes normalized over the whole box.
	rgba := fieldToRGBA(u)

	jsBuf := js.Global().Get("Uint8Array").New(len(rgba))
	js.CopyBytesToJS(jsBuf, rgba)

	return jsSuccess(map[string]interface{}{
		"width":  nx,
		"height": ny,
		"depth":  nz,
		"k":      k,
		"lambda": twoPiOver(k),
		"rgba":   jsBuf,
	})
}

// buildGaussianSource returns a narrow Gaussian bump centred at (sx, sy) in
// grid-cell coordinates.
func buildGaussianSource(nx, ny int, sx, sy, radius float64) []float64 {
	if radius <= 0 {
		radius = 1
	}
	rhs := make([]float64, nx*ny)
	twoR2 := 2.0 * radius * radius

	for y := range ny {
		dy := float64(y) - sy
		for x := range nx {
			dx := float64(x) - sx
			rhs[y*nx+x] = math.Exp(-(dx*dx + dy*dy) / twoR2)
		}
	}
	return rhs
}

// buildGaussianSource3D returns a narrow Gaussian bump centred at (sx, sy, sz)
// in grid-cell coordinates, laid out with x contiguous and z slowest.
func buildGaussianSource3D(nx, ny, nz int, sx, sy, sz, radius float64) []float64 {
	if radius <= 0 {
		radius = 1
	}
	rhs := make([]float64, nx*ny*nz)
	twoR2 := 2.0 * radius * radius

	for z := range nz {
		dz := float64(z) - sz
		for y := range ny {
			dy := float64(y) - sy
			for x := range nx {
				dx := float64(x) - sx
				rhs[z*ny*nx+y*nx+x] = math.Exp(-(dx*dx + dy*dy + dz*dz) / twoR2)
			}
		}
	}
	return rhs
}

// fieldToRGBA maps the real part of the complex pressure field to a blue-white-
// red diverging colormap, symmetric about zero so standing-wave nodes and
// antinodes read cleanly. A gamma < 1 lifts low-amplitude detail.
func fieldToRGBA(u []complex128) []byte {
	maxAbs := 1e-30
	for _, v := range u {
		if a := math.Abs(real(v)); a > maxAbs {
			maxAbs = a
		}
	}

	rgba := make([]byte, len(u)*4)
	for i, v := range u {
		t := real(v) / maxAbs // roughly [-1, 1]
		if t > 1 {
			t = 1
		} else if t < -1 {
			t = -1
		}
		s := math.Copysign(math.Pow(math.Abs(t), 0.7), t)

		var r, g, b byte
		if s < 0 {
			// blue -> white
			c := byte(255 * (1 + s))
			r, g, b = c, c, 255
		} else {
			// white -> red
			c := byte(255 * (1 - s))
			r, g, b = 255, c, c
		}

		rgba[i*4+0] = r
		rgba[i*4+1] = g
		rgba[i*4+2] = b
		rgba[i*4+3] = 255
	}
	return rgba
}

// twoPiOver returns 2π/k (the wavelength) or 0 when k is ~0.
func twoPiOver(k float64) float64 {
	if k <= 1e-12 {
		return 0
	}
	return 2 * math.Pi / k
}

func jsSuccess(data map[string]interface{}) interface{} {
	result := map[string]interface{}{"success": true}
	for key, value := range data {
		result[key] = value
	}
	return result
}

func jsError(message string) interface{} {
	return map[string]interface{}{
		"success": false,
		"error":   message,
	}
}
