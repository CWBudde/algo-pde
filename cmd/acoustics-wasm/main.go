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

// planCache memoizes plans across Solve calls. WASM here is single-threaded (Go
// scheduling inside one JS event loop, no SharedArrayBuffer / web-worker
// concurrency touches this map), so a plain map without a mutex is safe.
var planCache = make(map[planKey]*poisson.Plan)

func main() {
	// The callbacks below live for the entire lifetime of the program: main
	// blocks on the channel forever and never returns, so the js.Func values are
	// never eligible for release. There is therefore no Release() to call — if
	// this ever became a short-lived instance, each js.FuncOf would need a
	// matching Release() to avoid leaking the Go-side callback.
	js.Global().Set("goSolveAcoustic", js.FuncOf(SolveAcoustic))

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

	plan, err := poisson.NewComplexHelmholtzPlan(
		2,
		[]int{nx, ny},
		[]float64{dx, dy},
		[]poisson.BCType{poisson.BCType(bcX), poisson.BCType(bcY)},
		alpha,
	)
	if err != nil {
		return nil, err
	}

	planCache[key] = plan
	return plan, nil
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
