package r2r

import (
	"fmt"
	"math"

	algofft "github.com/cwbudde/algo-fft"
)

// DCTPlan is a pre-computed Discrete Cosine Transform plan.
// DCT is used for Neumann boundary conditions where ∂u/∂n = 0 at boundaries.
//
// This implements DCT-I (Type I), which is appropriate for the standard
// second-order finite difference Laplacian with Neumann BCs.
//
// For input x[0..N-1], the DCT-I is defined as:
//
//	X[k] = x[0] + (-1)^k * x[N-1] + 2 * Σ x[n] * cos(πnk/(N-1)) for n=1..N-2
//
// For N points, DCT-I produces N coefficients.
// Note: DCT-I requires N >= 2.
//
// Thread safety: A single DCTPlan instance is NOT safe for concurrent use.
// For parallel transforms, create separate plan instances per goroutine.
type DCTPlan struct {
	n    int // Original transform size
	opts Options

	// Extended FFT size: 2*(N-1) for DCT-I
	extendedN int

	// Underlying complex FFT plan for the extended size
	fftPlan *algofft.Plan[complex128]

	// Pre-allocated buffers
	// Note: We use separate input and output buffers because the algo-fft
	// library has issues with in-place FFT for certain sizes.
	fftIn  []complex128 // FFT input buffer
	fftOut []complex128 // FFT output buffer
}

// DCT2Plan is a pre-computed Discrete Cosine Transform plan (Type II).
//
// For input x[0..N-1], the DCT-II is defined as:
//
//	X[k] = Σ x[n] * cos(π(n+1/2)k/N) for k = 0..N-1
//
// The inverse is the weighted transpose of the DCT-II kernel.
//
// Thread safety: A single DCT2Plan instance is NOT safe for concurrent use.
type DCT2Plan struct {
	n    int // Original transform size
	opts Options

	// Extended FFT size: 2*N for DCT-II
	extendedN int

	// Underlying complex FFT plan for the extended size
	fftPlan *algofft.Plan[complex128]

	// Pre-allocated buffers
	fftIn  []complex128 // FFT input buffer
	fftOut []complex128 // FFT output buffer
	phase  []complex128 // exp(-i*pi*k/(2N)) phase factors
}

// NewDCTPlan creates a new DCT-I plan for the given size.
// The size n must be at least 2.
func NewDCTPlan(n int, opts ...Option) (*DCTPlan, error) {
	if n < 2 {
		return nil, ErrInvalidSize
	}

	// DCT-I via FFT uses even extension: embed n points into 2*(n-1) points
	// x[0..n-1] -> [x[0], x[1], ..., x[n-1], x[n-2], ..., x[1]]
	extendedN := 2 * (n - 1)

	fftPlan, err := algofft.NewPlan64(extendedN)
	if err != nil {
		return nil, fmt.Errorf("creating FFT plan: %w", err)
	}

	return &DCTPlan{
		n:         n,
		opts:      applyOptions(opts),
		extendedN: extendedN,
		fftPlan:   fftPlan,
		fftIn:     make([]complex128, extendedN),
		fftOut:    make([]complex128, extendedN),
	}, nil
}

// NewDCT2Plan creates a new DCT-II plan for the given size.
// The size n must be at least 1.
func NewDCT2Plan(n int, opts ...Option) (*DCT2Plan, error) {
	if n < 1 {
		return nil, ErrInvalidSize
	}

	extendedN := 2 * n

	fftPlan, err := algofft.NewPlan64(extendedN)
	if err != nil {
		return nil, fmt.Errorf("creating FFT plan: %w", err)
	}

	phase := make([]complex128, n)
	den := 2.0 * float64(n)
	for k := range n {
		angle := -math.Pi * float64(k) / den
		phase[k] = complex(math.Cos(angle), math.Sin(angle))
	}

	return &DCT2Plan{
		n:         n,
		opts:      applyOptions(opts),
		extendedN: extendedN,
		fftPlan:   fftPlan,
		fftIn:     make([]complex128, extendedN),
		fftOut:    make([]complex128, extendedN),
		phase:     phase,
	}, nil
}

// Len returns the transform size.
func (p *DCTPlan) Len() int {
	return p.n
}

// Len returns the transform size.
func (p *DCT2Plan) Len() int {
	return p.n
}

// Forward computes the forward DCT-I transform.
// dst and src must have length n. They may be the same slice for in-place operation.
//
// Output normalization: The output is NOT normalized.
// For orthogonal normalization, divide by sqrt(2*(N-1)).
func (p *DCTPlan) Forward(dst, src []float64) error {
	if len(dst) != p.n || len(src) != p.n {
		return ErrSizeMismatch
	}

	// Clear input buffer
	for i := range p.extendedN {
		p.fftIn[i] = 0
	}

	// Build even-symmetric extension:
	// Position 0..n-1: x[0..n-1]
	// Position n..2n-3: x[n-2..1] (mirror without endpoints)
	for i := range p.n {
		p.fftIn[i] = complex(src[i], 0)
	}

	for i := 1; i < p.n-1; i++ {
		p.fftIn[p.extendedN-i] = complex(src[i], 0)
	}

	// FFT with separate input/output buffers (avoids in-place FFT issues)
	err := p.fftPlan.Forward(p.fftOut, p.fftIn)
	if err != nil {
		return fmt.Errorf("FFT forward: %w", err)
	}

	// Extract DCT coefficients from real parts
	// The DCT-I coefficients are in the real part of FFT output at positions 0..n-1
	scale := 1.0
	if p.opts.Normalization == NormOrtho {
		scale = 1.0 / math.Sqrt(2.0*float64(p.n-1))
	}

	for k := range p.n {
		dst[k] = real(p.fftOut[k]) * scale
	}

	return nil
}

// Forward computes the forward DCT-II transform.
// dst and src must have length n. They may be the same slice for in-place operation.
//
// Output normalization: The output is NOT normalized.
// For orthogonal normalization, multiply by sqrt(2/N) with k=0 scaled by 1/sqrt(N).
func (p *DCT2Plan) Forward(dst, src []float64) error {
	if len(dst) != p.n || len(src) != p.n {
		return ErrSizeMismatch
	}

	for i := range p.extendedN {
		p.fftIn[i] = 0
	}

	// Even extension (mirror the sequence, repeating both endpoints):
	// x[0..n-1] -> [x[0], ..., x[n-1] | x[n-1], ..., x[0]]
	for i := range p.n {
		p.fftIn[i] = complex(src[i], 0)
		p.fftIn[p.extendedN-1-i] = complex(src[i], 0)
	}

	err := p.fftPlan.Forward(p.fftOut, p.fftIn)
	if err != nil {
		return fmt.Errorf("FFT forward: %w", err)
	}

	for k := range p.n {
		shifted := p.fftOut[k] * p.phase[k]
		value := real(shifted) / 2.0

		if p.opts.Normalization == NormOrtho {
			scale := math.Sqrt(2.0 / float64(p.n))
			if k == 0 {
				scale = 1.0 / math.Sqrt(float64(p.n))
			}

			value *= scale
		}

		dst[k] = value
	}

	return nil
}

// Inverse computes the inverse DCT-I transform.
// dst and src must have length n. They may be the same slice for in-place operation.
//
// Note: DCT-I is its own inverse up to scaling. The inverse is:
// x[n] = (1/(N-1)) * DCT-I(X)[n] for interior points.
// x[0] and x[N-1] are scaled by 1/(2*(N-1)).
func (p *DCTPlan) Inverse(dst, src []float64) error {
	// DCT-I is self-inverse (up to normalization)
	err := p.Forward(dst, src)
	if err != nil {
		return err
	}

	// Apply normalization factor: 1/(2*(N-1)) for endpoints, 1/(N-1) for interior
	// For uniform scaling in spectral methods, we use 1/(N-1) everywhere
	// and adjust the endpoint weights in the eigenvalue computation
	scale := 1.0 / float64(p.extendedN)
	if p.opts.Normalization == NormOrtho {
		scale = 1.0
	}

	for i := range p.n {
		dst[i] *= scale
	}

	return nil
}

// Inverse computes the inverse DCT-II transform (a DCT-III up to scaling).
// dst and src must have length n. They may be the same slice for in-place operation.
//
// It evaluates x[n] = Σ_k w[k]·X[k]·cos(π(n+½)k/N) — the weighted transpose of
// the DCT-II kernel — in O(N log N) via a single 2N-point FFT rather than the
// naive O(N²) double loop. The weighted coefficients w[k]·X[k] are packed with
// the plan's phase factors so that Re(FFT(·)) yields the DCT-III result
// directly. src is fully consumed into the plan's FFT buffer before dst is
// written, so aliasing (Inverse(buf, buf)) is safe without an extra copy.
func (p *DCT2Plan) Inverse(dst, src []float64) error {
	if len(dst) != p.n || len(src) != p.n {
		return ErrSizeMismatch
	}

	n := p.n
	ortho := p.opts.Normalization == NormOrtho

	// The FFT input is kept purely real: the cosine component is packed with
	// even symmetry and the sine component with odd symmetry around the 2N
	// midpoint, so that Re(FFT)+Im(FFT) reconstructs the DCT-III. Feeding real
	// input also sidesteps an upstream complex-input FFT defect at some
	// composite sizes (e.g. 2*100), which the real-only forward never triggers.
	for i := range p.extendedN {
		p.fftIn[i] = 0
	}

	for k := range n {
		weight := 2.0 / float64(n)
		if k == 0 {
			weight = 1.0 / float64(n)
		}

		if ortho {
			weight = math.Sqrt(2.0 / float64(n))
			if k == 0 {
				weight = 1.0 / math.Sqrt(float64(n))
			}
		}

		a := src[k] * weight
		// cos(πk/2N), sin(πk/2N) recovered from the stored phase factor
		// phase[k] = exp(-iπk/2N).
		cosPart := a * real(p.phase[k])
		sinPart := a * -imag(p.phase[k])

		if k == 0 {
			p.fftIn[0] = complex(cosPart, 0)
			continue
		}

		p.fftIn[k] = complex((cosPart+sinPart)/2, 0)
		p.fftIn[p.extendedN-k] = complex((cosPart-sinPart)/2, 0)
	}

	if err := p.fftPlan.Forward(p.fftOut, p.fftIn); err != nil {
		return fmt.Errorf("FFT forward: %w", err)
	}

	for i := range n {
		dst[i] = real(p.fftOut[i]) + imag(p.fftOut[i])
	}

	return nil
}

// NormalizationFactor returns the factor by which values are scaled
// after a Forward followed by Inverse transform.
// For DCT-I: Forward * Inverse = 2*(N-1) * I.
func (p *DCTPlan) NormalizationFactor() float64 {
	if p.opts.Normalization == NormOrtho {
		return 1.0
	}

	return float64(p.extendedN)
}

// NormalizationFactor returns the factor by which values are scaled
// after a Forward followed by Inverse transform.
// For DCT-II: Forward followed by Inverse returns the original signal.
func (p *DCT2Plan) NormalizationFactor() float64 {
	return 1.0
}

// Bytes returns the memory used by the plan in bytes.
func (p *DCTPlan) Bytes() int {
	return len(p.fftIn)*16 + len(p.fftOut)*16
}

// Bytes returns the memory used by the plan in bytes.
func (p *DCT2Plan) Bytes() int {
	return len(p.fftIn)*16 + len(p.fftOut)*16 + len(p.phase)*16
}

// DCT1 computes a one-shot DCT-I transform without reusing a plan.
// This is convenient but allocates memory on each call.
// For repeated transforms of the same size, use NewDCTPlan instead.
func DCT1(dst, src []float64) error {
	plan, err := NewDCTPlan(len(src))
	if err != nil {
		return err
	}

	return plan.Forward(dst, src)
}

// DCT2Forward computes a one-shot DCT-II transform without reusing a plan.
func DCT2Forward(dst, src []float64) error {
	plan, err := NewDCT2Plan(len(src))
	if err != nil {
		return err
	}

	return plan.Forward(dst, src)
}

// DCT1Inverse computes a one-shot inverse DCT-I transform.
func DCT1Inverse(dst, src []float64) error {
	plan, err := NewDCTPlan(len(src))
	if err != nil {
		return err
	}

	return plan.Inverse(dst, src)
}

// DCT2Inverse computes a one-shot inverse DCT-II transform.
func DCT2Inverse(dst, src []float64) error {
	plan, err := NewDCT2Plan(len(src))
	if err != nil {
		return err
	}

	return plan.Inverse(dst, src)
}

// DCT1Coefficient returns the DCT-I coefficient for mode k at position n.
// This is the basis function: cos(πnk/(size-1)).
func DCT1Coefficient(n, k, size int) float64 {
	if size <= 1 {
		return 1.0
	}

	return math.Cos(math.Pi * float64(n) * float64(k) / float64(size-1))
}

// DCT2Coefficient returns the DCT-II coefficient for mode k at position n.
// This is the basis function: cos(π(n+1/2)k/size).
func DCT2Coefficient(n, k, size int) float64 {
	if size <= 0 {
		return 0
	}

	return math.Cos(math.Pi * (float64(n) + 0.5) * float64(k) / float64(size))
}
