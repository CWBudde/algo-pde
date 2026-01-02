package r2r

import (
	"fmt"
	"math"

	algofft "github.com/MeKo-Christian/algo-fft"
)

// DSTPlan is a pre-computed Discrete Sine Transform plan.
// DST is used for Dirichlet boundary conditions where u = 0 at boundaries.
//
// This implements DST-I (Type I), which is appropriate for the standard
// second-order finite difference Laplacian with Dirichlet BCs.
//
// For input x[0..N-1], the DST-I is defined as:
//
//	X[k] = Σ x[n] * sin(π(n+1)(k+1)/(N+1)) for k = 0..N-1
//
// The inverse is the same transform scaled by 2/(N+1).
//
// Thread safety: A single DSTPlan instance is NOT safe for concurrent use.
// For parallel transforms, create separate plan instances per goroutine.
type DSTPlan struct {
	n    int // Original transform size
	opts Options

	// Extended FFT size: 2*(N+1) for DST-I
	extendedN int

	// Underlying complex FFT plan for the extended size
	fftPlan *algofft.Plan[complex128]

	// Pre-allocated buffers
	// Note: We use separate input and output buffers because the algo-fft
	// library has issues with in-place FFT for certain sizes (e.g., 18).
	fftIn  []complex128 // FFT input buffer
	fftOut []complex128 // FFT output buffer
}

// DST2Plan is a pre-computed Discrete Sine Transform plan (Type II).
//
// For input x[0..N-1], the DST-II is defined as:
//
//	X[k] = Σ x[n] * sin(π(n+1/2)(k+1)/N) for k = 0..N-1
//
// The inverse is DST-II followed by a scale of 2/N.
//
// Thread safety: A single DST2Plan instance is NOT safe for concurrent use.
type DST2Plan struct {
	n    int // Original transform size
	opts Options

	// Extended FFT size: 2*N for DST-II
	extendedN int

	// Underlying complex FFT plan for the extended size
	fftPlan *algofft.Plan[complex128]

	// Pre-allocated buffers
	fftIn  []complex128 // FFT input buffer
	fftOut []complex128 // FFT output buffer
	phase  []complex128 // exp(-i*pi*(k+1)/(2N)) phase factors
}

// NewDSTPlan creates a new DST-I plan for the given size.
// The size n must be at least 1.
func NewDSTPlan(n int, opts ...Option) (*DSTPlan, error) {
	if n < 1 {
		return nil, ErrInvalidSize
	}

	// DST-I via FFT uses odd extension: embed n points into 2*(n+1) points
	// x[0..n-1] -> [0, x[0], x[1], ..., x[n-1], 0, -x[n-1], ..., -x[0]]
	extendedN := 2 * (n + 1)

	fftPlan, err := algofft.NewPlan64(extendedN)
	if err != nil {
		return nil, fmt.Errorf("creating FFT plan: %w", err)
	}

	return &DSTPlan{
		n:         n,
		opts:      applyOptions(opts),
		extendedN: extendedN,
		fftPlan:   fftPlan,
		fftIn:     make([]complex128, extendedN),
		fftOut:    make([]complex128, extendedN),
	}, nil
}

// NewDST2Plan creates a new DST-II plan for the given size.
// The size n must be at least 1.
func NewDST2Plan(n int, opts ...Option) (*DST2Plan, error) {
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
		angle := -math.Pi * float64(k+1) / den
		phase[k] = complex(math.Cos(angle), math.Sin(angle))
	}

	return &DST2Plan{
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
func (p *DSTPlan) Len() int {
	return p.n
}

// Len returns the transform size.
func (p *DST2Plan) Len() int {
	return p.n
}

// Forward computes the forward DST-I transform.
// dst and src must have length n. They may be the same slice for in-place operation.
//
// Output normalization: The output is NOT normalized.
// For orthogonal normalization, divide by sqrt(2*(N+1)).
func (p *DSTPlan) Forward(dst, src []float64) error {
	if len(dst) != p.n || len(src) != p.n {
		return ErrSizeMismatch
	}

	// Build odd-symmetric extension:
	// Position 0: 0
	// Position 1..n: x[0..n-1]
	// Position n+1: 0
	// Position n+2..2n+1: -x[n-1..0]
	for i := range p.extendedN {
		p.fftIn[i] = 0
	}

	for i := range p.n {
		p.fftIn[i+1] = complex(src[i], 0)
		p.fftIn[p.extendedN-1-i] = complex(-src[i], 0)
	}

	// FFT with separate input/output buffers (avoids in-place FFT issues)
	err := p.fftPlan.Forward(p.fftOut, p.fftIn)
	if err != nil {
		return fmt.Errorf("FFT forward: %w", err)
	}

	// Extract DST coefficients from imaginary parts
	// For DST-I with odd extension of size 2*(N+1):
	// X_dst[k] = -Im(Y[k+1]) / 2 for k = 0..n-1
	scale := 1.0
	if p.opts.Normalization == NormOrtho {
		scale = math.Sqrt(2.0 / float64(p.n+1))
	}

	for k := range p.n {
		dst[k] = (-imag(p.fftOut[k+1]) / 2) * scale
	}

	return nil
}

// Forward computes the forward DST-II transform.
// dst and src must have length n. They may be the same slice for in-place operation.
//
// Output normalization: The output is NOT normalized.
// For orthogonal normalization, divide by sqrt(N/2).
func (p *DST2Plan) Forward(dst, src []float64) error {
	if len(dst) != p.n || len(src) != p.n {
		return ErrSizeMismatch
	}

	for i := range p.extendedN {
		p.fftIn[i] = 0
	}

	// Odd extension:
	// x[0..n-1] -> [x[0], ..., x[n-1], -x[n-1], ..., -x[0]]
	for i := range p.n {
		p.fftIn[i] = complex(src[i], 0)
		p.fftIn[p.extendedN-1-i] = complex(-src[i], 0)
	}

	err := p.fftPlan.Forward(p.fftOut, p.fftIn)
	if err != nil {
		return fmt.Errorf("FFT forward: %w", err)
	}

	// DST-II coefficients from phase-shifted FFT output.
	for k := range p.n {
		shifted := p.fftOut[k+1] * p.phase[k]
		value := -imag(shifted) / 2.0

		if p.opts.Normalization == NormOrtho {
			scale := math.Sqrt(2.0 / float64(p.n))
			if k == p.n-1 {
				scale = 1.0 / math.Sqrt(float64(p.n))
			}

			value *= scale
		}

		dst[k] = value
	}

	return nil
}

// Inverse computes the inverse DST-I transform.
// dst and src must have length n. They may be the same slice for in-place operation.
//
// Note: DST-I is its own inverse up to scaling.
// The inverse is: x[n] = (2/(N+1)) * DST-I(X)[n].
func (p *DSTPlan) Inverse(dst, src []float64) error {
	// DST-I is self-inverse (up to normalization)
	if err := p.Forward(dst, src); err != nil {
		return err
	}

	scale := 2.0 / float64(p.n+1)
	if p.opts.Normalization == NormOrtho {
		scale = 1.0
	}

	for i := range p.n {
		dst[i] *= scale
	}

	return nil
}

// Inverse computes the inverse DST-II transform.
// dst and src must have length n. They may be the same slice for in-place operation.
//
// Note: DST-II is inverted by a weighted transpose of the DST-II kernel.
func (p *DST2Plan) Inverse(dst, src []float64) error {
	if len(dst) != p.n || len(src) != p.n {
		return ErrSizeMismatch
	}

	srcData := src
	if len(src) > 0 && len(dst) > 0 && &src[0] == &dst[0] {
		srcData = make([]float64, p.n)
		copy(srcData, src)
	}

	// Weighted transpose of DST-II kernel (O(N^2)); TODO: replace with FFT-based DST-II inverse.
	for n := range p.n {
		sum := 0.0
		for k := range p.n {
			weight := 2.0 / float64(p.n)
			if k == p.n-1 {
				weight = 1.0 / float64(p.n)
			}

			if p.opts.Normalization == NormOrtho {
				weight = math.Sqrt(2.0 / float64(p.n))
				if k == p.n-1 {
					weight = 1.0 / math.Sqrt(float64(p.n))
				}
			}

			sum += (srcData[k] * weight) * DST2Coefficient(n, k, p.n)
		}

		dst[n] = sum
	}

	return nil
}

// NormalizationFactor returns the factor by which values are scaled
// after a Forward followed by Inverse transform.
// For DST-I: Forward * Inverse = (N+1)/2 * I.
func (p *DSTPlan) NormalizationFactor() float64 {
	if p.opts.Normalization == NormOrtho {
		return 1.0
	}

	return float64(p.n+1) / 2.0
}

// NormalizationFactor returns the factor by which values are scaled
// after a Forward followed by Inverse transform.
// For DST-II: Forward followed by Inverse returns the original signal.
func (p *DST2Plan) NormalizationFactor() float64 {
	return 1.0
}

// Bytes returns the memory used by the plan in bytes.
func (p *DSTPlan) Bytes() int {
	return len(p.fftIn)*16 + len(p.fftOut)*16
}

// Bytes returns the memory used by the plan in bytes.
func (p *DST2Plan) Bytes() int {
	return len(p.fftIn)*16 + len(p.fftOut)*16 + len(p.phase)*16
}

// DST1 computes a one-shot DST-I transform without reusing a plan.
// This is convenient but allocates memory on each call.
// For repeated transforms of the same size, use NewDSTPlan instead.
func DST1(dst, src []float64) error {
	plan, err := NewDSTPlan(len(src))
	if err != nil {
		return err
	}

	return plan.Forward(dst, src)
}

// DST2Forward computes a one-shot DST-II transform without reusing a plan.
func DST2Forward(dst, src []float64) error {
	plan, err := NewDST2Plan(len(src))
	if err != nil {
		return err
	}

	return plan.Forward(dst, src)
}

// DST1Inverse computes a one-shot inverse DST-I transform.
func DST1Inverse(dst, src []float64) error {
	plan, err := NewDSTPlan(len(src))
	if err != nil {
		return err
	}

	return plan.Inverse(dst, src)
}

// DST2Inverse computes a one-shot inverse DST-II transform.
func DST2Inverse(dst, src []float64) error {
	plan, err := NewDST2Plan(len(src))
	if err != nil {
		return err
	}

	return plan.Inverse(dst, src)
}

// DST1Coefficient returns the DST-I coefficient for mode k at position n.
// This is the basis function: sin(π(n+1)(k+1)/(size+1)).
func DST1Coefficient(n, k, size int) float64 {
	return math.Sin(math.Pi * float64(n+1) * float64(k+1) / float64(size+1))
}

// DST2Coefficient returns the DST-II coefficient for mode k at position n.
// This is the basis function: sin(π(n+1/2)(k+1)/size).
func DST2Coefficient(n, k, size int) float64 {
	if size <= 0 {
		return 0
	}

	return math.Sin(math.Pi * (float64(n) + 0.5) * float64(k+1) / float64(size))
}

// DST3Coefficient returns the DST-III coefficient for mode k at position n.
// This is the basis function: sin(π(n+1)(k+1/2)/size).
func DST3Coefficient(n, k, size int) float64 {
	if size <= 0 {
		return 0
	}

	return math.Sin(math.Pi * float64(n+1) * (float64(k) + 0.5) / float64(size))
}
