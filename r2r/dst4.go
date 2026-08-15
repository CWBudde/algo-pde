package r2r

import (
	"fmt"
	"math"

	algofft "github.com/cwbudde/algo-fft"
)

// DST4Plan is a pre-computed Discrete Sine Transform plan (Type IV).
//
// For input x[0..N-1], the DST-IV is defined as:
//
//	X[k] = Σ x[n] * sin(π(2n+1)(2k+1)/(4N)) for k = 0..N-1
//
// DST-IV diagonalises the second-difference operator with a Dirichlet condition
// at the low boundary and a Neumann condition at the high boundary, on a
// cell-centred (quarter-wave) grid. It is a symmetric, orthogonal transform:
// with NormNone, applying it twice yields (N/2)·I, so the inverse is the
// forward scaled by 2/N.
//
// Implementation: it shares the DCT-IV embedding — a real-input 4N-point FFT of
// the zero-padded sequence, rotated per mode by exp(-iπ(2k+1)/(4N)) — and reads
// X_dst[k] = -Im(rotated bin 2k+1). See DCT4Plan for the derivation and the note
// on the redundant 4N length (Phase G.3).
//
// Thread safety: A single DST4Plan instance is NOT safe for concurrent use.
// For parallel transforms, create separate plan instances per goroutine.
type DST4Plan struct {
	n    int // Original transform size
	opts Options

	// Extended FFT size: 4*N for DST-IV
	extendedN int

	// Underlying real-input FFT plan for the extended size (Phase G.3).
	fftPlan *algofft.PlanReal[float64, complex128]

	// Pre-allocated buffers: real (zero-padded) input, half-spectrum output.
	fftIn  []float64    // real FFT input buffer, length extendedN
	fftOut []complex128 // half-spectrum output buffer, length extendedN/2+1
	phase  []complex128 // exp(-i*pi*(2k+1)/(4N)) phase factors
}

// NewDST4Plan creates a new DST-IV plan for the given size.
// The size n must be at least 1.
func NewDST4Plan(n int, opts ...Option) (*DST4Plan, error) {
	if n < 1 {
		return nil, ErrInvalidSize
	}

	extendedN := 4 * n

	fftPlan, err := algofft.NewPlanReal64(extendedN)
	if err != nil {
		return nil, fmt.Errorf("creating FFT plan: %w", err)
	}

	return &DST4Plan{
		n:         n,
		opts:      applyOptions(opts),
		extendedN: extendedN,
		fftPlan:   fftPlan,
		fftIn:     make([]float64, extendedN),
		fftOut:    make([]complex128, extendedN/2+1),
		phase:     quarterWavePhase(n),
	}, nil
}

// Len returns the transform size.
func (p *DST4Plan) Len() int {
	return p.n
}

// Forward computes the forward DST-IV transform.
// dst and src must have length n. They may be the same slice for in-place operation.
//
// Output normalization: With NormNone the output is unnormalized. With NormOrtho
// the transform is scaled by sqrt(2/N), making it a true orthonormal (and, being
// symmetric, self-inverse) DST-IV.
func (p *DST4Plan) Forward(dst, src []float64) error {
	if len(dst) != p.n || len(src) != p.n {
		return ErrSizeMismatch
	}

	// src is fully consumed into the FFT buffer before dst is written, so
	// aliasing (Forward(buf, buf)) needs no scratch copy.
	for i := range p.extendedN {
		p.fftIn[i] = 0
	}

	for i := range p.n {
		p.fftIn[i] = src[i]
	}

	if err := p.fftPlan.Forward(p.fftOut, p.fftIn); err != nil {
		return fmt.Errorf("FFT forward: %w", err)
	}

	scale := 1.0
	if p.opts.Normalization == NormOrtho {
		scale = math.Sqrt(2.0 / float64(p.n))
	}

	for k := range p.n {
		y := p.fftOut[2*k+1] * p.phase[k]
		dst[k] = -imag(y) * scale
	}

	return nil
}

// Inverse computes the inverse DST-IV transform.
// dst and src must have length n. They may be the same slice for in-place operation.
//
// DST-IV is symmetric and orthogonal, hence its own inverse up to scaling: the
// inverse is the forward scaled by 2/N (or 1 under NormOrtho).
func (p *DST4Plan) Inverse(dst, src []float64) error {
	if err := p.Forward(dst, src); err != nil {
		return err
	}

	scale := 2.0 / float64(p.n)
	if p.opts.Normalization == NormOrtho {
		scale = 1.0
	}

	for i := range p.n {
		dst[i] *= scale
	}

	return nil
}

// Bytes returns the memory used by the plan in bytes.
func (p *DST4Plan) Bytes() int {
	return len(p.fftIn)*8 + len(p.fftOut)*16 + len(p.phase)*16
}

// DST4Forward computes a one-shot DST-IV transform without reusing a plan.
func DST4Forward(dst, src []float64) error {
	plan, err := NewDST4Plan(len(src))
	if err != nil {
		return err
	}

	return plan.Forward(dst, src)
}

// DST4Inverse computes a one-shot inverse DST-IV transform.
func DST4Inverse(dst, src []float64) error {
	plan, err := NewDST4Plan(len(src))
	if err != nil {
		return err
	}

	return plan.Inverse(dst, src)
}

// DST4Coefficient returns the DST-IV coefficient for mode k at position n.
// This is the basis function: sin(π(2n+1)(2k+1)/(4*size)).
func DST4Coefficient(n, k, size int) float64 {
	if size <= 0 {
		return 0
	}

	return math.Sin(math.Pi * float64(2*n+1) * float64(2*k+1) / (4.0 * float64(size)))
}
