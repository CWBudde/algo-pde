package r2r

import (
	"fmt"
	"math"

	algofft "github.com/cwbudde/algo-fft"
)

// DCT4Plan is a pre-computed Discrete Cosine Transform plan (Type IV).
//
// For input x[0..N-1], the DCT-IV is defined as:
//
//	X[k] = Σ x[n] * cos(π(2n+1)(2k+1)/(4N)) for k = 0..N-1
//
// DCT-IV diagonalises the second-difference operator with a Neumann condition
// at the low boundary and a Dirichlet condition at the high boundary, on a
// cell-centred (quarter-wave) grid. It is a symmetric, orthogonal transform:
// with NormNone, applying it twice yields (N/2)·I, so the inverse is the
// forward scaled by 2/N.
//
// Implementation: a real-input 4N-point FFT of the zero-padded sequence,
// followed by a per-mode phase rotation, reads the type-IV coefficients from
// the odd output bins (2k+1). Placing the samples at the first N positions of a
// 4N buffer, transforming, and rotating by exp(-iπ(2k+1)/(4N)) gives
// X_dct[k] = Re(rotated bin), X_dst[k] = -Im(rotated bin). The FFT is a
// real-to-complex transform (NewPlanReal64) that returns the non-redundant
// half-spectrum at full float64 precision. The 4N length remains redundant
// (only the first N inputs are non-zero); collapsing it to an N-point transform
// via a compact quarter-wave algorithm is a further Phase G.3 opportunity.
//
// Thread safety: A single DCT4Plan instance is NOT safe for concurrent use.
// For parallel transforms, create separate plan instances per goroutine.
type DCT4Plan struct {
	n    int // Original transform size
	opts Options

	// Extended FFT size: 4*N for DCT-IV
	extendedN int

	// Underlying real-input FFT plan for the extended size (Phase G.3).
	fftPlan *algofft.PlanReal[float64, complex128]

	// Pre-allocated buffers: real (zero-padded) input, half-spectrum output.
	fftIn  []float64    // real FFT input buffer, length extendedN
	fftOut []complex128 // half-spectrum output buffer, length extendedN/2+1
	phase  []complex128 // exp(-i*pi*(2k+1)/(4N)) phase factors
}

// NewDCT4Plan creates a new DCT-IV plan for the given size.
// The size n must be at least 1.
func NewDCT4Plan(n int, opts ...Option) (*DCT4Plan, error) {
	if n < 1 {
		return nil, ErrInvalidSize
	}

	extendedN := 4 * n

	fftPlan, err := algofft.NewPlanReal64(extendedN)
	if err != nil {
		return nil, fmt.Errorf("creating FFT plan: %w", err)
	}

	phase := quarterWavePhase(n)

	return &DCT4Plan{
		n:         n,
		opts:      applyOptions(opts),
		extendedN: extendedN,
		fftPlan:   fftPlan,
		fftIn:     make([]float64, extendedN),
		fftOut:    make([]complex128, extendedN/2+1),
		phase:     phase,
	}, nil
}

// quarterWavePhase returns exp(-iπ(2k+1)/(4N)) for k = 0..n-1, the post-FFT
// rotation shared by the DCT-IV and DST-IV embeddings.
func quarterWavePhase(n int) []complex128 {
	phase := make([]complex128, n)
	den := 4.0 * float64(n)
	for k := range n {
		angle := -math.Pi * float64(2*k+1) / den
		phase[k] = complex(math.Cos(angle), math.Sin(angle))
	}

	return phase
}

// Len returns the transform size.
func (p *DCT4Plan) Len() int {
	return p.n
}

// Forward computes the forward DCT-IV transform.
// dst and src must have length n. They may be the same slice for in-place operation.
//
// Output normalization: With NormNone the output is unnormalized. With NormOrtho
// the transform is scaled by sqrt(2/N), making it a true orthonormal (and, being
// symmetric, self-inverse) DCT-IV.
func (p *DCT4Plan) Forward(dst, src []float64) error {
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
		dst[k] = real(y) * scale
	}

	return nil
}

// Inverse computes the inverse DCT-IV transform.
// dst and src must have length n. They may be the same slice for in-place operation.
//
// DCT-IV is symmetric and orthogonal, hence its own inverse up to scaling: the
// inverse is the forward scaled by 2/N (or 1 under NormOrtho).
func (p *DCT4Plan) Inverse(dst, src []float64) error {
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
func (p *DCT4Plan) Bytes() int {
	return len(p.fftIn)*8 + len(p.fftOut)*16 + len(p.phase)*16
}

// DCT4Forward computes a one-shot DCT-IV transform without reusing a plan.
func DCT4Forward(dst, src []float64) error {
	plan, err := NewDCT4Plan(len(src))
	if err != nil {
		return err
	}

	return plan.Forward(dst, src)
}

// DCT4Inverse computes a one-shot inverse DCT-IV transform.
func DCT4Inverse(dst, src []float64) error {
	plan, err := NewDCT4Plan(len(src))
	if err != nil {
		return err
	}

	return plan.Inverse(dst, src)
}

// DCT4Coefficient returns the DCT-IV coefficient for mode k at position n.
// This is the basis function: cos(π(2n+1)(2k+1)/(4*size)).
func DCT4Coefficient(n, k, size int) float64 {
	if size <= 0 {
		return 0
	}

	return math.Cos(math.Pi * float64(2*n+1) * float64(2*k+1) / (4.0 * float64(size)))
}
