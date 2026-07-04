package poisson

import (
	"context"
	"fmt"
	"math"
)

// SolveComplex solves the complex Helmholtz equation (alpha - Δ)u = f, where
// the plan's alpha may be complex and the right-hand side f (rhs) is real. The
// complex solution u is written into dst.
//
// The plan must be built with NewComplexHelmholtzPlan (or any constructor — a
// real-alpha plan simply produces a solution whose imaginary part is ~0). When
// imag(alpha) != 0 the imaginary shift damps the operator, so a driven mode
// near resonance yields a finite field rather than ErrResonant; with a purely
// real alpha the usual near-resonance guard (ErrResonant) still applies.
func (p *Plan) SolveComplex(dst []complex128, rhs []float64) error {
	if dst == nil || rhs == nil {
		return ErrNilBuffer
	}

	size := p.size()
	if len(dst) != size || len(rhs) != size {
		return ErrSizeMismatch
	}

	workspace := p.work.get()
	defer p.work.put(workspace)

	return p.solveComplex(dst, rhs, workspace)
}

// solveComplex runs the transform pipeline with a complex per-mode divide. It
// mirrors Plan.solve but reads the full complex output; the real and imaginary
// parts of the workspace are carried independently by every axis transform.
func (p *Plan) solveComplex(dst []complex128, rhs []float64, workspace *Workspace) error {
	hasNullspace := p.hasNullspace()

	offset := 0.0
	if hasNullspace {
		mean, maxAbs := meanAndMaxAbs(rhs)
		if p.opts.Nullspace == NullspaceZeroMode && !meanWithinTolerance(mean, maxAbs, p.meanRelTol()) {
			return ErrNonZeroMean
		}

		if p.opts.Nullspace == NullspaceSubtractMean {
			offset = mean
		}
	}

	for i, v := range rhs {
		workspace.Complex[i] = complex(v-offset, 0)
	}

	shape := p.shape()
	for axis := range p.dim {
		if err := p.tr[axis].Forward(workspace.Complex, shape, axis); err != nil {
			return fmt.Errorf("forward axis %d: %w", axis, err)
		}
	}

	if err := p.applyComplexEigenvalues(workspace.Complex); err != nil {
		return err
	}

	for axis := p.dim - 1; axis >= 0; axis-- {
		if err := p.tr[axis].Inverse(workspace.Complex, shape, axis); err != nil {
			return fmt.Errorf("inverse axis %d: %w", axis, err)
		}
	}

	addMean := 0.0
	if hasNullspace && p.opts.SolutionMean != nil {
		addMean = *p.opts.SolutionMean
	}

	for i := range workspace.Complex {
		dst[i] = workspace.Complex[i] + complex(addMean, 0)
	}

	return nil
}

// applyComplexEigenvalues divides each spectral mode by the complex denominator
// alpha + Σλ. It deliberately parallels applyEigenvalues rather than sharing it:
// the common real Solve must stay a pure-real divide with no complex arithmetic
// in its hot loop.
func (p *Plan) applyComplexEigenvalues(buf []complex128) error {
	_, ny, nz := p.n[0], p.n[1], p.n[2]
	strideYZ := ny * nz
	strideZ := nz
	allowZeroMode := p.hasNullspace()
	// A nonzero imaginary part (complex shift) keeps |denom| bounded away from
	// zero, so the divide never blows up and there is no resonance to guard.
	damped := imag(p.alphaComplex) != 0
	size := p.size()
	workers := clampWorkers(p.opts.Workers, size)

	return parallelFor(workers, size, func(ctx context.Context, _ int, start, end int) error {
		for idx := start; idx < end; idx++ {
			if idx&cancelPollMask == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}

			i := idx / strideYZ
			rem := idx % strideYZ
			j := rem / strideZ
			k := rem % strideZ

			eigSum := p.eig[0][i]
			if p.dim > 1 {
				eigSum += p.eig[1][j]
			}
			if p.dim > 2 {
				eigSum += p.eig[2][k]
			}

			denom := p.alphaComplex + complex(eigSum, 0)

			if !damped {
				// Real alpha through the complex API: a genuine resonance must
				// error rather than amplify, except for the compatible DC mode of
				// a nullspace problem. scale (the term-magnitude sum) is only
				// needed for this guard, so it is skipped in the damped path.
				scale := math.Abs(real(p.alphaComplex)) + math.Abs(p.eig[0][i])
				if p.dim > 1 {
					scale += math.Abs(p.eig[1][j])
				}
				if p.dim > 2 {
					scale += math.Abs(p.eig[2][k])
				}
				if math.Abs(real(denom)) <= resonanceRelTol*scale {
					if allowZeroMode && i == 0 && (p.dim < 2 || j == 0) && (p.dim < 3 || k == 0) {
						buf[idx] = 0
						continue
					}
					return ErrResonant
				}
			}

			buf[idx] /= denom
		}
		return nil
	})
}
