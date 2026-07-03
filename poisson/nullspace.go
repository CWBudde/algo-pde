package poisson

import "math"

const (
	// meanRoundoffFloor is the smallest relative tolerance the zero-mean
	// consistency gate ever uses. It absorbs floating-point noise in the
	// (compensated) summation so the gate never rejects a truly zero-mean RHS.
	meanRoundoffFloor = 1e-12

	// meanDiscretizationConst scales the O(h^2) allowance of the gate. The
	// discrete mean of a compatible inhomogeneous Neumann problem is the RHS
	// integral evaluated by a second-order quadrature, so it is zero only up to
	// O((h/L)^2) = O(1/minN^2) truncation error. The gate must accept that or
	// the analytically compatible default path is unusable (previously every
	// test had to route around it via WithSubtractMean). The constant gives a
	// generous multiple of the expected truncation while still separating it by
	// many orders of magnitude from a genuinely inconsistent O(1) mean.
	meanDiscretizationConst = 8.0
)

// meanRelTol returns the relative tolerance for the zero-mean consistency gate
// on a grid whose smallest extent is minExtent. It scales as 1/minExtent^2 so
// the gate tightens as the grid is refined (matching the O(h^2) shrink of the
// quadrature error) yet never drops below the roundoff floor.
func meanRelTol(minExtent int) float64 {
	if minExtent < 1 {
		minExtent = 1
	}

	disc := meanDiscretizationConst / float64(minExtent*minExtent)
	return math.Max(disc, meanRoundoffFloor)
}

// meanWithinTolerance reports whether |mean| is small enough (relative to the
// RHS magnitude and the discretization-scaled relTol) for the RHS to count as
// mean-zero for a nullspace-bearing problem.
func meanWithinTolerance(mean, maxAbs, relTol float64) bool {
	return math.Abs(mean) <= relTol*(1.0+maxAbs)
}

// meanAndMaxAbs returns the arithmetic mean and the maximum absolute value of
// values. The sum uses Kahan/Neumaier compensated summation so the mean is
// stable even across the ~10^8 elements of a 512^3 grid, where naive summation
// would accumulate enough rounding error to trip the zero-mean gate on its own.
func meanAndMaxAbs(values []float64) (mean, maxAbs float64) {
	if len(values) == 0 {
		return 0, 0
	}

	sum := 0.0
	comp := 0.0 // running compensation for lost low-order bits
	for _, v := range values {
		t := sum + v
		if math.Abs(sum) >= math.Abs(v) {
			comp += (sum - t) + v
		} else {
			comp += (v - t) + sum
		}
		sum = t

		if abs := math.Abs(v); abs > maxAbs {
			maxAbs = abs
		}
	}

	return (sum + comp) / float64(len(values)), maxAbs
}
