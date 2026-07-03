package poisson

import "math"

// validSpacing reports whether a grid spacing is a usable positive, finite
// number. NaN and +Inf slip past a bare `h <= 0` guard (NaN compares false to
// everything, +Inf is > 0), so callers must funnel spacing through here.
func validSpacing(h float64) bool {
	return h > 0 && !math.IsInf(h, 1)
}

// validAlpha reports whether a Helmholtz alpha is finite. A NaN or Inf alpha
// would silently poison every eigenvalue denominator, so it is rejected at
// plan creation.
func validAlpha(alpha float64) bool {
	return !math.IsNaN(alpha) && !math.IsInf(alpha, 0)
}
