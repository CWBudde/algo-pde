package poisson

import "math"

func eigenvaluesPeriodic(n int, h float64) []float64 {
	eig := make([]float64, n)
	h2 := h * h
	for m := range n {
		eig[m] = (2.0 - 2.0*math.Cos(2.0*math.Pi*float64(m)/float64(n))) / h2
	}

	return eig
}
