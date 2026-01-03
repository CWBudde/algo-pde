package poisson

import "math"

func eigenvaluesDirichlet(n int, h float64) []float64 {
	eig := make([]float64, n)
	h2 := h * h
	for m := 1; m <= n; m++ {
		eig[m-1] = (2.0 - 2.0*math.Cos(math.Pi*float64(m)/float64(n+1))) / h2
	}

	return eig
}

func eigenvaluesNeumann(n int, h float64) []float64 {
	eig := make([]float64, n)
	h2 := h * h
	for m := range n {
		eig[m] = (2.0 - 2.0*math.Cos(math.Pi*float64(m)/float64(n))) / h2
	}

	return eig
}
