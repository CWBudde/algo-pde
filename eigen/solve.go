package eigen

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

var ErrNotConverged = errors.New("eigen: iteration did not converge")

type Options struct {
	NumEigenpairs  int
	Tolerance      float64
	MaxIterations  int
	Seed           uint64
	Preconditioner Preconditioner
}

type Result struct {
	Eigenvalues  []float64
	Eigenvectors [][]float64
	Residuals    []float64
	Iterations   int
	Converged    bool
}

// Solve returns the lowest generalized symmetric eigenpairs. Eigenvectors are
// B-orthonormal and have a deterministic sign (largest component positive).
// A non-converged Result is returned together with ErrNotConverged.
func Solve(a, b Operator, options Options) (*Result, error) {
	if a == nil {
		return nil, errors.New("eigen: nil stiffness operator")
	}
	n := a.Dim()
	if b == nil {
		b = Identity{N: n}
	}
	if n <= 0 || b.Dim() != n {
		return nil, errors.New("eigen: operator dimensions differ or are empty")
	}
	k := options.NumEigenpairs
	if k <= 0 || k >= n {
		return nil, fmt.Errorf("eigen: NumEigenpairs must be in [1,%d)", n)
	}
	tol := options.Tolerance
	if tol == 0 {
		tol = 1e-8
	}
	if !(tol > 0) || math.IsNaN(tol) || math.IsInf(tol, 0) {
		return nil, errors.New("eigen: tolerance must be finite and positive")
	}
	maxIter := options.MaxIterations
	if maxIter == 0 {
		maxIter = 300
	}
	if maxIter < 1 {
		return nil, errors.New("eigen: MaxIterations must be positive")
	}

	rng := rand.New(rand.NewPCG(options.Seed, options.Seed^0x9e3779b97f4a7c15))
	x := make([][]float64, 0, k)
	for len(x) < k {
		v := make([]float64, n)
		for i := range v {
			v[i] = rng.Float64()*2 - 1
		}
		if appendMOrthonormal(&x, v, b, 1e-12) {
			continue
		}
		return nil, errors.New("eigen: mass operator is not positive definite")
	}

	var previous [][]float64
	result := &Result{}
	for iter := 1; iter <= maxIter; iter++ {
		values, vectors := rayleighRitz(a, b, x, k)
		x = vectors
		residualVectors, residuals, allConverged := residualsFor(a, b, x, values, tol)
		result = &Result{Eigenvalues: values, Eigenvectors: cloneVectors(x), Residuals: residuals, Iterations: iter, Converged: allConverged}
		if allConverged {
			canonicalSigns(result.Eigenvectors)
			return result, nil
		}

		basis := cloneVectors(x)
		for _, r := range residualVectors {
			w := append([]float64(nil), r...)
			if options.Preconditioner != nil {
				options.Preconditioner.Apply(w, r)
			}
			appendMOrthonormal(&basis, w, b, 1e-11)
		}
		for _, p := range previous {
			appendMOrthonormal(&basis, append([]float64(nil), p...), b, 1e-11)
		}
		if len(basis) == k {
			canonicalSigns(result.Eigenvectors)
			return result, fmt.Errorf("%w: search space stagnated", ErrNotConverged)
		}
		previous = make([][]float64, 0, len(basis)-k)
		for _, v := range basis[k:] {
			previous = append(previous, append([]float64(nil), v...))
		}
		x = basis
	}
	canonicalSigns(result.Eigenvectors)
	return result, ErrNotConverged
}

func rayleighRitz(a, b Operator, basis [][]float64, count int) ([]float64, [][]float64) {
	// Re-orthonormalize because accumulated floating-point error otherwise makes
	// the projected generalized problem drift away from the identity mass.
	q := make([][]float64, 0, len(basis))
	for _, v := range basis {
		appendMOrthonormal(&q, append([]float64(nil), v...), b, 1e-13)
	}
	m := len(q)
	h := make([][]float64, m)
	av := make([][]float64, m)
	for j := range q {
		av[j] = make([]float64, a.Dim())
		a.MulVec(av[j], q[j])
		h[j] = make([]float64, m)
	}
	for i := 0; i < m; i++ {
		for j := i; j < m; j++ {
			h[i][j] = dot(q[i], av[j])
			h[j][i] = h[i][j]
		}
	}
	values, coefficients := jacobiSymmetric(h)
	order := make([]int, m)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return values[order[i]] < values[order[j]] })
	outValues := make([]float64, count)
	out := make([][]float64, count)
	for p := 0; p < count; p++ {
		idx := order[p]
		outValues[p] = values[idx]
		out[p] = make([]float64, a.Dim())
		for j := range q {
			axpy(out[p], coefficients[j][idx], q[j])
		}
	}
	return outValues, out
}

func residualsFor(a, b Operator, x [][]float64, values []float64, tol float64) ([][]float64, []float64, bool) {
	residualVectors := make([][]float64, len(x))
	residuals := make([]float64, len(x))
	all := true
	for j := range x {
		av, bv := make([]float64, a.Dim()), make([]float64, a.Dim())
		a.MulVec(av, x[j])
		b.MulVec(bv, x[j])
		r := make([]float64, a.Dim())
		for i := range r {
			r[i] = av[i] - values[j]*bv[i]
		}
		denom := norm(av) + math.Abs(values[j])*norm(bv)
		if denom < 1 {
			denom = 1
		}
		residuals[j] = norm(r) / denom
		if residuals[j] > tol {
			all = false
		}
		residualVectors[j] = r
	}
	return residualVectors, residuals, all
}

func appendMOrthonormal(q *[][]float64, v []float64, b Operator, threshold float64) bool {
	bv := make([]float64, len(v))
	for pass := 0; pass < 2; pass++ {
		b.MulVec(bv, v)
		for _, u := range *q {
			alpha := dot(u, bv)
			axpy(v, -alpha, u)
		}
	}
	b.MulVec(bv, v)
	norm2 := dot(v, bv)
	if !(norm2 > threshold*threshold) || math.IsNaN(norm2) {
		return false
	}
	scale(v, 1/math.Sqrt(norm2))
	*q = append(*q, v)
	return true
}

func jacobiSymmetric(a [][]float64) ([]float64, [][]float64) {
	n := len(a)
	v := make([][]float64, n)
	for i := range v {
		v[i] = make([]float64, n)
		v[i][i] = 1
	}
	limit := 100 * n * n
	for sweep := 0; sweep < limit; sweep++ {
		p, q, largest := 0, 0, 0.0
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if math.Abs(a[i][j]) > largest {
					p, q, largest = i, j, math.Abs(a[i][j])
				}
			}
		}
		if largest <= 1e-14*(1+math.Abs(a[p][p])+math.Abs(a[q][q])) {
			break
		}
		tau := (a[q][q] - a[p][p]) / (2 * a[p][q])
		t := 1 / (math.Abs(tau) + math.Sqrt(1+tau*tau))
		if tau < 0 {
			t = -t
		}
		c := 1 / math.Sqrt(1+t*t)
		s := t * c
		app, aqq, apq := a[p][p], a[q][q], a[p][q]
		a[p][p] = app - t*apq
		a[q][q] = aqq + t*apq
		a[p][q], a[q][p] = 0, 0
		for i := 0; i < n; i++ {
			if i == p || i == q {
				continue
			}
			aip, aiq := a[i][p], a[i][q]
			a[i][p], a[p][i] = c*aip-s*aiq, c*aip-s*aiq
			a[i][q], a[q][i] = s*aip+c*aiq, s*aip+c*aiq
		}
		for i := 0; i < n; i++ {
			vip, viq := v[i][p], v[i][q]
			v[i][p] = c*vip - s*viq
			v[i][q] = s*vip + c*viq
		}
	}
	values := make([]float64, n)
	for i := range values {
		values[i] = a[i][i]
	}
	return values, v
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func norm(a []float64) float64 { return math.Sqrt(dot(a, a)) }
func axpy(dst []float64, alpha float64, src []float64) {
	for i := range dst {
		dst[i] += alpha * src[i]
	}
}

func scale(v []float64, alpha float64) {
	for i := range v {
		v[i] *= alpha
	}
}

func cloneVectors(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = append([]float64(nil), x[i]...)
	}
	return out
}

func canonicalSigns(x [][]float64) {
	for _, v := range x {
		index := 0
		for i := 1; i < len(v); i++ {
			if math.Abs(v[i]) > math.Abs(v[index]) {
				index = i
			}
		}
		if v[index] < 0 {
			scale(v, -1)
		}
	}
}
