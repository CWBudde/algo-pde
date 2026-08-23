package eigen

import (
	"errors"
	"math"
	"testing"
)

func diagonal(values []float64) *SparseSymmetric {
	b := NewSymmetricBuilder(len(values))
	for i, value := range values {
		b.Add(i, i, value)
	}
	return b.Build()
}

func TestSolveGeneralizedDiagonal(t *testing.T) {
	a := diagonal([]float64{2, 12, 30, 56, 90, 132})
	b := diagonal([]float64{2, 3, 5, 7, 9, 11})
	result, err := Solve(a, b, Options{NumEigenpairs: 3, Tolerance: 1e-11, MaxIterations: 100, Seed: 42, Preconditioner: NewDiagonalPreconditioner(a, 0)})
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{1, 4, 6}
	for i := range want {
		if math.Abs(result.Eigenvalues[i]-want[i]) > 1e-9 {
			t.Fatalf("eigenvalue %d = %.12g, want %.12g", i, result.Eigenvalues[i], want[i])
		}
		if result.Residuals[i] > 1e-11 {
			t.Fatalf("residual %d = %g", i, result.Residuals[i])
		}
	}
	for i := range result.Eigenvectors {
		bxi := make([]float64, b.Dim())
		b.MulVec(bxi, result.Eigenvectors[i])
		for j := range result.Eigenvectors {
			got := dot(result.Eigenvectors[j], bxi)
			wantDot := 0.0
			if i == j {
				wantDot = 1
			}
			if math.Abs(got-wantDot) > 1e-10 {
				t.Fatalf("x[%d]^T B x[%d] = %g", j, i, got)
			}
		}
	}
}

func TestSolveDeterministic(t *testing.T) {
	a := diagonal([]float64{1, 2, 4, 8, 16})
	b := Identity{N: 5}
	options := Options{NumEigenpairs: 2, Tolerance: 1e-12, Seed: 7}
	first, err := Solve(a, b, options)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Solve(a, b, options)
	if err != nil {
		t.Fatal(err)
	}
	for i := range first.Eigenvalues {
		if first.Eigenvalues[i] != second.Eigenvalues[i] {
			t.Fatalf("eigenvalues differ: %v vs %v", first.Eigenvalues, second.Eigenvalues)
		}
		for j := range first.Eigenvectors[i] {
			if first.Eigenvectors[i][j] != second.Eigenvectors[i][j] {
				t.Fatal("eigenvectors are not deterministic")
			}
		}
	}
}

func TestSolveNonConvergenceReturnsBestResult(t *testing.T) {
	a := diagonal([]float64{1, 2, 3, 4, 5, 6})
	result, err := Solve(a, nil, Options{NumEigenpairs: 2, Tolerance: 1e-30, MaxIterations: 1, Seed: 1})
	if !errors.Is(err, ErrNotConverged) {
		t.Fatalf("error = %v, want ErrNotConverged", err)
	}
	if result == nil || result.Iterations != 1 || result.Converged {
		t.Fatalf("unexpected partial result: %#v", result)
	}
}

func TestIC0Preconditioner(t *testing.T) {
	b := NewSymmetricBuilder(3)
	b.Add(0, 0, 4)
	b.Add(0, 1, -1)
	b.Add(1, 1, 4)
	b.Add(1, 2, -1)
	b.Add(2, 2, 3)
	a := b.Build()
	p, err := NewIC0Preconditioner(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	rhs := []float64{2, -1, 3}
	x := make([]float64, 3)
	p.Apply(x, rhs)
	got := make([]float64, 3)
	a.MulVec(got, x)
	for i := range rhs {
		if math.Abs(got[i]-rhs[i]) > 1e-12 {
			t.Fatalf("A M^-1 rhs[%d] = %g, want %g", i, got[i], rhs[i])
		}
	}
}

func BenchmarkSolveSparse(b *testing.B) {
	const n = 200
	builder := NewSymmetricBuilder(n)
	for i := range n {
		builder.Add(i, i, 2)
		if i+1 < n {
			builder.Add(i, i+1, -1)
		}
	}
	a := builder.Build()
	preconditioner := NewDiagonalPreconditioner(a, 0)
	b.ResetTimer()
	for range b.N {
		_, _ = Solve(a, nil, Options{NumEigenpairs: 8, Tolerance: 1e-7, MaxIterations: 200, Seed: 1, Preconditioner: preconditioner})
	}
}
