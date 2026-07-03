package fd

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/bc"
)

// TestEigenvaluesMatchStencil is an independent, non-circular verification that
// Eigenvalues returns the true eigenvalues of the discrete operator that
// Apply1D implements. It does NOT restate the closed-form (2-2cos)/h² formula.
// Instead, for every boundary condition and every mode k it constructs the
// analytic eigenvector sampled at that BC's node positions, applies the actual
// finite-difference stencil, and asserts the result equals λ_k · v_k
// elementwise. If any eigenvalue in Eigenvalues were wrong, the eigenpair
// identity A·v_k = λ_k·v_k would fail for that mode.
func TestEigenvaluesMatchStencil(t *testing.T) {
	const n = 12
	const h = 0.1

	cases := []struct {
		name string
		bc   bc.BCType
		// vec returns the k-th eigenvector's value at node i, sampled at the
		// node positions the BC uses (periodic: i·h, Dirichlet: (i+1)·h,
		// Neumann: (i+½)·h). These are the analytic eigenvectors of the
		// respective discrete operators, independent of the λ formula.
		vec func(i, k int) float64
	}{
		{
			name: "Periodic",
			bc:   bc.Periodic,
			vec: func(i, k int) float64 {
				return math.Cos(2.0 * math.Pi * float64(k) * float64(i) / float64(n))
			},
		},
		{
			name: "Dirichlet",
			bc:   bc.Dirichlet,
			vec: func(i, k int) float64 {
				return math.Sin(math.Pi * float64(k+1) * float64(i+1) / float64(n+1))
			},
		},
		{
			name: "Neumann",
			bc:   bc.Neumann,
			vec: func(i, k int) float64 {
				return math.Cos(math.Pi * float64(k) * (float64(i) + 0.5) / float64(n))
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			eig, err := bc.Eigenvalues(n, h, tc.bc)
			if err != nil {
				t.Fatalf("Eigenvalues failed: %v", err)
			}
			if len(eig) != n {
				t.Fatalf("expected %d eigenvalues, got %d", n, len(eig))
			}

			for k := range n {
				v := make([]float64, n)
				for i := range n {
					v[i] = tc.vec(i, k)
				}

				got := make([]float64, n)
				if err := Apply1D(got, v, h, tc.bc); err != nil {
					t.Fatalf("Apply1D failed: %v", err)
				}

				scale := 0.0
				for i := range n {
					if a := math.Abs(eig[k] * v[i]); a > scale {
						scale = a
					}
				}

				tol := 1e-9 * scale
				if scale == 0 {
					// λ = 0 nullspace mode: the operator must annihilate v.
					tol = 1e-9
				}

				for i := range n {
					want := eig[k] * v[i]
					if math.Abs(got[i]-want) > tol {
						t.Fatalf("%s mode %d node %d: Apply1D=%g want λ·v=%g (λ=%g, tol=%g)",
							tc.name, k, i, got[i], want, eig[k], tol)
					}
				}
			}
		})
	}
}
