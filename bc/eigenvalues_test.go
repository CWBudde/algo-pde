package bc

import (
	"errors"
	"math"
	"testing"
)

const tolerance = 1e-12

func TestEigenvaluesPeriodic(t *testing.T) {
	n := 8
	h := 1.0 / float64(n)
	eig := EigenvaluesPeriodic(n, h)

	if len(eig) != n {
		t.Fatalf("expected %d eigenvalues, got %d", n, len(eig))
	}

	// λ_0 should be 0 (constant mode)
	if math.Abs(eig[0]) > tolerance {
		t.Errorf("λ_0 = %v, expected 0", eig[0])
	}

	// Check a few known values
	// λ_1 = (2 - 2*cos(2π/8)) / h² = (2 - 2*cos(π/4)) / h²
	//     = (2 - 2*√2/2) / h² = (2 - √2) / h²
	h2 := h * h

	expected1 := (2.0 - math.Sqrt(2.0)) / h2
	if math.Abs(eig[1]-expected1) > tolerance {
		t.Errorf("λ_1 = %v, expected %v", eig[1], expected1)
	}

	// λ_4 = (2 - 2*cos(π)) / h² = 4/h² (highest frequency)
	expected4 := 4.0 / h2
	if math.Abs(eig[4]-expected4) > tolerance {
		t.Errorf("λ_4 = %v, expected %v", eig[4], expected4)
	}

	// Eigenvalues should be symmetric: λ_m = λ_{N-m}
	for m := 1; m < n/2; m++ {
		if math.Abs(eig[m]-eig[n-m]) > tolerance {
			t.Errorf("symmetry violated: λ_%d = %v, λ_%d = %v", m, eig[m], n-m, eig[n-m])
		}
	}
}

func TestEigenvaluesDirichlet(t *testing.T) {
	n := 8
	h := 1.0 / float64(n+1) // Note: Dirichlet grid has n interior points
	eig := EigenvaluesDirichlet(n, h)

	if len(eig) != n {
		t.Fatalf("expected %d eigenvalues, got %d", n, len(eig))
	}

	// All eigenvalues should be positive (no nullspace)
	for m, v := range eig {
		if v <= 0 {
			t.Errorf("λ_%d = %v, expected positive", m, v)
		}
	}

	// Check smallest eigenvalue (m=1)
	// λ_1 = (2 - 2*cos(π/9)) / h²
	h2 := h * h

	expected1 := (2.0 - 2.0*math.Cos(math.Pi/float64(n+1))) / h2
	if math.Abs(eig[0]-expected1) > tolerance {
		t.Errorf("λ_1 = %v, expected %v", eig[0], expected1)
	}

	// Eigenvalues should be strictly increasing
	for m := 1; m < n; m++ {
		if eig[m] <= eig[m-1] {
			t.Errorf("eigenvalues not strictly increasing: λ_%d = %v <= λ_%d = %v",
				m, eig[m], m-1, eig[m-1])
		}
	}
}

func TestEigenvaluesNeumann(t *testing.T) {
	n := 8
	h := 1.0 / float64(n)
	eig := EigenvaluesNeumann(n, h)

	if len(eig) != n {
		t.Fatalf("expected %d eigenvalues, got %d", n, len(eig))
	}

	// λ_0 should be 0 (constant mode / nullspace)
	if math.Abs(eig[0]) > tolerance {
		t.Errorf("λ_0 = %v, expected 0", eig[0])
	}

	// All other eigenvalues should be positive
	for m := 1; m < n; m++ {
		if eig[m] <= 0 {
			t.Errorf("λ_%d = %v, expected positive", m, eig[m])
		}
	}

	// Check λ_1 = (2 - 2*cos(π/8)) / h²
	h2 := h * h

	expected1 := (2.0 - 2.0*math.Cos(math.Pi/float64(n))) / h2
	if math.Abs(eig[1]-expected1) > tolerance {
		t.Errorf("λ_1 = %v, expected %v", eig[1], expected1)
	}
}

func TestHasZeroEigenvalue(t *testing.T) {
	tests := []struct {
		bc   BCType
		want bool
	}{
		{Periodic, true},
		{Dirichlet, false},
		{Neumann, true},
	}
	for _, tt := range tests {
		if got := HasZeroEigenvalue(tt.bc); got != tt.want {
			t.Errorf("HasZeroEigenvalue(%v) = %v, want %v", tt.bc, got, tt.want)
		}
	}
}

func TestZeroEigenvalueIndex(t *testing.T) {
	tests := []struct {
		bc   BCType
		want int
	}{
		{Periodic, 0},
		{Dirichlet, -1},
		{Neumann, 0},
	}
	for _, tt := range tests {
		if got := ZeroEigenvalueIndex(tt.bc); got != tt.want {
			t.Errorf("ZeroEigenvalueIndex(%v) = %v, want %v", tt.bc, got, tt.want)
		}
	}
}

func TestHasNullspace(t *testing.T) {
	tests := []struct {
		bc   BCType
		want bool
	}{
		{Periodic, true},
		{Dirichlet, false},
		{Neumann, true},
	}
	for _, tt := range tests {
		if got := tt.bc.HasNullspace(); got != tt.want {
			t.Errorf("%v.HasNullspace() = %v, want %v", tt.bc, got, tt.want)
		}
	}
}

func TestEigenvaluesGeneric(t *testing.T) {
	n := 16
	h := 0.1

	// Test that the generic function matches the specific ones.
	tests := []struct {
		bc       BCType
		specific []float64
	}{
		{Periodic, EigenvaluesPeriodic(n, h)},
		{Dirichlet, EigenvaluesDirichlet(n, h)},
		{Neumann, EigenvaluesNeumann(n, h)},
	}

	for _, tt := range tests {
		generic, err := Eigenvalues(n, h, tt.bc)
		if err != nil {
			t.Fatalf("%v: Eigenvalues failed: %v", tt.bc, err)
		}
		if len(generic) != len(tt.specific) {
			t.Errorf("%v: length mismatch: %d vs %d", tt.bc, len(generic), len(tt.specific))
			continue
		}

		for i := range generic {
			if math.Abs(generic[i]-tt.specific[i]) > tolerance {
				t.Errorf("%v: eigenvalue[%d] mismatch: %v vs %v",
					tt.bc, i, generic[i], tt.specific[i])
			}
		}
	}
}

func TestEigenvaluesInvalidBC(t *testing.T) {
	if _, err := Eigenvalues(4, 1.0, BCType(99)); !errors.Is(err, ErrInvalidBC) {
		t.Fatalf("bogus BC: got %v, want ErrInvalidBC", err)
	}
}

func TestString(t *testing.T) {
	tests := []struct {
		bc   BCType
		want string
	}{
		{Periodic, "Periodic"},
		{Dirichlet, "Dirichlet"},
		{Neumann, "Neumann"},
		{BCType(99), "Unknown"},
	}
	for _, tt := range tests {
		if got := tt.bc.String(); got != tt.want {
			t.Errorf("BCType(%d).String() = %q, want %q", int(tt.bc), got, tt.want)
		}
	}
}
