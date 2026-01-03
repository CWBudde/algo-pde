package poisson_test

import (
	"errors"
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

const periodicNDTol = 1e-9

func TestNewPlanNDPeriodic_InvalidInputs(t *testing.T) {
	if _, err := poisson.NewPlanNDPeriodic(nil, nil); !errors.Is(err, poisson.ErrInvalidSize) {
		t.Fatalf("expected ErrInvalidSize, got %v", err)
	}

	if _, err := poisson.NewPlanNDPeriodic(poisson.Shape{4, 0}, []float64{1.0, 1.0}); !errors.Is(err, poisson.ErrInvalidSize) {
		t.Fatalf("expected ErrInvalidSize, got %v", err)
	}

	if _, err := poisson.NewPlanNDPeriodic(poisson.Shape{4, 4}, []float64{1.0}); err == nil {
		t.Fatalf("expected error for mismatched h length")
	}

	if _, err := poisson.NewPlanNDPeriodic(poisson.Shape{4, 4}, []float64{1.0, 0.0}); !errors.Is(err, poisson.ErrInvalidSpacing) {
		t.Fatalf("expected ErrInvalidSpacing, got %v", err)
	}
}

func TestPlanNDPeriodic_Solve_Manufactured(t *testing.T) {
	dims := poisson.Shape{4, 5, 6, 7}
	h := make([]float64, len(dims))
	for i, n := range dims {
		h[i] = 1.0 / float64(n)
	}

	plan, err := poisson.NewPlanNDPeriodic(dims, h)
	if err != nil {
		t.Fatalf("NewPlanNDPeriodic failed: %v", err)
	}

	u, rhs := manufacturedND(dims, h)

	got := make([]float64, dims.Size())
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > periodicNDTol {
		t.Fatalf("max error %g exceeds tol %g", max, periodicNDTol)
	}
}

func TestPlanNDPeriodic_SolveInPlace(t *testing.T) {
	dims := poisson.Shape{4, 4, 4, 4}
	h := make([]float64, len(dims))
	for i, n := range dims {
		h[i] = 1.0 / float64(n)
	}

	plan, err := poisson.NewPlanNDPeriodic(dims, h)
	if err != nil {
		t.Fatalf("NewPlanNDPeriodic failed: %v", err)
	}

	u, rhs := manufacturedND(dims, h)
	buf := append([]float64(nil), rhs...)

	if err := plan.SolveInPlace(buf); err != nil {
		t.Fatalf("SolveInPlace failed: %v", err)
	}

	if max := maxAbsDiff(buf, u); max > periodicNDTol {
		t.Fatalf("max error %g exceeds tol %g", max, periodicNDTol)
	}
}

func manufacturedND(dims poisson.Shape, h []float64) ([]float64, []float64) {
	L := make([]float64, len(dims))
	for i, n := range dims {
		L[i] = float64(n) * h[i]
	}

	size := dims.Size()
	u := make([]float64, size)
	rhs := make([]float64, size)
	indices := make([]int, len(dims))

	for idx := range u {
		val := 1.0
		for d := range dims {
			x := float64(indices[d]) * h[d]
			val *= math.Sin(2.0 * math.Pi * x / L[d])
		}

		u[idx] = val

		for d := len(indices) - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < dims[d] {
				break
			}
			indices[d] = 0
		}
	}

	applyPeriodicND(rhs, u, dims, h)

	return u, rhs
}

func applyPeriodicND(dst, src []float64, dims poisson.Shape, h []float64) {
	nDims := len(dims)
	if len(h) != nDims {
		return
	}

	total := dims.Size()
	if len(dst) != total || len(src) != total {
		return
	}

	strides := make([]int, nDims)
	stride := 1
	for d := nDims - 1; d >= 0; d-- {
		strides[d] = stride
		stride *= dims[d]
	}

	indices := make([]int, nDims)
	for idx := range src {
		u := src[idx]
		sum := 0.0

		for d := 0; d < nDims; d++ {
			left := indices[d] - 1
			if left < 0 {
				left = dims[d] - 1
			}

			right := indices[d] + 1
			if right >= dims[d] {
				right = 0
			}

			leftOffset := idx + (left-indices[d])*strides[d]
			rightOffset := idx + (right-indices[d])*strides[d]

			sum += (2.0*u - src[leftOffset] - src[rightOffset]) / (h[d] * h[d])
		}

		dst[idx] = sum

		for d := nDims - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < dims[d] {
				break
			}
			indices[d] = 0
		}
	}
}
