package r2r

import (
	"math"
	"testing"
)

// dst4Reference evaluates the DST-IV by direct O(N^2) summation, an independent
// reference for the FFT-based DST4Plan.Forward.
func dst4Reference(src []float64, ortho bool) []float64 {
	n := len(src)
	out := make([]float64, n)
	scale := 1.0
	if ortho {
		scale = math.Sqrt(2.0 / float64(n))
	}

	for k := range n {
		sum := 0.0
		for i := range n {
			sum += src[i] * DST4Coefficient(i, k, n)
		}
		out[k] = sum * scale
	}

	return out
}

func TestDST4Plan_Reference(t *testing.T) {
	for _, norm := range []Normalization{NormNone, NormOrtho} {
		ortho := norm == NormOrtho
		for _, n := range fftSoundSizes4 {
			plan, err := NewDST4Plan(n, WithNormalization(norm))
			if err != nil {
				t.Fatalf("NewDST4Plan(%d) failed: %v", n, err)
			}

			src := make([]float64, n)
			for i := range n {
				src[i] = math.Cos(0.7*float64(i)) + 0.3*float64(i%5)
			}

			want := dst4Reference(src, ortho)

			got := make([]float64, n)
			if err := plan.Forward(got, src); err != nil {
				t.Fatalf("Forward failed (n=%d): %v", n, err)
			}

			for i := range n {
				if math.Abs(got[i]-want[i]) > 1e-9 {
					t.Errorf("n=%d ortho=%v [%d]: got %v want %v", n, ortho, i, got[i], want[i])
				}
			}
		}
	}
}

// TestDST4Plan_RoundTrip verifies Inverse∘Forward is the identity (NormNone).
func TestDST4Plan_RoundTrip(t *testing.T) {
	for _, n := range fftSoundSizes4 {
		plan, err := NewDST4Plan(n)
		if err != nil {
			t.Fatalf("NewDST4Plan(%d) failed: %v", n, err)
		}

		src := make([]float64, n)
		for i := range n {
			src[i] = math.Sin(0.4*float64(i)+0.1) + 0.2*float64(i%3)
		}

		spectral := make([]float64, n)
		if err := plan.Forward(spectral, src); err != nil {
			t.Fatalf("Forward failed (n=%d): %v", n, err)
		}

		got := make([]float64, n)
		if err := plan.Inverse(got, spectral); err != nil {
			t.Fatalf("Inverse failed (n=%d): %v", n, err)
		}

		for i := range n {
			if math.Abs(got[i]-src[i]) > 1e-9 {
				t.Errorf("n=%d [%d]: round-trip got %v want %v", n, i, got[i], src[i])
			}
		}
	}
}

// TestDST4Plan_OrthoSelfInverse verifies the orthonormal DST-IV is its own
// inverse: Forward∘Forward = identity under NormOrtho.
func TestDST4Plan_OrthoSelfInverse(t *testing.T) {
	for _, n := range fftSoundSizes4 {
		plan, err := NewDST4Plan(n, WithNormalization(NormOrtho))
		if err != nil {
			t.Fatalf("NewDST4Plan(%d) failed: %v", n, err)
		}

		src := make([]float64, n)
		for i := range n {
			src[i] = float64(i+1) * 0.17
		}

		mid := make([]float64, n)
		if err := plan.Forward(mid, src); err != nil {
			t.Fatalf("Forward failed (n=%d): %v", n, err)
		}

		got := make([]float64, n)
		if err := plan.Forward(got, mid); err != nil {
			t.Fatalf("Forward^2 failed (n=%d): %v", n, err)
		}

		for i := range n {
			if math.Abs(got[i]-src[i]) > 1e-9 {
				t.Errorf("n=%d [%d]: ortho self-inverse got %v want %v", n, i, got[i], src[i])
			}
		}
	}
}
