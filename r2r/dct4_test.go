package r2r

import (
	"math"
	"testing"
)

// fftSoundSizes4 are transform lengths whose 4N-point FFT (used by the type-IV
// quarter-wave transforms) is computed correctly by upstream algo-fft. The
// known-bad FFT sizes ({40,80,160,320,200,400}) all carry a factor of 5, so any
// n divisible by 5 is excluded; the sizes below keep 4N free of that factor.
var fftSoundSizes4 = []int{1, 2, 3, 4, 7, 8, 16, 31, 32, 63, 64, 127, 128, 129, 256}

// dct4Reference evaluates the DCT-IV by direct O(N^2) summation, an independent
// reference for the FFT-based DCT4Plan.Forward.
func dct4Reference(src []float64, ortho bool) []float64 {
	n := len(src)
	out := make([]float64, n)
	scale := 1.0
	if ortho {
		scale = math.Sqrt(2.0 / float64(n))
	}

	for k := range n {
		sum := 0.0
		for i := range n {
			sum += src[i] * DCT4Coefficient(i, k, n)
		}
		out[k] = sum * scale
	}

	return out
}

func TestDCT4Plan_Reference(t *testing.T) {
	for _, norm := range []Normalization{NormNone, NormOrtho} {
		ortho := norm == NormOrtho
		for _, n := range fftSoundSizes4 {
			plan, err := NewDCT4Plan(n, WithNormalization(norm))
			if err != nil {
				t.Fatalf("NewDCT4Plan(%d) failed: %v", n, err)
			}

			src := make([]float64, n)
			for i := range n {
				src[i] = math.Sin(0.4*float64(i)+0.1) + 0.2*float64(i%3)
			}

			want := dct4Reference(src, ortho)

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

// TestDCT4Plan_RoundTrip verifies Inverse∘Forward is the identity (NormNone).
func TestDCT4Plan_RoundTrip(t *testing.T) {
	for _, n := range fftSoundSizes4 {
		plan, err := NewDCT4Plan(n)
		if err != nil {
			t.Fatalf("NewDCT4Plan(%d) failed: %v", n, err)
		}

		src := make([]float64, n)
		for i := range n {
			src[i] = math.Cos(0.7*float64(i)) + 0.3*float64(i%5)
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

// TestDCT4Plan_OrthoSelfInverse verifies the orthonormal DCT-IV is its own
// inverse: Forward∘Forward = identity under NormOrtho.
func TestDCT4Plan_OrthoSelfInverse(t *testing.T) {
	for _, n := range fftSoundSizes4 {
		plan, err := NewDCT4Plan(n, WithNormalization(NormOrtho))
		if err != nil {
			t.Fatalf("NewDCT4Plan(%d) failed: %v", n, err)
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

// TestDCT4Plan_Aliased checks the in-place path Forward(buf, buf) matches the
// non-aliased path.
func TestDCT4Plan_Aliased(t *testing.T) {
	n := 37
	plan, err := NewDCT4Plan(n)
	if err != nil {
		t.Fatalf("NewDCT4Plan failed: %v", err)
	}

	src := make([]float64, n)
	for i := range n {
		src[i] = float64(i+1) * 0.13
	}

	separate := make([]float64, n)
	if err := plan.Forward(separate, src); err != nil {
		t.Fatalf("Forward (separate) failed: %v", err)
	}

	aliased := make([]float64, n)
	copy(aliased, src)
	if err := plan.Forward(aliased, aliased); err != nil {
		t.Fatalf("Forward (aliased) failed: %v", err)
	}

	for i := range n {
		if math.Abs(aliased[i]-separate[i]) > tolerance {
			t.Errorf("[%d]: aliased %v != separate %v", i, aliased[i], separate[i])
		}
	}
}
