package r2r

import (
	"math"
	"testing"
)

// naiveDST2Inverse evaluates the weighted transpose of the DST-II kernel by
// direct O(N^2) summation. It mirrors the exact weighting the plan uses and
// serves as an independent reference for the FFT-based DST2Plan.Inverse.
func naiveDST2Inverse(src []float64, ortho bool) []float64 {
	n := len(src)
	out := make([]float64, n)

	for row := range n {
		sum := 0.0
		for k := range n {
			weight := 2.0 / float64(n)
			if k == n-1 {
				weight = 1.0 / float64(n)
			}

			if ortho {
				weight = math.Sqrt(2.0 / float64(n))
				if k == n-1 {
					weight = 1.0 / math.Sqrt(float64(n))
				}
			}

			sum += (src[k] * weight) * DST2Coefficient(row, k, n)
		}

		out[row] = sum
	}

	return out
}

// naiveDCT2Inverse is the DCT-II analogue of naiveDST2Inverse.
func naiveDCT2Inverse(src []float64, ortho bool) []float64 {
	n := len(src)
	out := make([]float64, n)

	for row := range n {
		sum := 0.0
		for k := range n {
			weight := 2.0 / float64(n)
			if k == 0 {
				weight = 1.0 / float64(n)
			}

			if ortho {
				weight = math.Sqrt(2.0 / float64(n))
				if k == 0 {
					weight = 1.0 / math.Sqrt(float64(n))
				}
			}

			sum += (src[k] * weight) * DCT2Coefficient(row, k, n)
		}

		out[row] = sum
	}

	return out
}

// fftSoundSizes are the transform lengths exercised against the naive
// reference. This list used to exclude n in {20,40,80,100,160,200} (2N in
// {40,80,160,200,320,400}) because an old algo-fft mixed-radix kernel returned
// wrong results for factor-5 lengths. That defect is gone: the excluded sizes
// were re-verified against both algo-fft v0.6.15 and v0.8.0 and pass, so they
// are back in the list and the factor-5 lengths are covered again.
var fftSoundSizes = []int{1, 2, 3, 4, 7, 8, 15, 16, 20, 31, 32, 40, 63, 64, 80, 100, 127, 128, 129, 150, 160, 200, 255, 256}

func TestDST2PlanInverse_MatchesNaive(t *testing.T) {
	sizes := fftSoundSizes

	for _, norm := range []Normalization{NormNone, NormOrtho} {
		ortho := norm == NormOrtho
		for _, n := range sizes {
			plan, err := NewDST2Plan(n, WithNormalization(norm))
			if err != nil {
				t.Fatalf("NewDST2Plan(%d) failed: %v", n, err)
			}

			src := make([]float64, n)
			for i := range n {
				src[i] = math.Cos(0.7*float64(i)) + 0.3*float64(i%5)
			}

			want := naiveDST2Inverse(src, ortho)

			got := make([]float64, n)
			if err := plan.Inverse(got, src); err != nil {
				t.Fatalf("Inverse failed (n=%d): %v", n, err)
			}

			for i := range n {
				if math.Abs(got[i]-want[i]) > 1e-9 {
					t.Errorf("n=%d ortho=%v [%d]: got %v want %v", n, ortho, i, got[i], want[i])
				}
			}
		}
	}
}

func TestDCT2PlanInverse_MatchesNaive(t *testing.T) {
	sizes := fftSoundSizes

	for _, norm := range []Normalization{NormNone, NormOrtho} {
		ortho := norm == NormOrtho
		for _, n := range sizes {
			plan, err := NewDCT2Plan(n, WithNormalization(norm))
			if err != nil {
				t.Fatalf("NewDCT2Plan(%d) failed: %v", n, err)
			}

			src := make([]float64, n)
			for i := range n {
				src[i] = math.Sin(0.4*float64(i)+0.1) + 0.2*float64(i%3)
			}

			want := naiveDCT2Inverse(src, ortho)

			got := make([]float64, n)
			if err := plan.Inverse(got, src); err != nil {
				t.Fatalf("Inverse failed (n=%d): %v", n, err)
			}

			for i := range n {
				if math.Abs(got[i]-want[i]) > 1e-9 {
					t.Errorf("n=%d ortho=%v [%d]: got %v want %v", n, ortho, i, got[i], want[i])
				}
			}
		}
	}
}

// TestDST2PlanInverse_Aliased checks that the in-place path Inverse(buf, buf)
// produces the same result as the non-aliased path without any extra
// allocation being required for correctness.
func TestDST2PlanInverse_Aliased(t *testing.T) {
	n := 37
	plan, err := NewDST2Plan(n)
	if err != nil {
		t.Fatalf("NewDST2Plan failed: %v", err)
	}

	src := make([]float64, n)
	for i := range n {
		src[i] = float64(i+1) * 0.11
	}

	separate := make([]float64, n)
	if err := plan.Inverse(separate, src); err != nil {
		t.Fatalf("Inverse (separate) failed: %v", err)
	}

	aliased := make([]float64, n)
	copy(aliased, src)
	if err := plan.Inverse(aliased, aliased); err != nil {
		t.Fatalf("Inverse (aliased) failed: %v", err)
	}

	for i := range n {
		if math.Abs(aliased[i]-separate[i]) > tolerance {
			t.Errorf("[%d]: aliased %v != separate %v", i, aliased[i], separate[i])
		}
	}
}

func TestDCT2PlanInverse_Aliased(t *testing.T) {
	n := 37
	plan, err := NewDCT2Plan(n)
	if err != nil {
		t.Fatalf("NewDCT2Plan failed: %v", err)
	}

	src := make([]float64, n)
	for i := range n {
		src[i] = float64(i+1) * 0.13
	}

	separate := make([]float64, n)
	if err := plan.Inverse(separate, src); err != nil {
		t.Fatalf("Inverse (separate) failed: %v", err)
	}

	aliased := make([]float64, n)
	copy(aliased, src)
	if err := plan.Inverse(aliased, aliased); err != nil {
		t.Fatalf("Inverse (aliased) failed: %v", err)
	}

	for i := range n {
		if math.Abs(aliased[i]-separate[i]) > tolerance {
			t.Errorf("[%d]: aliased %v != separate %v", i, aliased[i], separate[i])
		}
	}
}

// TestInverseAllocFree asserts the inverse transforms no longer allocate in the
// aliased path (the old O(N^2) code allocated a scratch copy on every call).
func TestInverseAllocFree(t *testing.T) {
	n := 256

	dst2, err := NewDST2Plan(n)
	if err != nil {
		t.Fatalf("NewDST2Plan failed: %v", err)
	}

	dct2, err := NewDCT2Plan(n)
	if err != nil {
		t.Fatalf("NewDCT2Plan failed: %v", err)
	}

	buf := make([]float64, n)
	for i := range n {
		buf[i] = float64(i)
	}

	if allocs := testing.AllocsPerRun(20, func() {
		_ = dst2.Inverse(buf, buf)
	}); allocs != 0 {
		t.Errorf("DST2 Inverse allocated %v times per run, want 0", allocs)
	}

	if allocs := testing.AllocsPerRun(20, func() {
		_ = dct2.Inverse(buf, buf)
	}); allocs != 0 {
		t.Errorf("DCT2 Inverse allocated %v times per run, want 0", allocs)
	}
}
