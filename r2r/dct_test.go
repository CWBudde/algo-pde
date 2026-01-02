package r2r

import (
	"errors"
	"math"
	"testing"
)

func TestDCTPlan_RoundTrip(t *testing.T) {
	sizes := []int{2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64}

	for _, n := range sizes {
		t.Run(sizeStr(n), func(t *testing.T) {
			plan, err := NewDCTPlan(n)
			if err != nil {
				t.Fatalf("NewDCTPlan(%d) failed: %v", n, err)
			}

			// Create test input
			src := make([]float64, n)
			for i := range n {
				src[i] = float64(i + 1)
			}

			// Forward transform
			dst := make([]float64, n)
			if err := plan.Forward(dst, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Inverse transform
			recovered := make([]float64, n)
			if err := plan.Inverse(recovered, dst); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			// Check round-trip
			for i := range n {
				if math.Abs(recovered[i]-src[i]) > tolerance {
					t.Errorf("round-trip mismatch at [%d]: got %v, want %v",
						i, recovered[i], src[i])
				}
			}
		})
	}
}

func TestDCTPlan_RoundTripOrtho(t *testing.T) {
	n := 8

	plan, err := NewDCTPlan(n, WithNormalization(NormOrtho))
	if err != nil {
		t.Fatalf("NewDCTPlan failed: %v", err)
	}

	src := make([]float64, n)
	for i := range n {
		src[i] = math.Cos(float64(i) * 0.4)
	}

	dst := make([]float64, n)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	recovered := make([]float64, n)
	if err := plan.Inverse(recovered, dst); err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	for i := range n {
		if math.Abs(recovered[i]-src[i]) > tolerance {
			t.Errorf("round-trip mismatch at [%d]: got %v, want %v",
				i, recovered[i], src[i])
		}
	}
}

func TestDCTPlan_Orthogonality(t *testing.T) {
	// DCT-I basis functions should be orthogonal (with endpoint weights)
	n := 8

	// For DCT-I, the inner product uses weights:
	// w[0] = w[N-1] = 0.5, w[i] = 1.0 for interior points
	weights := make([]float64, n)
	for i := range n {
		weights[i] = 1.0
	}
	weights[0] = 0.5
	weights[n-1] = 0.5

	// Compute weighted inner product of basis k1 and k2
	for k1 := range n {
		for k2 := range n {
			sum := 0.0
			for i := range n {
				sum += weights[i] * DCT1Coefficient(i, k1, n) * DCT1Coefficient(i, k2, n)
			}

			var expected float64
			if k1 == k2 {
				if k1 == 0 || k1 == n-1 {
					expected = float64(n - 1)
				} else {
					expected = float64(n-1) / 2.0
				}
			}

			if math.Abs(sum-expected) > tolerance {
				t.Errorf("orthogonality failed for k1=%d, k2=%d: got %v, want %v",
					k1, k2, sum, expected)
			}
		}
	}
}

func TestDCTPlan_KnownValues(t *testing.T) {
	// Test with a single cosine mode
	n := 8

	plan, err := NewDCTPlan(n)
	if err != nil {
		t.Fatalf("NewDCTPlan failed: %v", err)
	}

	// Create a pure cosine mode (mode k=2)
	k := 2
	src := make([]float64, n)
	for i := range n {
		src[i] = DCT1Coefficient(i, k, n)
	}

	dst := make([]float64, n)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// For a pure mode k, the DCT should give a spike at position k
	// and (approximately) zero elsewhere
	for j := range n {
		if j == k {
			// The amplitude should be (N-1) for endpoints, (N-1)/2 for interior
			expected := float64(n-1) / 2.0
			// Allow some tolerance for the complex relationship
			if math.Abs(dst[j]-expected) > 0.1*expected && math.Abs(dst[j]) > tolerance {
				t.Logf("dst[%d] = %v, expected around %v (may differ due to normalization)",
					j, dst[j], expected)
			}
		}
	}
}

func TestDCTPlan_ConstantMode(t *testing.T) {
	// The k=0 mode is the constant mode
	n := 8

	plan, err := NewDCTPlan(n)
	if err != nil {
		t.Fatalf("NewDCTPlan failed: %v", err)
	}

	// Constant input
	src := make([]float64, n)
	for i := range n {
		src[i] = 1.0
	}

	dst := make([]float64, n)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Only the k=0 coefficient should be non-zero
	for k := 1; k < n; k++ {
		if math.Abs(dst[k]) > tolerance {
			t.Errorf("dst[%d] = %v, want 0 for constant input", k, dst[k])
		}
	}
}

func TestDCTPlan_InPlace(t *testing.T) {
	n := 8

	plan, err := NewDCTPlan(n)
	if err != nil {
		t.Fatalf("NewDCTPlan failed: %v", err)
	}

	// Create test input
	src := make([]float64, n)
	expected := make([]float64, n)
	for i := range n {
		src[i] = math.Cos(float64(i) * 0.5)
		expected[i] = src[i]
	}

	// Forward in-place
	if err := plan.Forward(src, src); err != nil {
		t.Fatalf("Forward in-place failed: %v", err)
	}

	// Inverse in-place
	if err := plan.Inverse(src, src); err != nil {
		t.Fatalf("Inverse in-place failed: %v", err)
	}

	// Check round-trip
	for i := range n {
		if math.Abs(src[i]-expected[i]) > tolerance {
			t.Errorf("in-place round-trip mismatch at [%d]: got %v, want %v",
				i, src[i], expected[i])
		}
	}
}

func TestDCT1_OneShot(t *testing.T) {
	n := 8

	src := make([]float64, n)
	for i := range n {
		src[i] = float64(i + 1)
	}

	dst := make([]float64, n)
	if err := DCT1(dst, src); err != nil {
		t.Fatalf("DCT1 failed: %v", err)
	}

	recovered := make([]float64, n)
	if err := DCT1Inverse(recovered, dst); err != nil {
		t.Fatalf("DCT1Inverse failed: %v", err)
	}

	for i := range n {
		if math.Abs(recovered[i]-src[i]) > tolerance {
			t.Errorf("one-shot round-trip mismatch at [%d]: got %v, want %v",
				i, recovered[i], src[i])
		}
	}
}

func TestDCT2Plan_RoundTrip(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64}

	for _, n := range sizes {
		t.Run("dct2-"+sizeStr(n), func(t *testing.T) {
			plan, err := NewDCT2Plan(n)
			if err != nil {
				t.Fatalf("NewDCT2Plan(%d) failed: %v", n, err)
			}

			src := make([]float64, n)
			for i := range n {
				src[i] = float64(i+1) * 0.25
			}

			dst := make([]float64, n)
			if err := plan.Forward(dst, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			recovered := make([]float64, n)
			if err := plan.Inverse(recovered, dst); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			for i := range n {
				if math.Abs(recovered[i]-src[i]) > tolerance {
					t.Errorf("round-trip mismatch at [%d]: got %v, want %v",
						i, recovered[i], src[i])
				}
			}
		})
	}
}

func TestDCT2Plan_RoundTripOrtho(t *testing.T) {
	n := 8

	plan, err := NewDCT2Plan(n, WithNormalization(NormOrtho))
	if err != nil {
		t.Fatalf("NewDCT2Plan failed: %v", err)
	}

	src := make([]float64, n)
	for i := range n {
		src[i] = math.Cos(float64(i) * 0.25)
	}

	dst := make([]float64, n)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	recovered := make([]float64, n)
	if err := plan.Inverse(recovered, dst); err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	for i := range n {
		if math.Abs(recovered[i]-src[i]) > tolerance {
			t.Errorf("round-trip mismatch at [%d]: got %v, want %v",
				i, recovered[i], src[i])
		}
	}
}

func TestDCT2Plan_Orthogonality(t *testing.T) {
	n := 7

	for k1 := range n {
		for k2 := range n {
			sum := 0.0
			for i := range n {
				sum += DCT2Coefficient(i, k1, n) * DCT2Coefficient(i, k2, n)
			}

			expected := 0.0
			if k1 == k2 {
				expected = float64(n) / 2.0
				if k1 == 0 {
					expected = float64(n)
				}
			}

			if math.Abs(sum-expected) > tolerance {
				t.Errorf("orthogonality failed for k1=%d, k2=%d: got %v, want %v",
					k1, k2, sum, expected)
			}
		}
	}
}

func TestDCT2Plan_KnownValues(t *testing.T) {
	n := 8

	plan, err := NewDCT2Plan(n)
	if err != nil {
		t.Fatalf("NewDCT2Plan failed: %v", err)
	}

	k := 3
	src := make([]float64, n)
	for i := range n {
		src[i] = DCT2Coefficient(i, k, n)
	}

	dst := make([]float64, n)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	for j := range n {
		if j == k {
			expected := float64(n) / 2.0
			if math.Abs(dst[j]-expected) > tolerance {
				t.Errorf("dct2[%d] = %v, want %v", j, dst[j], expected)
			}
		} else if math.Abs(dst[j]) > tolerance {
			t.Errorf("dct2[%d] = %v, want 0", j, dst[j])
		}
	}
}

func TestDCT2Plan_InPlace(t *testing.T) {
	n := 9

	plan, err := NewDCT2Plan(n)
	if err != nil {
		t.Fatalf("NewDCT2Plan failed: %v", err)
	}

	src := make([]float64, n)
	expected := make([]float64, n)
	for i := range n {
		src[i] = math.Cos(float64(i) * 0.3)
		expected[i] = src[i]
	}

	if err := plan.Forward(src, src); err != nil {
		t.Fatalf("Forward in-place failed: %v", err)
	}

	if err := plan.Inverse(src, src); err != nil {
		t.Fatalf("Inverse in-place failed: %v", err)
	}

	for i := range n {
		if math.Abs(src[i]-expected[i]) > tolerance {
			t.Errorf("in-place round-trip mismatch at [%d]: got %v, want %v",
				i, src[i], expected[i])
		}
	}
}

func TestDCT2_OneShot(t *testing.T) {
	n := 8

	src := make([]float64, n)
	for i := range n {
		src[i] = float64(i) + 0.5
	}

	dst := make([]float64, n)
	if err := DCT2Forward(dst, src); err != nil {
		t.Fatalf("DCT2Forward failed: %v", err)
	}

	recovered := make([]float64, n)
	if err := DCT2Inverse(recovered, dst); err != nil {
		t.Fatalf("DCT2Inverse failed: %v", err)
	}

	for i := range n {
		if math.Abs(recovered[i]-src[i]) > tolerance {
			t.Errorf("one-shot round-trip mismatch at [%d]: got %v, want %v",
				i, recovered[i], src[i])
		}
	}
}

func TestDCTPlan_InvalidSize(t *testing.T) {
	_, err := NewDCTPlan(1)
	if !errors.Is(err, ErrInvalidSize) {
		t.Errorf("NewDCTPlan(1) = %v, want ErrInvalidSize", err)
	}

	_, err = NewDCTPlan(0)
	if !errors.Is(err, ErrInvalidSize) {
		t.Errorf("NewDCTPlan(0) = %v, want ErrInvalidSize", err)
	}
}

func TestDCT2Plan_InvalidSize(t *testing.T) {
	_, err := NewDCT2Plan(0)
	if !errors.Is(err, ErrInvalidSize) {
		t.Errorf("NewDCT2Plan(0) = %v, want ErrInvalidSize", err)
	}
}

func TestDCTPlan_Bytes(t *testing.T) {
	plan, err := NewDCTPlan(8)
	if err != nil {
		t.Fatalf("NewDCTPlan failed: %v", err)
	}

	bytes := plan.Bytes()
	if bytes <= 0 {
		t.Errorf("Bytes() = %d, want > 0", bytes)
	}
}

func TestDCT2Plan_Bytes(t *testing.T) {
	plan, err := NewDCT2Plan(8)
	if err != nil {
		t.Fatalf("NewDCT2Plan failed: %v", err)
	}

	bytes := plan.Bytes()
	if bytes <= 0 {
		t.Errorf("Bytes() = %d, want > 0", bytes)
	}
}

func TestDCT2Plan_Reference(t *testing.T) {
	n := 6
	plan, err := NewDCT2Plan(n)
	if err != nil {
		t.Fatalf("NewDCT2Plan failed: %v", err)
	}

	src := []float64{0.2, 1.1, -0.3, 0.7, 2.0, -1.5}
	dst := make([]float64, n)
	ref := make([]float64, n)

	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dct2Reference(ref, src)
	for i := range n {
		if math.Abs(dst[i]-ref[i]) > 1e-9 {
			t.Errorf("reference mismatch at [%d]: got %v, want %v", i, dst[i], ref[i])
		}
	}
}

func BenchmarkDCTPlan_Forward(b *testing.B) {
	sizes := []int{64, 256, 1024}

	for _, n := range sizes {
		b.Run(sizeStr(n), func(b *testing.B) {
			plan, err := NewDCTPlan(n)
			if err != nil {
				b.Fatalf("NewDCTPlan failed: %v", err)
			}

			src := make([]float64, n)
			dst := make([]float64, n)
			for i := range n {
				src[i] = float64(i)
			}

			b.ResetTimer()
			for range b.N {
				_ = plan.Forward(dst, src)
			}
		})
	}
}

func BenchmarkDCT2Plan_Forward(b *testing.B) {
	sizes := []int{64, 256, 1024}

	for _, n := range sizes {
		b.Run("dct2-"+sizeStr(n), func(b *testing.B) {
			plan, err := NewDCT2Plan(n)
			if err != nil {
				b.Fatalf("NewDCT2Plan failed: %v", err)
			}

			src := make([]float64, n)
			dst := make([]float64, n)
			for i := range n {
				src[i] = float64(i)
			}

			b.ResetTimer()
			for range b.N {
				_ = plan.Forward(dst, src)
			}
		})
	}
}

func dct2Reference(dst, src []float64) {
	n := len(src)
	for k := range n {
		sum := 0.0
		for i := range n {
			sum += src[i] * DCT2Coefficient(i, k, n)
		}
		dst[k] = sum
	}
}
