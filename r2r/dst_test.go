package r2r

import (
	"math"
	"testing"
)

const tolerance = 1e-10

func TestDSTPlan_RoundTrip(t *testing.T) {
	sizes := []int{3, 4, 7, 8, 15, 16, 31, 32, 63, 64}

	for _, n := range sizes {
		t.Run(sizeStr(n), func(t *testing.T) {
			plan, err := NewDSTPlan(n)
			if err != nil {
				t.Fatalf("NewDSTPlan(%d) failed: %v", n, err)
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

func TestDSTPlan_RoundTripOrtho(t *testing.T) {
	n := 9

	plan, err := NewDSTPlan(n, WithNormalization(NormOrtho))
	if err != nil {
		t.Fatalf("NewDSTPlan failed: %v", err)
	}

	src := make([]float64, n)
	for i := range n {
		src[i] = math.Sin(float64(i) * 0.2)
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

func TestDST2Plan_RoundTrip(t *testing.T) {
	sizes := []int{3, 4, 7, 8, 15, 16, 31, 32, 63, 64}

	for _, n := range sizes {
		t.Run("dst2-"+sizeStr(n), func(t *testing.T) {
			plan, err := NewDST2Plan(n)
			if err != nil {
				t.Fatalf("NewDST2Plan(%d) failed: %v", n, err)
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

func TestDST2Plan_RoundTripOrtho(t *testing.T) {
	n := 8

	plan, err := NewDST2Plan(n, WithNormalization(NormOrtho))
	if err != nil {
		t.Fatalf("NewDST2Plan failed: %v", err)
	}

	src := make([]float64, n)
	for i := range n {
		src[i] = math.Sin(float64(i) * 0.3)
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

func TestDSTPlan_Orthogonality(t *testing.T) {
	// DST-I basis functions should be orthogonal
	n := 7

	// Compute inner product of basis k1 and k2
	for k1 := range n {
		for k2 := range n {
			sum := 0.0
			for i := range n {
				sum += DST1Coefficient(i, k1, n) * DST1Coefficient(i, k2, n)
			}

			expected := 0.0
			if k1 == k2 {
				expected = float64(n+1) / 2.0
			}

			if math.Abs(sum-expected) > tolerance {
				t.Errorf("orthogonality failed for k1=%d, k2=%d: got %v, want %v",
					k1, k2, sum, expected)
			}
		}
	}
}

func TestDST2Plan_Orthogonality(t *testing.T) {
	n := 7

	for k1 := range n {
		for k2 := range n {
			sum := 0.0
			for i := range n {
				sum += DST2Coefficient(i, k1, n) * DST2Coefficient(i, k2, n)
			}

			expected := 0.0
			if k1 == k2 {
				expected = float64(n) / 2.0
				if k1 == n-1 {
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

func TestDSTPlan_KnownValues(t *testing.T) {
	// Test with a single sine mode
	n := 7

	plan, err := NewDSTPlan(n)
	if err != nil {
		t.Fatalf("NewDSTPlan failed: %v", err)
	}

	// Create a pure sine mode (mode k=2)
	k := 2
	src := make([]float64, n)
	for i := range n {
		src[i] = DST1Coefficient(i, k, n)
	}

	dst := make([]float64, n)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// For a pure mode k, the DST should give a spike at position k
	// and (approximately) zero elsewhere
	for j := range n {
		if j == k {
			// The amplitude should be (N+1)/2 for a unit sine input
			expected := float64(n+1) / 2.0
			if math.Abs(dst[j]-expected) > tolerance {
				t.Errorf("dst[%d] = %v, want %v", j, dst[j], expected)
			}
		} else if math.Abs(dst[j]) > tolerance {
			t.Errorf("dst[%d] = %v, want 0", j, dst[j])
		}
	}
}

func TestDST2Plan_KnownValues(t *testing.T) {
	n := 8

	plan, err := NewDST2Plan(n)
	if err != nil {
		t.Fatalf("NewDST2Plan failed: %v", err)
	}

	k := 3
	src := make([]float64, n)
	for i := range n {
		src[i] = DST2Coefficient(i, k, n)
	}

	dst := make([]float64, n)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	for j := range n {
		if j == k {
			expected := float64(n) / 2.0
			if math.Abs(dst[j]-expected) > tolerance {
				t.Errorf("dst[%d] = %v, want %v", j, dst[j], expected)
			}
		} else if math.Abs(dst[j]) > tolerance {
			t.Errorf("dst[%d] = %v, want 0", j, dst[j])
		}
	}
}

func TestDSTPlan_InPlace(t *testing.T) {
	n := 8

	plan, err := NewDSTPlan(n)
	if err != nil {
		t.Fatalf("NewDSTPlan failed: %v", err)
	}

	// Create test input
	src := make([]float64, n)
	expected := make([]float64, n)
	for i := range n {
		src[i] = math.Sin(float64(i) * 0.5)
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

func TestDST2Plan_InPlace(t *testing.T) {
	n := 9

	plan, err := NewDST2Plan(n)
	if err != nil {
		t.Fatalf("NewDST2Plan failed: %v", err)
	}

	src := make([]float64, n)
	expected := make([]float64, n)
	for i := range n {
		src[i] = math.Sin(float64(i) * 0.25)
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

func TestDST1_OneShot(t *testing.T) {
	n := 8

	src := make([]float64, n)
	for i := range n {
		src[i] = float64(i + 1)
	}

	dst := make([]float64, n)
	if err := DST1(dst, src); err != nil {
		t.Fatalf("DST1 failed: %v", err)
	}

	recovered := make([]float64, n)
	if err := DST1Inverse(recovered, dst); err != nil {
		t.Fatalf("DST1Inverse failed: %v", err)
	}

	for i := range n {
		if math.Abs(recovered[i]-src[i]) > tolerance {
			t.Errorf("one-shot round-trip mismatch at [%d]: got %v, want %v",
				i, recovered[i], src[i])
		}
	}
}

func TestDST2_OneShot(t *testing.T) {
	n := 8

	src := make([]float64, n)
	for i := range n {
		src[i] = float64(i) + 0.5
	}

	dst := make([]float64, n)
	if err := DST2Forward(dst, src); err != nil {
		t.Fatalf("DST2Forward failed: %v", err)
	}

	recovered := make([]float64, n)
	if err := DST2Inverse(recovered, dst); err != nil {
		t.Fatalf("DST2Inverse failed: %v", err)
	}

	for i := range n {
		if math.Abs(recovered[i]-src[i]) > tolerance {
			t.Errorf("one-shot round-trip mismatch at [%d]: got %v, want %v",
				i, recovered[i], src[i])
		}
	}
}

func TestDSTPlan_Bytes(t *testing.T) {
	plan, err := NewDSTPlan(8)
	if err != nil {
		t.Fatalf("NewDSTPlan failed: %v", err)
	}

	bytes := plan.Bytes()
	if bytes <= 0 {
		t.Errorf("Bytes() = %d, want > 0", bytes)
	}
}

func TestDST2Plan_Reference(t *testing.T) {
	n := 6
	plan, err := NewDST2Plan(n)
	if err != nil {
		t.Fatalf("NewDST2Plan failed: %v", err)
	}

	src := []float64{0.2, 1.1, -0.3, 0.7, 2.0, -1.5}
	dst := make([]float64, n)
	ref := make([]float64, n)

	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dst2Reference(ref, src)
	for i := range n {
		if math.Abs(dst[i]-ref[i]) > 1e-9 {
			t.Errorf("reference mismatch at [%d]: got %v, want %v", i, dst[i], ref[i])
		}
	}
}

func TestDSTPlan_ConcurrentSeparatePlans(t *testing.T) {
	// Verify that separate plan instances can be used concurrently.
	// Note: A single plan instance is NOT safe for concurrent use.
	const size = 16
	const numGoroutines = 4
	const iterations = 100

	done := make(chan error, numGoroutines)

	for workerID := range numGoroutines {
		go func(worker int) {
			done <- runConcurrentWorker(size, worker, iterations)
		}(workerID)
	}

	// Wait for all goroutines and check for errors
	for range numGoroutines {
		if err := <-done; err != nil {
			t.Errorf("concurrent test error: %v", err)
		}
	}
}

func runConcurrentWorker(size, worker, iterations int) error {
	plan, err := NewDSTPlan(size)
	if err != nil {
		return err
	}

	src := make([]float64, size)
	dst := make([]float64, size)

	for iter := range iterations {
		for i := range size {
			src[i] = float64(worker*1000 + iter*10 + i)
		}

		if err := plan.Forward(dst, src); err != nil {
			return err
		}
		if err := plan.Inverse(dst, dst); err != nil {
			return err
		}

		for i := range size {
			expected := float64(worker*1000 + iter*10 + i)
			if math.Abs(dst[i]-expected) > tolerance {
				return ErrSizeMismatch // Use existing error as a placeholder
			}
		}
	}

	return nil
}

func BenchmarkDSTPlan_Forward(b *testing.B) {
	sizes := []int{64, 256, 1024}

	for _, n := range sizes {
		b.Run(sizeStr(n), func(b *testing.B) {
			plan, err := NewDSTPlan(n)
			if err != nil {
				b.Fatalf("NewDSTPlan failed: %v", err)
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

func dst2Reference(dst, src []float64) {
	n := len(src)
	for k := range n {
		sum := 0.0
		for i := range n {
			sum += src[i] * DST2Coefficient(i, k, n)
		}
		dst[k] = sum
	}
}

func sizeStr(n int) string {
	if n >= 1024 {
		return itoa(n/1024) + "K"
	}

	return itoa(n)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}

	var buf [20]byte

	i := len(buf)

	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}

	return string(buf[i:])
}
