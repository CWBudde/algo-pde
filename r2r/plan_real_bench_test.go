package r2r

import (
	"math"
	"strconv"
	"testing"
)

// sizeName builds a stable sub-benchmark label like "DST2/1024".
func sizeName(kind string, n int) string {
	return kind + "/" + strconv.Itoa(n)
}

// Benchmarks for the r2r forward/inverse transforms across sizes. These are the
// "profile first" evidence for the Phase G.3 real-input-FFT work: run once on
// the complex-FFT baseline, once after the switch to NewPlanReal64, and compare.
//
//	just bench-pkg pkg=r2r
//	go test ./r2r -run '^$' -bench 'Transform' -benchmem

var benchSizes = []int{256, 512, 1024, 2048}

func benchInput(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(0.3*float64(i)) + 0.2*math.Cos(0.11*float64(i))
	}
	return x
}

// forwardInversePlan is the common shape of every r2r plan used here.
type forwardInversePlan interface {
	Forward(dst, src []float64) error
	Inverse(dst, src []float64) error
}

func benchForward(b *testing.B, plan forwardInversePlan, n int) {
	b.Helper()
	src := benchInput(n)
	dst := make([]float64, n)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := plan.Forward(dst, src); err != nil {
			b.Fatalf("Forward: %v", err)
		}
	}
}

func benchInverse(b *testing.B, plan forwardInversePlan, n int) {
	b.Helper()
	buf := benchInput(n)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := plan.Inverse(buf, buf); err != nil {
			b.Fatalf("Inverse: %v", err)
		}
	}
}

func BenchmarkTransformForward(b *testing.B) {
	for _, n := range benchSizes {
		dst1, _ := NewDSTPlan(n)
		dst2, _ := NewDST2Plan(n)
		dct1, _ := NewDCTPlan(n)
		dct2, _ := NewDCT2Plan(n)
		dst4, _ := NewDST4Plan(n)
		dct4, _ := NewDCT4Plan(n)

		b.Run(sizeName("DST1", n), func(b *testing.B) { benchForward(b, dst1, n) })
		b.Run(sizeName("DST2", n), func(b *testing.B) { benchForward(b, dst2, n) })
		b.Run(sizeName("DCT1", n), func(b *testing.B) { benchForward(b, dct1, n) })
		b.Run(sizeName("DCT2", n), func(b *testing.B) { benchForward(b, dct2, n) })
		b.Run(sizeName("DST4", n), func(b *testing.B) { benchForward(b, dst4, n) })
		b.Run(sizeName("DCT4", n), func(b *testing.B) { benchForward(b, dct4, n) })
	}
}

func BenchmarkTransformInverse(b *testing.B) {
	for _, n := range benchSizes {
		dst1, _ := NewDSTPlan(n)
		dst2, _ := NewDST2Plan(n)
		dct1, _ := NewDCTPlan(n)
		dct2, _ := NewDCT2Plan(n)
		dst4, _ := NewDST4Plan(n)
		dct4, _ := NewDCT4Plan(n)

		b.Run(sizeName("DST1", n), func(b *testing.B) { benchInverse(b, dst1, n) })
		b.Run(sizeName("DST2", n), func(b *testing.B) { benchInverse(b, dst2, n) })
		b.Run(sizeName("DCT1", n), func(b *testing.B) { benchInverse(b, dct1, n) })
		b.Run(sizeName("DCT2", n), func(b *testing.B) { benchInverse(b, dct2, n) })
		b.Run(sizeName("DST4", n), func(b *testing.B) { benchInverse(b, dst4, n) })
		b.Run(sizeName("DCT4", n), func(b *testing.B) { benchInverse(b, dct4, n) })
	}
}
