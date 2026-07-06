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

// benchPlanBuilders maps each transform kind to its constructor, so the two
// benchmark drivers share one list and each constructor error is reported
// (via b.Fatalf) rather than swallowed into a nil plan that panics later.
func benchPlanBuilders() []struct {
	kind    string
	newPlan func(int) (forwardInversePlan, error)
} {
	return []struct {
		kind    string
		newPlan func(int) (forwardInversePlan, error)
	}{
		{"DST1", func(n int) (forwardInversePlan, error) { return NewDSTPlan(n) }},
		{"DST2", func(n int) (forwardInversePlan, error) { return NewDST2Plan(n) }},
		{"DCT1", func(n int) (forwardInversePlan, error) { return NewDCTPlan(n) }},
		{"DCT2", func(n int) (forwardInversePlan, error) { return NewDCT2Plan(n) }},
		{"DST4", func(n int) (forwardInversePlan, error) { return NewDST4Plan(n) }},
		{"DCT4", func(n int) (forwardInversePlan, error) { return NewDCT4Plan(n) }},
	}
}

func BenchmarkTransformForward(b *testing.B) {
	for _, n := range benchSizes {
		for _, pb := range benchPlanBuilders() {
			b.Run(sizeName(pb.kind, n), func(b *testing.B) {
				plan, err := pb.newPlan(n)
				if err != nil {
					b.Fatalf("%s(%d): %v", pb.kind, n, err)
				}
				benchForward(b, plan, n)
			})
		}
	}
}

func BenchmarkTransformInverse(b *testing.B) {
	for _, n := range benchSizes {
		for _, pb := range benchPlanBuilders() {
			b.Run(sizeName(pb.kind, n), func(b *testing.B) {
				plan, err := pb.newPlan(n)
				if err != nil {
					b.Fatalf("%s(%d): %v", pb.kind, n, err)
				}
				benchInverse(b, plan, n)
			})
		}
	}
}
