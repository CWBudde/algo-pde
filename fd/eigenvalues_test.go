package fd

import (
	"strconv"
	"testing"

	"github.com/CWBudde/algo-pde/bc"
)

func BenchmarkEigenvaluesPeriodic(b *testing.B) {
	sizes := []int{64, 256, 1024, 65536}
	for _, n := range sizes {
		b.Run(sizeStr(n), func(b *testing.B) {
			h := 1.0 / float64(n)
			for range b.N {
				_ = bc.EigenvaluesPeriodic(n, h)
			}
		})
	}
}

// sizeStr formats a grid size for benchmark sub-test names, using a "K" suffix
// for exact multiples of 1024 (e.g. 64 -> "64", 1024 -> "1K", 65536 -> "64K").
func sizeStr(n int) string {
	if n >= 1024 && n%1024 == 0 {
		return strconv.Itoa(n/1024) + "K"
	}

	return strconv.Itoa(n)
}
