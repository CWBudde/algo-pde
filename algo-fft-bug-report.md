# Bug Report: FFT size 40 produces incorrect results in v0.6.x

## Description

FFT computations for size 40 produce completely incorrect results in algo-fft v0.6.0 through v0.6.6, with errors around 1.0 instead of the expected machine precision (~1e-16). This is a critical regression from v0.5.4 where size 40 works correctly.

## Reproduction

### Minimal Test Case

```go
package main

import (
	"fmt"
	"math"
	algofft "github.com/MeKo-Christian/algo-fft"
)

func main() {
	n := 40

	// Create test data: sine wave
	data := make([]complex128, n)
	for i := range data {
		data[i] = complex(math.Sin(2.0*math.Pi*float64(i)/float64(n)), 0)
	}

	original := make([]complex128, n)
	copy(original, data)

	// Create FFT plan and do roundtrip
	plan, _ := algofft.NewPlan64(n)
	fwd := make([]complex128, n)
	plan.Forward(fwd, data)
	inv := make([]complex128, n)
	plan.Inverse(inv, fwd)

	// Check error
	maxErr := 0.0
	for i := range data {
		err := math.Abs(real(inv[i]) - real(original[i]))
		if err > maxErr {
			maxErr = err
		}
	}

	fmt.Printf("n=%d, max roundtrip error: %e\n", n, maxErr)
}
```

### Expected vs Actual Results

| Version | Max Error | Status |
|---------|-----------|--------|
| v0.5.4  | 2.8e-16   | ✓ Correct (machine precision) |
| v0.6.6  | 1.05      | ✗ **Complete failure** |

### Comprehensive Testing

Tested multiple sizes with v0.6.6:

| Size | Factorization | Error | Status |
|------|---------------|-------|--------|
| 16   | 2⁴            | 1.1e-16 | ✓ |
| 24   | 2³×3          | 2.2e-16 | ✓ |
| 30   | 2×3×5         | 2.8e-16 | ✓ |
| 36   | 2²×3²         | 4.4e-16 | ✓ |
| **40** | **2³×5**    | **1.05** | **✗ FAIL** |
| 48   | 2⁴×3          | 4.4e-16 | ✓ |
| 60   | 2²×3×5        | 5.6e-16 | ✓ |
| 72   | 2³×3²         | 5.0e-16 | ✓ |

**Only size 40 fails**. All other sizes, including other composites with factor 5 (30, 60), work correctly.

## Environment

- **Platform**: Linux amd64
- **Go version**: 1.24.7, 1.25.6
- **algo-fft versions tested**: v0.5.4 (works), v0.6.0 through v0.6.6 (broken)

## Impact

This bug blocks downstream projects from upgrading to v0.6.x. Specifically, it breaks algo-pde test suite when processing grids with dimension 40.

## Root Cause Hypothesis

The v0.6.x release introduced significant architectural changes:
- New codelet twiddle factor handling
- Modified forwardCodelet signature
- New scratch buffer management

The regression appears specific to the size-40 factorization (2³×5), suggesting an issue in the radix-5 or mixed-radix codelet implementation introduced in v0.6.0.

## Workaround

Downgrade to v0.5.4 until fixed:
```bash
go get github.com/MeKo-Christian/algo-fft@v0.5.4
```

## Additional Notes

- v0.5.4 is stable and includes major SIMD optimizations (v0.5.0 had 140+ commits)
- This bug appears to be in the codelet preparation logic, not the basic FFT algorithm
- Other sizes with factor 5 work fine, so it's specific to the 2³×5 decomposition

---

Please let me know if you need any additional information or if I can help test fixes!
