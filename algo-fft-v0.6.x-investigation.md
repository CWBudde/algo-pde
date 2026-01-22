# algo-fft v0.6.x Investigation Report

## Summary

algo-fft v0.6.x introduces a critical bug for FFT size 40 that makes it incompatible with algo-pde. The bug causes incorrect FFT computations with errors ~1.0 instead of machine precision (~1e-16).

## Issue Details

### Problem
- **Affected versions**: v0.6.0 through v0.6.6
- **Last working version**: v0.5.4
- **Specific failure**: FFT size 40 (factorization: 2³ × 5)
- **Error magnitude**: ~1.0 (complete failure) vs expected ~1e-16 (machine precision)

### Test Results

#### Size 40 FFT Roundtrip Test
```
v0.5.4: max error = 2.775558e-16  ✓
v0.6.6: max error = 1.054861e+00  ✗ FAIL
```

#### Comprehensive Size Testing
All sizes work correctly in v0.6.6 EXCEPT size 40:
- Powers of 2 (16, 32, 64, 128): ✓
- Other composites (24, 30, 36, 48, 56, 60, 72): ✓
- Primes (31, 37, 41, 43): ✓
- **Size 40 (2³×5): ✗ FAIL**

### Impact on algo-pde

The following tests fail with v0.6.x:
1. `TestManufactured2D/Periodic` - uses grid size 48×40
2. `TestManufactured3D/Neumann` - affected by the bug
3. `TestPlan2D_DirichletNeumann` - affected by the bug
4. `TestApplyNeumannRHS2D_NonZero` - affected by the bug

All failures stem from the size-40 FFT bug, as the tests use dimensions that include 40.

## Root Cause Analysis

The v0.6.x release introduced architectural changes:
1. New codelet twiddle factors (`codeletTwiddleForward`, `codeletTwiddleInverse`)
2. Changed forwardCodelet signature from `(dst, src, twiddle, scratch, bitrev)` to `(dst, src, codeletTwiddle, scratch)`
3. Modified scratch buffer management with `getScratch()` and thread-safe pooling

These changes appear to have introduced a regression specifically for size 40.

## Recommendation

**Stay on algo-fft v0.5.4** until the bug is fixed in a future release.

### Why v0.5.4?
- All tests pass ✓
- Stable and proven
- Includes significant SIMD optimizations (140+ commits in v0.5.0 release)
- Compatible with algo-pde's usage patterns

### Future Migration Path
Once algo-fft releases a fix (likely v0.6.7 or v0.7.0):
1. Re-test with the comprehensive size test suite
2. Run full algo-pde test suite
3. Verify all manufactured solution tests pass
4. Update dependency

## Test Programs

Created test programs to reproduce the issue:
- `/tmp/test-fft/test_fft_40.go` - Minimal reproduction for size 40
- `/tmp/test-fft/test_multiple_sizes.go` - Comprehensive size testing

These can be used to verify future fixes.

## References

- algo-fft repository: https://github.com/MeKo-Christian/algo-fft
- v0.6.0 release notes: Introduced "fast-path API" architecture
- Issue to be filed: Size 40 FFT produces incorrect results in v0.6.x

## Date
2026-01-22
