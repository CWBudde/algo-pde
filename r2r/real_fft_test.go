package r2r

import (
	"math"
	"testing"
)

// realFFTSizes deliberately spans the transform lengths that the stale
// fftSoundSizes list (see inverse_fft_test.go) used to exclude, plus a general
// mix of primes, powers of two, and neighbours. The DST-II/DCT-II excluded set
// was n in {20,40,80,100,160,200} (extendedN=2n in {40,80,160,200,320,400});
// the DST-I/DCT-I/DST-IV/DCT-IV excluded sizes fall out of the same 2(n±1) and
// 4n embeddings, so this single set exercises every previously-suspect FFT
// length across the six transforms. algo-fft v0.6.15 computes all of them
// correctly, and the real-input FFT introduced for Phase G.3 must match its
// naive reference at each one.
var realFFTSizes = []int{
	2, 3, 5, 7, 8, 10, 16, 17, 19, 20, 21, 25, 32, 39, 40, 50,
	64, 79, 80, 99, 100, 128, 159, 160, 199, 200, 256,
}

func realFFTInput(n int) []float64 {
	src := make([]float64, n)
	for i := range n {
		src[i] = math.Sin(0.7*float64(i)+0.2) + 0.3*float64(i%5) - 0.4*math.Cos(0.13*float64(i))
	}
	return src
}

func assertClose(t *testing.T, name string, n int, got, want []float64) {
	t.Helper()
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-8*(1+math.Abs(want[i])) {
			t.Errorf("%s n=%d [%d]: got %.12g want %.12g", name, n, i, got[i], want[i])
			return
		}
	}
}

// TestRealFFT_Forward_MatchesReference pins every real-FFT-backed forward against
// its independent O(N^2) naive reference across realFFTSizes and both
// normalizations. This is the decisive Phase G.3 regression gate: it proves the
// switch from a complex FFT to NewPlanReal64 preserved every coefficient at
// sizes the old suite skipped.
func TestRealFFT_Forward_MatchesReference(t *testing.T) {
	for _, norm := range []Normalization{NormNone, NormOrtho} {
		ortho := norm == NormOrtho
		for _, n := range realFFTSizes {
			src := realFFTInput(n)

			// DST-I / DST-II / DCT-II references only exist unnormalized, so
			// exercise those with NormNone; the type-IV references take ortho.
			if !ortho {
				dst1, err := NewDSTPlan(n)
				if err != nil {
					t.Fatalf("NewDSTPlan(%d): %v", n, err)
				}
				got := make([]float64, n)
				if err := dst1.Forward(got, src); err != nil {
					t.Fatalf("DST-I Forward n=%d: %v", n, err)
				}
				want := make([]float64, n)
				dst1Reference(want, src)
				assertClose(t, "DST-I", n, got, want)

				dst2, _ := NewDST2Plan(n)
				got2 := make([]float64, n)
				if err := dst2.Forward(got2, src); err != nil {
					t.Fatalf("DST-II Forward n=%d: %v", n, err)
				}
				want2 := make([]float64, n)
				dst2Reference(want2, src)
				assertClose(t, "DST-II", n, got2, want2)

				if n >= 2 {
					dct1, _ := NewDCTPlan(n)
					gotc := make([]float64, n)
					if err := dct1.Forward(gotc, src); err != nil {
						t.Fatalf("DCT-I Forward n=%d: %v", n, err)
					}
					wantc := make([]float64, n)
					dct1Reference(wantc, src)
					assertClose(t, "DCT-I", n, gotc, wantc)
				}

				dct2, _ := NewDCT2Plan(n)
				gotc2 := make([]float64, n)
				if err := dct2.Forward(gotc2, src); err != nil {
					t.Fatalf("DCT-II Forward n=%d: %v", n, err)
				}
				wantc2 := make([]float64, n)
				dct2Reference(wantc2, src)
				assertClose(t, "DCT-II", n, gotc2, wantc2)
			}

			// DST-IV / DCT-IV references cover both normalizations.
			dst4, _ := NewDST4Plan(n, WithNormalization(norm))
			gots4 := make([]float64, n)
			if err := dst4.Forward(gots4, src); err != nil {
				t.Fatalf("DST-IV Forward n=%d: %v", n, err)
			}
			assertClose(t, "DST-IV", n, gots4, dst4Reference(src, ortho))

			dct4, _ := NewDCT4Plan(n, WithNormalization(norm))
			gotc4 := make([]float64, n)
			if err := dct4.Forward(gotc4, src); err != nil {
				t.Fatalf("DCT-IV Forward n=%d: %v", n, err)
			}
			assertClose(t, "DCT-IV", n, gotc4, dct4Reference(src, ortho))
		}
	}
}

// TestRealFFT_RoundTrip verifies Forward followed by Inverse reconstructs the
// input for every transform at the previously-excluded sizes — the end-to-end
// property the Poisson solver relies on when it forward-transforms an axis and
// inverts it after the spectral divide.
func TestRealFFT_RoundTrip(t *testing.T) {
	for _, n := range realFFTSizes {
		src := realFFTInput(n)

		roundTrip := func(name string, plan forwardInversePlan, minN int) {
			if n < minN {
				return
			}
			fwd := make([]float64, n)
			if err := plan.Forward(fwd, src); err != nil {
				t.Fatalf("%s Forward n=%d: %v", name, n, err)
			}
			back := make([]float64, n)
			if err := plan.Inverse(back, fwd); err != nil {
				t.Fatalf("%s Inverse n=%d: %v", name, n, err)
			}
			assertClose(t, name+" round-trip", n, back, src)
		}

		dst1, _ := NewDSTPlan(n)
		roundTrip("DST-I", dst1, 1)
		dst2, _ := NewDST2Plan(n)
		roundTrip("DST-II", dst2, 1)
		dct2, _ := NewDCT2Plan(n)
		roundTrip("DCT-II", dct2, 1)
		dst4, _ := NewDST4Plan(n)
		roundTrip("DST-IV", dst4, 1)
		dct4, _ := NewDCT4Plan(n)
		roundTrip("DCT-IV", dct4, 1)
		if n >= 2 {
			dct1, _ := NewDCTPlan(n)
			roundTrip("DCT-I", dct1, 2)
		}
	}
}
