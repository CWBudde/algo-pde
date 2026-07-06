package poisson

// divByReal divides z by the real scalar d. It exists to avoid
// runtime.complex128div, which the compiler emits for `z / complex(d, 0)` even
// though the divisor's imaginary part is always zero — a branchy Smith's algorithm
// call that profiling showed to be ~three quarters of the spectral-divide loop's
// cost. Splitting the divide into two real divides is bit-identical to what
// complex128div computes for z/(d+0i) (real/d, imag/d), so the numerics are
// unchanged; it just drops the wasted runtime call. The function is small enough
// to inline, so no call remains in the hot loop. d must be nonzero.
func divByReal(z complex128, d float64) complex128 {
	return complex(real(z)/d, imag(z)/d)
}

// divByReal32 is the single-precision analogue used by the real-FFT spectral
// divides: it divides a complex64 by a float32 scalar (the "32" denotes the
// float32 component precision, matching the float64 divByReal above). The float32
// paths carry a documented ~1e-6 accuracy contract, which a direct single-
// precision divide sits well inside.
func divByReal32(z complex64, d float32) complex64 {
	return complex(real(z)/d, imag(z)/d)
}
