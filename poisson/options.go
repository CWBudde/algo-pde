package poisson

// NullspaceHandling specifies how to handle the nullspace (constant mode)
// for boundary conditions that have zero eigenvalues (Periodic, Neumann).
type NullspaceHandling int

const (
	// NullspaceZeroMode sets the zero-mode coefficient to zero in the solution.
	// The RHS must have mean zero, otherwise the problem is inconsistent.
	// This is the default for pure Periodic or pure Neumann problems.
	NullspaceZeroMode NullspaceHandling = iota

	// NullspaceSubtractMean automatically subtracts the mean from the RHS
	// before solving, and sets the solution mean to zero.
	NullspaceSubtractMean

	// NullspaceError returns an error if the problem has a nullspace.
	// Use this when you expect a unique solution.
	NullspaceError
)

// Options configures the behavior of a Poisson solver.
type Options struct {
	// Nullspace handling for problems with zero eigenvalues.
	Nullspace NullspaceHandling

	// SolutionMean sets the mean of the solution for nullspace problems.
	// When nil, the solver leaves the mean as computed (typically zero-mode).
	SolutionMean *float64

	// UseRealFFT enables real FFT plans on 2D/3D periodic plans when the sizes
	// qualify (power-of-two extents, even last axis).
	//
	// WARNING: this is a single-vs-double precision trade, not just a layout
	// change. algo-fft's real transforms operate on float32 buffers, so the
	// whole solve runs in single precision and the result carries only ~1e-6
	// relative accuracy instead of the ~1e-14 of the default complex128 path.
	// Enable it only when that precision is acceptable (e.g. real-time visuals).
	// See WithFloat32, whose name states the trade-off outright.
	UseRealFFT bool

	// Workers is the number of parallel workers for transforms.
	// 0 means use runtime.GOMAXPROCS.
	Workers int

	// InPlace allows the solver to modify the input RHS buffer.
	// When true, Solve may use rhs as scratch space.
	InPlace bool
}

// Option is a function that modifies Options.
type Option func(*Options)

// DefaultOptions returns the default solver options.
func DefaultOptions() Options {
	return Options{
		Nullspace:    NullspaceZeroMode,
		SolutionMean: nil,
		UseRealFFT:   false,
		Workers:      0,
		InPlace:      false,
	}
}

// WithNullspace sets the nullspace handling mode.
func WithNullspace(h NullspaceHandling) Option {
	return func(o *Options) {
		o.Nullspace = h
	}
}

// WithSubtractMean enables automatic mean subtraction for nullspace handling.
func WithSubtractMean() Option {
	return func(o *Options) {
		o.Nullspace = NullspaceSubtractMean
	}
}

// WithWorkers sets the number of parallel workers.
func WithWorkers(n int) Option {
	return func(o *Options) {
		o.Workers = n
	}
}

// WithSolutionMean sets the desired mean value for the solution.
func WithSolutionMean(mean float64) Option {
	return func(o *Options) {
		m := mean
		o.SolutionMean = &m
	}
}

// WithRealFFT enables or disables the real (float32) FFT path when the plan
// sizes qualify.
//
// IMPORTANT: enabling this runs the entire solve in single precision, dropping
// accuracy from ~1e-14 to ~1e-6. WithFloat32 is the same switch under a name
// that makes the precision trade-off explicit; prefer it in new code.
func WithRealFFT(enabled bool) Option {
	return func(o *Options) {
		o.UseRealFFT = enabled
	}
}

// WithFloat32 selects the single-precision (float32) real-FFT solve path on
// 2D/3D periodic plans whose sizes qualify. It is an alias for WithRealFFT with
// a name that states the cost up front: the solution is only ~1e-6 accurate
// rather than the ~1e-14 of the default float64 path. When the sizes do not
// qualify the plan silently falls back to the float64 complex FFT; check
// UsedRealFFT to confirm which path a plan took.
func WithFloat32(enabled bool) Option {
	return WithRealFFT(enabled)
}

// WithInPlace allows the solver to modify the input RHS.
func WithInPlace(inPlace bool) Option {
	return func(o *Options) {
		o.InPlace = inPlace
	}
}

// ApplyOptions applies option functions to a base Options struct.
func ApplyOptions(base Options, opts []Option) Options {
	for _, opt := range opts {
		opt(&base)
	}

	return base
}
