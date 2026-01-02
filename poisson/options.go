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

	// UseRealFFT enables real FFT plans when available (2D/3D periodic).
	// This uses algo-fft's real FFT plans, which operate on float32 buffers.
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

// WithRealFFT enables or disables real FFT plans when available.
func WithRealFFT(enabled bool) Option {
	return func(o *Options) {
		o.UseRealFFT = enabled
	}
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
