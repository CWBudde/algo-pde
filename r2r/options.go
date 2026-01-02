package r2r

// Normalization controls output scaling of transforms.
type Normalization int

const (
	// NormNone leaves outputs unnormalized (default).
	NormNone Normalization = iota
	// NormOrtho applies orthonormal scaling.
	NormOrtho
)

// Options configures transform plans.
type Options struct {
	Normalization Normalization
}

// Option applies a configuration option.
type Option func(*Options)

// DefaultOptions returns default options.
func DefaultOptions() Options {
	return Options{Normalization: NormNone}
}

// WithNormalization sets the transform normalization.
func WithNormalization(norm Normalization) Option {
	return func(o *Options) {
		o.Normalization = norm
	}
}

func applyOptions(opts []Option) Options {
	base := DefaultOptions()
	for _, opt := range opts {
		opt(&base)
	}

	return base
}
