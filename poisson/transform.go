package poisson

import "github.com/MeKo-Tech/algo-pde/grid"

// AxisTransform defines the interface for axis-wise transforms.
// Implementations include FFT (periodic), DST (Dirichlet), and DCT (Neumann).
type AxisTransform interface {
	// Forward applies the forward transform along lines of the given axis.
	// The data is modified in-place.
	// shape defines the N-dimensional grid shape.
	// axis specifies which axis to transform along (0=x, 1=y, 2=z).
	Forward(data []float64, shape grid.Shape, axis int) error

	// Inverse applies the inverse transform along lines of the given axis.
	// The data is modified in-place.
	Inverse(data []float64, shape grid.Shape, axis int) error

	// Length returns the transform size along the axis.
	Length() int

	// NormalizationFactor returns the normalization factor for the round-trip.
	// After Forward then Inverse, values are scaled by this factor.
	NormalizationFactor() float64
}

// Workspace holds pre-allocated buffers for solver operations.
type Workspace struct {
	// Real holds real-valued intermediate data.
	Real []float64

	// Complex holds complex intermediate data (for FFT).
	Complex []complex128
}

// NewWorkspace creates a Workspace with the given buffer sizes.
func NewWorkspace(realSize, complexSize int) Workspace {
	return Workspace{
		Real:    make([]float64, realSize),
		Complex: make([]complex128, complexSize),
	}
}

// Bytes returns the total memory used by the Workspace in bytes.
func (w *Workspace) Bytes() int {
	return len(w.Real)*8 + len(w.Complex)*16
}
