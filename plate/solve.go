package plate

import (
	"errors"
	"fmt"
	"math"

	"github.com/cwbudde/algo-pde/eigen"
)

type SolveOptions struct {
	ModeCount     int
	Tolerance     float64
	MaxIterations int
	Seed          uint64
	ICShift       float64
}

type Mode struct {
	FrequencyHz float64
	LossFactor  float64
	Shape       []float64
	Residual    float64
}

type ModalResult struct {
	Modes      []Mode
	Iterations int
	System     *System
}

func Solve(model *Model, options SolveOptions) (*ModalResult, error) {
	system, err := Assemble(model)
	if err != nil {
		return nil, err
	}
	if options.ModeCount <= 0 || options.ModeCount >= len(system.FreeDOFs) {
		return nil, fmt.Errorf("plate: mode count must be in [1,%d)", len(system.FreeDOFs))
	}
	preconditioner, icErr := eigen.NewIC0Preconditioner(system.Stiffness, options.ICShift)
	if icErr != nil {
		preconditioner = eigen.NewDiagonalPreconditioner(system.Stiffness, 1e-14)
	}
	result, solveErr := eigen.Solve(system.Stiffness, system.Mass, eigen.Options{
		NumEigenpairs: options.ModeCount, Tolerance: options.Tolerance, MaxIterations: options.MaxIterations,
		Seed: options.Seed, Preconditioner: preconditioner,
	})
	if result == nil {
		return nil, solveErr
	}
	out := &ModalResult{Iterations: result.Iterations, System: system, Modes: make([]Mode, len(result.Eigenvalues))}
	for i, lambda := range result.Eigenvalues {
		if !(lambda > 0) || math.IsNaN(lambda) || math.IsInf(lambda, 0) {
			return nil, fmt.Errorf("plate: non-positive structural eigenvalue %g", lambda)
		}
		shape := make([]float64, 3*len(model.Mesh.Nodes))
		for free, global := range system.FreeDOFs {
			shape[global] = result.Eigenvectors[i][free]
		}
		lossNumerator := quadratic(system.LossStiffness, result.Eigenvectors[i])
		loss := lossNumerator / lambda // vectors are M-normalized.
		out.Modes[i] = Mode{FrequencyHz: math.Sqrt(lambda) / (2 * math.Pi), LossFactor: loss, Shape: shape, Residual: result.Residuals[i]}
	}
	if solveErr != nil {
		return out, errors.Join(errors.New("plate: modal solve incomplete"), solveErr)
	}
	return out, nil
}

func quadratic(operator eigen.Operator, v []float64) float64 {
	tmp := make([]float64, len(v))
	operator.MulVec(tmp, v)
	s := 0.0
	for i := range v {
		s += v[i] * tmp[i]
	}
	return s
}
