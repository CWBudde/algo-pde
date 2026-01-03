// +build js,wasm

package main

import (
	"crypto/sha256"
	"fmt"
	"math"
	"syscall/js"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

type PlanEntry struct {
	plan *poisson.Plan
	nx   int
	ny   int
	dx   float64
	dy   float64
	bcX  int
	bcY  int
}

var planCache = make(map[string]*PlanEntry)

func main() {
	// Register Go functions for JS
	js.Global().Set("goInitPlan", js.FuncOf(InitPlan))
	js.Global().Set("goSolve", js.FuncOf(Solve))
	js.Global().Set("goGetPlanInfo", js.FuncOf(GetPlanInfo))

	// Signal that WASM is ready
	js.Global().Set("goReady", js.ValueOf(true))

	// Keep the program alive
	<-make(chan struct{})
}

// InitPlan creates and caches a Helmholtz plan
// Args: nx, ny int, dx, dy float64, bcX, bcY int
// Returns: planID string
func InitPlan(this js.Value, args []js.Value) interface{} {
	if len(args) != 6 {
		return jsError("InitPlan requires 6 arguments: nx, ny, dx, dy, bcX, bcY")
	}

	nx := args[0].Int()
	ny := args[1].Int()
	dx := args[2].Float()
	dy := args[3].Float()
	bcX := args[4].Int()
	bcY := args[5].Int()

	// Validate inputs
	if nx < 1 || ny < 1 {
		return jsError("Grid dimensions must be positive")
	}
	if dx <= 0 || dy <= 0 {
		return jsError("Grid spacing must be positive")
	}
	if bcX < 0 || bcX > 2 || bcY < 0 || bcY > 2 {
		return jsError("Boundary conditions must be 0 (Periodic), 1 (Dirichlet), or 2 (Neumann)")
	}

	// Generate plan ID from parameters
	planID := generatePlanID(nx, ny, dx, dy, bcX, bcY)

	// Check if plan already exists
	if _, exists := planCache[planID]; exists {
		return jsSuccess(map[string]interface{}{
			"planID": planID,
			"nx":     nx,
			"ny":     ny,
			"cached": true,
		})
	}

	// Create new plan
	// Note: We'll set alpha=0 initially, will be overridden per solve
	plan, err := poisson.NewHelmholtzPlan(
		2,
		[]int{nx, ny},
		[]float64{dx, dy},
		[]poisson.BCType{poisson.BCType(bcX), poisson.BCType(bcY)},
		0, // alpha will be set per-solve via new plan creation
	)
	if err != nil {
		return jsError(fmt.Sprintf("Failed to create plan: %v", err))
	}

	planCache[planID] = &PlanEntry{
		plan: plan,
		nx:   nx,
		ny:   ny,
		dx:   dx,
		dy:   dy,
		bcX:  bcX,
		bcY:  bcY,
	}

	return jsSuccess(map[string]interface{}{
		"planID": planID,
		"nx":     nx,
		"ny":     ny,
		"cached": false,
	})
}

// Solve applies Gaussian source and solves Helmholtz equation
// Args: planID string, alpha float64, sx, sy, srcRadius float64
// Returns: Float32Array of solution field
func Solve(this js.Value, args []js.Value) interface{} {
	if len(args) != 5 {
		return jsError("Solve requires 5 arguments: planID, alpha, sx, sy, srcRadius")
	}

	planID := args[0].String()
	alpha := args[1].Float()
	sx := args[2].Float()
	sy := args[3].Float()
	srcRadius := args[4].Float()

	// Get cached plan metadata
	entry, exists := planCache[planID]
	if !exists {
		return jsError("Plan not found. Call InitPlan first.")
	}

	// Create new Helmholtz plan with specific alpha
	// This is necessary because alpha is baked into the plan
	plan, err := poisson.NewHelmholtzPlan(
		2,
		[]int{entry.nx, entry.ny},
		[]float64{entry.dx, entry.dy},
		[]poisson.BCType{poisson.BCType(entry.bcX), poisson.BCType(entry.bcY)},
		alpha,
	)
	if err != nil {
		return jsError(fmt.Sprintf("Failed to create Helmholtz plan: %v", err))
	}

	// Build Gaussian source
	rhs := buildGaussianSource(entry.nx, entry.ny, sx, sy, srcRadius)

	// Solve
	dst := make([]float64, entry.nx*entry.ny)
	if err := plan.Solve(dst, rhs); err != nil {
		return jsError(fmt.Sprintf("Solve failed: %v", err))
	}

	// Convert to Float32 and return as JS TypedArray
	float32Dst := make([]float32, len(dst))
	for i, v := range dst {
		float32Dst[i] = float32(v)
	}

	// Create JS Float32Array
	jsArray := js.Global().Get("Float32Array").New(len(float32Dst))

	// Copy data element by element
	// Note: We can't use CopyBytesToJS with Float32Array, only with Uint8Array/Uint8ClampedArray
	for i, v := range float32Dst {
		jsArray.SetIndex(i, js.ValueOf(v))
	}

	return jsSuccess(map[string]interface{}{
		"field": jsArray,
	})
}

// GetPlanInfo returns grid dimensions for a plan
// Args: planID string
// Returns: {nx, ny, dx, dy}
func GetPlanInfo(this js.Value, args []js.Value) interface{} {
	if len(args) != 1 {
		return jsError("GetPlanInfo requires 1 argument: planID")
	}

	planID := args[0].String()
	entry, exists := planCache[planID]
	if !exists {
		return jsError("Plan not found")
	}

	return jsSuccess(map[string]interface{}{
		"nx": entry.nx,
		"ny": entry.ny,
		"dx": entry.dx,
		"dy": entry.dy,
	})
}

// Helper functions

func generatePlanID(nx, ny int, dx, dy float64, bcX, bcY int) string {
	key := fmt.Sprintf("%d_%d_%.10f_%.10f_%d_%d", nx, ny, dx, dy, bcX, bcY)
	hash := sha256.Sum256([]byte(key))
	return fmt.Sprintf("%x", hash[:8])
}

func buildGaussianSource(nx, ny int, sx, sy, radius float64) []float64 {
	rhs := make([]float64, nx*ny)
	r2 := radius * radius

	for y := 0; y < ny; y++ {
		for x := 0; x < nx; x++ {
			dx := float64(x) - sx
			dy := float64(y) - sy
			dist2 := dx*dx + dy*dy
			idx := y*nx + x
			rhs[idx] = math.Exp(-dist2 / (2.0 * r2))
		}
	}

	return rhs
}

func jsSuccess(data map[string]interface{}) interface{} {
	result := map[string]interface{}{
		"success": true,
	}
	for k, v := range data {
		result[k] = v
	}
	return result
}

func jsError(message string) interface{} {
	return map[string]interface{}{
		"success": false,
		"error":   message,
	}
}
