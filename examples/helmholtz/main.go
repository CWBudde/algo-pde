package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

func main() {
	// Helmholtz: (alpha - Delta) u = f
	// Here alpha > 0 (screened Poisson).
	// Let's simulate a source in the center, decaying.
	// 2D Periodic.
	
	nx, ny := 128, 128
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	
	// Screening length lambda ~ 0.1 => alpha = 1/lambda^2 = 100
	alpha := 100.0

	fmt.Printf("2D Helmholtz Solver (alpha=%.1f)\n", alpha)

	plan, err := poisson.NewHelmholtzPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Periodic, poisson.Periodic},
		alpha,
	)
	if err != nil {
		panic(err)
	}

	// Source f: Gaussian at center
	rhs := make([]float64, nx*ny)
	cx, cy := 0.5, 0.5
	sigma := 0.05
	
	for i := 0; i < nx; i++ {
		x := float64(i) * hx
		dx := x - cx
		// Handle periodicity for distance
		if dx > 0.5 { dx -= 1.0 }
		if dx < -0.5 { dx += 1.0 }
		
		for j := 0; j < ny; j++ {
			y := float64(j) * hy
			dy := y - cy
			if dy > 0.5 { dy -= 1.0 }
			if dy < -0.5 { dy += 1.0 }
			
			rhs[i*ny+j] = math.Exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
		}
	}

	u := make([]float64, nx*ny)
	if err := plan.Solve(u, rhs); err != nil {
		panic(err)
	}

	fmt.Printf("Solved. Max value: %.3f\n", maxVal(u))

	if err := savePNG("helmholtz.png", u, nx, ny); err != nil {
		panic(err)
	}
	fmt.Println("Saved helmholtz.png")
}

func maxVal(data []float64) float64 {
	m := data[0]
	for _, v := range data {
		if v > m {
			m = v
		}
	}
	return m
}

func savePNG(filename string, data []float64, nx, ny int) error {
	img := image.NewGray(image.Rect(0, 0, nx, ny))
	
	min, max := data[0], data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	
	scale := 255.0 / (max - min)

	for i := 0; i < nx; i++ {
		for j := 0; j < ny; j++ {
			val := data[i*ny+j]
			gray := uint8((val - min) * scale)
			img.Set(i, j, color.Gray{Y: gray})
		}
	}

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}
