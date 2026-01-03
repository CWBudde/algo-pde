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
	nx, ny := 256, 256
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)

	fmt.Printf("Generating 2D Periodic Poisson example (%dx%d)...\n", nx, ny)

	plan, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy)
	if err != nil {
		panic(err)
	}

	// u_exact = sin(2*pi*x) * sin(2*pi*y)
	// -Lap u = 8*pi^2 * u
	rhs := make([]float64, nx*ny)
	uExact := make([]float64, nx*ny)

	for i := 0; i < nx; i++ {
		x := float64(i) * hx
		for j := 0; j < ny; j++ {
			y := float64(j) * hy
			val := math.Sin(2.0*math.Pi*x) * math.Sin(2.0*math.Pi*y)
			uExact[i*ny+j] = val
			rhs[i*ny+j] = 8.0 * math.Pi * math.Pi * val
		}
	}

	u := make([]float64, nx*ny)
	if err := plan.Solve(u, rhs); err != nil {
		panic(err)
	}

	maxErr := 0.0
	for i := range u {
		diff := math.Abs(u[i] - uExact[i])
		if diff > maxErr {
			maxErr = diff
		}
	}
	fmt.Printf("Max Error: %.3e\n", maxErr)

	if err := savePNG("solution.png", u, nx, ny); err != nil {
		panic(err)
	}
	fmt.Println("Saved solution.png")
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
