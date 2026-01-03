package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

func BenchmarkHelmholtz2D_Spectral(b *testing.B) {
	nx, ny := 128, 128
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	Lx := float64(nx+1) * hx
	Ly := float64(ny+1) * hy
	alpha := 2.0

	plan, err := poisson.NewHelmholtzPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
		alpha,
	)
	if err != nil {
		b.Fatalf("NewHelmholtzPlan failed: %v", err)
	}

	u := make([]float64, nx*ny)
	for i := 0; i < nx; i++ {
		x := float64(i+1) * hx
		for j := 0; j < ny; j++ {
			y := float64(j+1) * hy
			u[i*ny+j] = math.Sin(math.Pi*x/Lx) * math.Sin(2.0*math.Pi*y/Ly)
		}
	}

	lap := make([]float64, nx*ny)
	fd.Apply2D(lap, u, grid.Shape{nx, ny}, [2]float64{hx, hy}, [2]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet})

	rhs := make([]float64, nx*ny)
	for i := range rhs {
		rhs[i] = lap[i] + alpha*u[i]
	}

	dst := make([]float64, nx*ny)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := plan.Solve(dst, rhs); err != nil {
			b.Fatalf("Solve failed: %v", err)
		}
	}
}

func BenchmarkHelmholtz2D_Jacobi(b *testing.B) {
	nx, ny := 128, 128
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	Lx := float64(nx+1) * hx
	Ly := float64(ny+1) * hy
	alpha := 2.0
	iters := 200

	u := make([]float64, nx*ny)
	for i := 0; i < nx; i++ {
		x := float64(i+1) * hx
		for j := 0; j < ny; j++ {
			y := float64(j+1) * hy
			u[i*ny+j] = math.Sin(math.Pi*x/Lx) * math.Sin(2.0*math.Pi*y/Ly)
		}
	}

	lap := make([]float64, nx*ny)
	fd.Apply2D(lap, u, grid.Shape{nx, ny}, [2]float64{hx, hy}, [2]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet})

	rhs := make([]float64, nx*ny)
	for i := range rhs {
		rhs[i] = lap[i] + alpha*u[i]
	}

	cur := make([]float64, nx*ny)
	next := make([]float64, nx*ny)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range cur {
			cur[j] = 0
		}
		jacobiHelmholtz2D(cur, next, rhs, nx, ny, hx, hy, alpha, iters)
	}
}

func jacobiHelmholtz2D(cur, next, rhs []float64, nx, ny int, hx, hy, alpha float64, iters int) {
	invHx2 := 1.0 / (hx * hx)
	invHy2 := 1.0 / (hy * hy)
	diag := alpha + 2.0*invHx2 + 2.0*invHy2

	for iter := 0; iter < iters; iter++ {
		for i := 0; i < nx; i++ {
			row := i * ny
			for j := 0; j < ny; j++ {
				idx := row + j

				left := 0.0
				if i > 0 {
					left = cur[(i-1)*ny+j]
				}
				right := 0.0
				if i+1 < nx {
					right = cur[(i+1)*ny+j]
				}

				down := 0.0
				if j > 0 {
					down = cur[row+j-1]
				}
				up := 0.0
				if j+1 < ny {
					up = cur[row+j+1]
				}

				next[idx] = (rhs[idx] + (left+right)*invHx2 + (down+up)*invHy2) / diag
			}
		}
		cur, next = next, cur
	}

	if iters%2 != 0 {
		copy(next, cur)
	}
}
