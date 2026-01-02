package fd

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

func TestApply1DPeriodicModes(t *testing.T) {
	n := 16
	h := 1.0 / float64(n)
	dst := make([]float64, n)
	src := make([]float64, n)

	k := 1
	for i := range n {
		src[i] = math.Cos(2.0 * math.Pi * float64(k) * float64(i) / float64(n))
	}

	Apply1D(dst, src, h, poisson.Periodic)
	eig := EigenvaluesPeriodic(n, h)

	for i := range n {
		want := eig[k] * src[i]
		if math.Abs(dst[i]-want) > 1e-12 {
			t.Fatalf("periodic mode k=%d i=%d: got %v want %v", k, i, dst[i], want)
		}
	}

	for i := range n {
		src[i] = 1.0
	}
	Apply1D(dst, src, h, poisson.Periodic)
	for i := range n {
		if math.Abs(dst[i]) > 1e-12 {
			t.Fatalf("periodic constant mode i=%d: got %v want 0", i, dst[i])
		}
	}
}

func TestApply1DDirichletModes(t *testing.T) {
	n := 12
	h := 1.0 / float64(n+1)
	dst := make([]float64, n)
	src := make([]float64, n)

	m := 2
	for i := range n {
		x := float64(i+1) / float64(n+1)
		src[i] = math.Sin(math.Pi * float64(m) * x)
	}

	Apply1D(dst, src, h, poisson.Dirichlet)
	eig := EigenvaluesDirichlet(n, h)

	for i := range n {
		want := eig[m-1] * src[i]
		if math.Abs(dst[i]-want) > 1e-12 {
			t.Fatalf("dirichlet mode m=%d i=%d: got %v want %v", m, i, dst[i], want)
		}
	}
}

func TestApply1DNeumannModes(t *testing.T) {
	n := 10
	h := 1.0
	dst := make([]float64, n)
	src := make([]float64, n)

	m := 1
	for i := range n {
		x := (float64(i) + 0.5) / float64(n)
		src[i] = math.Cos(math.Pi * float64(m) * x)
	}

	Apply1D(dst, src, h, poisson.Neumann)
	eig := EigenvaluesNeumann(n, h)

	for i := range n {
		want := eig[m] * src[i]
		if math.Abs(dst[i]-want) > 1e-12 {
			t.Fatalf("neumann mode m=%d i=%d: got %v want %v", m, i, dst[i], want)
		}
	}
}

func TestApply1DInPlace(t *testing.T) {
	n := 8
	h := 1.0
	src := make([]float64, n)

	for i := range n {
		src[i] = math.Sin(2.0 * math.Pi * float64(i) / float64(n))
	}

	want := make([]float64, n)
	Apply1D(want, src, h, poisson.Periodic)
	Apply1D(src, src, h, poisson.Periodic)

	for i := range n {
		if math.Abs(src[i]-want[i]) > 1e-12 {
			t.Fatalf("in-place i=%d: got %v want %v", i, src[i], want[i])
		}
	}
}

func TestApply2DPeriodicModes(t *testing.T) {
	nx, ny := 12, 10
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	shape := grid.NewShape2D(nx, ny)
	src := make([]float64, nx*ny)
	dst := make([]float64, nx*ny)

	kx, ky := 2, 3
	for i := range nx {
		x := float64(i) / float64(nx)
		for j := range ny {
			y := float64(j) / float64(ny)
			src[i*ny+j] = math.Cos(2.0*math.Pi*float64(kx)*x) * math.Cos(2.0*math.Pi*float64(ky)*y)
		}
	}

	Apply2D(dst, src, shape, [2]float64{hx, hy}, [2]poisson.BCType{poisson.Periodic, poisson.Periodic})
	eigx := EigenvaluesPeriodic(nx, hx)
	eigy := EigenvaluesPeriodic(ny, hy)
	lambda := eigx[kx] + eigy[ky]

	for i := range nx {
		for j := range ny {
			idx := i*ny + j
			want := lambda * src[idx]
			if math.Abs(dst[idx]-want) > 1e-12 {
				t.Fatalf("periodic 2D i=%d j=%d: got %v want %v", i, j, dst[idx], want)
			}
		}
	}
}

func TestApply2DDirichletModes(t *testing.T) {
	nx, ny := 11, 9
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	shape := grid.NewShape2D(nx, ny)
	src := make([]float64, nx*ny)
	dst := make([]float64, nx*ny)

	mx, my := 1, 2
	for i := range nx {
		x := float64(i+1) / float64(nx+1)
		for j := range ny {
			y := float64(j+1) / float64(ny+1)
			src[i*ny+j] = math.Sin(math.Pi*float64(mx)*x) * math.Sin(math.Pi*float64(my)*y)
		}
	}

	Apply2D(dst, src, shape, [2]float64{hx, hy}, [2]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet})
	eigx := EigenvaluesDirichlet(nx, hx)
	eigy := EigenvaluesDirichlet(ny, hy)
	lambda := eigx[mx-1] + eigy[my-1]

	for i := range nx {
		for j := range ny {
			idx := i*ny + j
			want := lambda * src[idx]
			if math.Abs(dst[idx]-want) > 1e-12 {
				t.Fatalf("dirichlet 2D i=%d j=%d: got %v want %v", i, j, dst[idx], want)
			}
		}
	}
}

func TestApply2DNeumannModes(t *testing.T) {
	nx, ny := 10, 8
	hx := 1.0
	hy := 1.0
	shape := grid.NewShape2D(nx, ny)
	src := make([]float64, nx*ny)
	dst := make([]float64, nx*ny)

	mx, my := 1, 2
	for i := range nx {
		x := (float64(i) + 0.5) / float64(nx)
		for j := range ny {
			y := (float64(j) + 0.5) / float64(ny)
			src[i*ny+j] = math.Cos(math.Pi*float64(mx)*x) * math.Cos(math.Pi*float64(my)*y)
		}
	}

	Apply2D(dst, src, shape, [2]float64{hx, hy}, [2]poisson.BCType{poisson.Neumann, poisson.Neumann})
	eigx := EigenvaluesNeumann(nx, hx)
	eigy := EigenvaluesNeumann(ny, hy)
	lambda := eigx[mx] + eigy[my]

	for i := range nx {
		for j := range ny {
			idx := i*ny + j
			want := lambda * src[idx]
			if math.Abs(dst[idx]-want) > 1e-12 {
				t.Fatalf("neumann 2D i=%d j=%d: got %v want %v", i, j, dst[idx], want)
			}
		}
	}
}

func TestApply3DPeriodicModes(t *testing.T) {
	nx, ny, nz := 8, 6, 10
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	hz := 1.0 / float64(nz)
	shape := grid.NewShape3D(nx, ny, nz)
	src := make([]float64, nx*ny*nz)
	dst := make([]float64, nx*ny*nz)

	kx, ky, kz := 1, 2, 3
	for i := range nx {
		x := float64(i) / float64(nx)
		for j := range ny {
			y := float64(j) / float64(ny)
			for k := range nz {
				z := float64(k) / float64(nz)
				idx := (i*ny+j)*nz + k
				src[idx] = math.Cos(2.0*math.Pi*float64(kx)*x) *
					math.Cos(2.0*math.Pi*float64(ky)*y) *
					math.Cos(2.0*math.Pi*float64(kz)*z)
			}
		}
	}

	Apply3D(dst, src, shape, [3]float64{hx, hy, hz}, [3]poisson.BCType{poisson.Periodic, poisson.Periodic, poisson.Periodic})
	eigx := EigenvaluesPeriodic(nx, hx)
	eigy := EigenvaluesPeriodic(ny, hy)
	eigz := EigenvaluesPeriodic(nz, hz)
	lambda := eigx[kx] + eigy[ky] + eigz[kz]

	for i := range nx {
		for j := range ny {
			for k := range nz {
				idx := (i*ny+j)*nz + k
				want := lambda * src[idx]
				if math.Abs(dst[idx]-want) > 1e-12 {
					t.Fatalf("periodic 3D i=%d j=%d k=%d: got %v want %v", i, j, k, dst[idx], want)
				}
			}
		}
	}
}

func TestApply3DDirichletModes(t *testing.T) {
	nx, ny, nz := 7, 5, 6
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	hz := 1.0 / float64(nz+1)
	shape := grid.NewShape3D(nx, ny, nz)
	src := make([]float64, nx*ny*nz)
	dst := make([]float64, nx*ny*nz)

	mx, my, mz := 1, 2, 1
	for i := range nx {
		x := float64(i+1) / float64(nx+1)
		for j := range ny {
			y := float64(j+1) / float64(ny+1)
			for k := range nz {
				z := float64(k+1) / float64(nz+1)
				idx := (i*ny+j)*nz + k
				src[idx] = math.Sin(math.Pi*float64(mx)*x) *
					math.Sin(math.Pi*float64(my)*y) *
					math.Sin(math.Pi*float64(mz)*z)
			}
		}
	}

	Apply3D(dst, src, shape, [3]float64{hx, hy, hz}, [3]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet, poisson.Dirichlet})
	eigx := EigenvaluesDirichlet(nx, hx)
	eigy := EigenvaluesDirichlet(ny, hy)
	eigz := EigenvaluesDirichlet(nz, hz)
	lambda := eigx[mx-1] + eigy[my-1] + eigz[mz-1]

	for i := range nx {
		for j := range ny {
			for k := range nz {
				idx := (i*ny+j)*nz + k
				want := lambda * src[idx]
				if math.Abs(dst[idx]-want) > 1e-12 {
					t.Fatalf("dirichlet 3D i=%d j=%d k=%d: got %v want %v", i, j, k, dst[idx], want)
				}
			}
		}
	}
}

func TestApply3DNeumannModes(t *testing.T) {
	nx, ny, nz := 9, 7, 8
	hx := 1.0
	hy := 1.0
	hz := 1.0
	shape := grid.NewShape3D(nx, ny, nz)
	src := make([]float64, nx*ny*nz)
	dst := make([]float64, nx*ny*nz)

	mx, my, mz := 1, 1, 2
	for i := range nx {
		x := (float64(i) + 0.5) / float64(nx)
		for j := range ny {
			y := (float64(j) + 0.5) / float64(ny)
			for k := range nz {
				z := (float64(k) + 0.5) / float64(nz)
				idx := (i*ny+j)*nz + k
				src[idx] = math.Cos(math.Pi*float64(mx)*x) *
					math.Cos(math.Pi*float64(my)*y) *
					math.Cos(math.Pi*float64(mz)*z)
			}
		}
	}

	Apply3D(dst, src, shape, [3]float64{hx, hy, hz}, [3]poisson.BCType{poisson.Neumann, poisson.Neumann, poisson.Neumann})
	eigx := EigenvaluesNeumann(nx, hx)
	eigy := EigenvaluesNeumann(ny, hy)
	eigz := EigenvaluesNeumann(nz, hz)
	lambda := eigx[mx] + eigy[my] + eigz[mz]

	for i := range nx {
		for j := range ny {
			for k := range nz {
				idx := (i*ny+j)*nz + k
				want := lambda * src[idx]
				if math.Abs(dst[idx]-want) > 1e-12 {
					t.Fatalf("neumann 3D i=%d j=%d k=%d: got %v want %v", i, j, k, dst[idx], want)
				}
			}
		}
	}
}
