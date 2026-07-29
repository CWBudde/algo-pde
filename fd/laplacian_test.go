package fd

import (
	"math"
	"testing"

	"github.com/CWBudde/algo-pde/bc"
	"github.com/CWBudde/algo-pde/grid"
)

// eigTol returns an absolute tolerance scaled to the magnitude of the expected
// eigenpair field λ·u. The discrete-Laplacian eigenvalues grow like 1/h², so a
// fixed absolute tolerance would spuriously fail on finer grids; scaling by
// |λ|·max|u| keeps the comparison meaningful at any resolution while staying
// tight at the sizes exercised here.
func eigTol(lambda float64, src []float64) float64 {
	peak := 1.0
	for _, v := range src {
		if a := math.Abs(v); a > peak {
			peak = a
		}
	}

	return 1e-12 * (1 + math.Abs(lambda)) * peak
}

func TestApply1DPeriodicModes(t *testing.T) {
	n := 16
	h := 1.0 / float64(n)
	dst := make([]float64, n)
	src := make([]float64, n)

	k := 1
	for i := range n {
		src[i] = math.Cos(2.0 * math.Pi * float64(k) * float64(i) / float64(n))
	}

	if err := Apply1D(dst, src, h, bc.Periodic); err != nil {
		t.Fatal(err)
	}
	eig := bc.EigenvaluesPeriodic(n, h)

	tol := eigTol(eig[k], src)
	for i := range n {
		want := eig[k] * src[i]
		if math.Abs(dst[i]-want) > tol {
			t.Fatalf("periodic mode k=%d i=%d: got %v want %v", k, i, dst[i], want)
		}
	}

	for i := range n {
		src[i] = 1.0
	}
	if err := Apply1D(dst, src, h, bc.Periodic); err != nil {
		t.Fatal(err)
	}
	// The constant mode lies in the nullspace; the residual scales with the
	// operator norm (~1/h²), so gauge it against that rather than a fixed value.
	tol0 := eigTol(1.0/(h*h), src)
	for i := range n {
		if math.Abs(dst[i]) > tol0 {
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

	if err := Apply1D(dst, src, h, bc.Dirichlet); err != nil {
		t.Fatal(err)
	}
	eig := bc.EigenvaluesDirichlet(n, h)

	tol := eigTol(eig[m-1], src)
	for i := range n {
		want := eig[m-1] * src[i]
		if math.Abs(dst[i]-want) > tol {
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

	if err := Apply1D(dst, src, h, bc.Neumann); err != nil {
		t.Fatal(err)
	}
	eig := bc.EigenvaluesNeumann(n, h)

	tol := eigTol(eig[m], src)
	for i := range n {
		want := eig[m] * src[i]
		if math.Abs(dst[i]-want) > tol {
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
	if err := Apply1D(want, src, h, bc.Periodic); err != nil {
		t.Fatal(err)
	}
	if err := Apply1D(src, src, h, bc.Periodic); err != nil {
		t.Fatal(err)
	}

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

	if err := Apply2D(dst, src, shape, [2]float64{hx, hy}, [2]bc.BCType{bc.Periodic, bc.Periodic}); err != nil {
		t.Fatal(err)
	}
	eigx := bc.EigenvaluesPeriodic(nx, hx)
	eigy := bc.EigenvaluesPeriodic(ny, hy)
	lambda := eigx[kx] + eigy[ky]

	tol := eigTol(lambda, src)
	for i := range nx {
		for j := range ny {
			idx := i*ny + j
			want := lambda * src[idx]
			if math.Abs(dst[idx]-want) > tol {
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

	if err := Apply2D(dst, src, shape, [2]float64{hx, hy}, [2]bc.BCType{bc.Dirichlet, bc.Dirichlet}); err != nil {
		t.Fatal(err)
	}
	eigx := bc.EigenvaluesDirichlet(nx, hx)
	eigy := bc.EigenvaluesDirichlet(ny, hy)
	lambda := eigx[mx-1] + eigy[my-1]

	tol := eigTol(lambda, src)
	for i := range nx {
		for j := range ny {
			idx := i*ny + j
			want := lambda * src[idx]
			if math.Abs(dst[idx]-want) > tol {
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

	if err := Apply2D(dst, src, shape, [2]float64{hx, hy}, [2]bc.BCType{bc.Neumann, bc.Neumann}); err != nil {
		t.Fatal(err)
	}
	eigx := bc.EigenvaluesNeumann(nx, hx)
	eigy := bc.EigenvaluesNeumann(ny, hy)
	lambda := eigx[mx] + eigy[my]

	tol := eigTol(lambda, src)
	for i := range nx {
		for j := range ny {
			idx := i*ny + j
			want := lambda * src[idx]
			if math.Abs(dst[idx]-want) > tol {
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

	if err := Apply3D(dst, src, shape, [3]float64{hx, hy, hz}, [3]bc.BCType{bc.Periodic, bc.Periodic, bc.Periodic}); err != nil {
		t.Fatal(err)
	}
	eigx := bc.EigenvaluesPeriodic(nx, hx)
	eigy := bc.EigenvaluesPeriodic(ny, hy)
	eigz := bc.EigenvaluesPeriodic(nz, hz)
	lambda := eigx[kx] + eigy[ky] + eigz[kz]

	tol := eigTol(lambda, src)
	for i := range nx {
		for j := range ny {
			for k := range nz {
				idx := (i*ny+j)*nz + k
				want := lambda * src[idx]
				if math.Abs(dst[idx]-want) > tol {
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

	if err := Apply3D(dst, src, shape, [3]float64{hx, hy, hz}, [3]bc.BCType{bc.Dirichlet, bc.Dirichlet, bc.Dirichlet}); err != nil {
		t.Fatal(err)
	}
	eigx := bc.EigenvaluesDirichlet(nx, hx)
	eigy := bc.EigenvaluesDirichlet(ny, hy)
	eigz := bc.EigenvaluesDirichlet(nz, hz)
	lambda := eigx[mx-1] + eigy[my-1] + eigz[mz-1]

	tol := eigTol(lambda, src)
	for i := range nx {
		for j := range ny {
			for k := range nz {
				idx := (i*ny+j)*nz + k
				want := lambda * src[idx]
				if math.Abs(dst[idx]-want) > tol {
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

	if err := Apply3D(dst, src, shape, [3]float64{hx, hy, hz}, [3]bc.BCType{bc.Neumann, bc.Neumann, bc.Neumann}); err != nil {
		t.Fatal(err)
	}
	eigx := bc.EigenvaluesNeumann(nx, hx)
	eigy := bc.EigenvaluesNeumann(ny, hy)
	eigz := bc.EigenvaluesNeumann(nz, hz)
	lambda := eigx[mx] + eigy[my] + eigz[mz]

	tol := eigTol(lambda, src)
	for i := range nx {
		for j := range ny {
			for k := range nz {
				idx := (i*ny+j)*nz + k
				want := lambda * src[idx]
				if math.Abs(dst[idx]-want) > tol {
					t.Fatalf("neumann 3D i=%d j=%d k=%d: got %v want %v", i, j, k, dst[idx], want)
				}
			}
		}
	}
}
