package r2r

import (
	"errors"
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/grid"
)

func TestDSTPlan_ForwardLines_2D_Axis0(t *testing.T) {
	// 2D array: 7 rows x 4 columns
	// Transform along axis 0 (columns): each column has 7 elements
	nx, ny := 7, 4
	shape := grid.NewShape2D(nx, ny)

	plan, err := NewDSTPlan(nx)
	if err != nil {
		t.Fatalf("NewDSTPlan failed: %v", err)
	}

	// Create test data: each column is a sine mode
	data := make([]float64, nx*ny)
	expected := make([]float64, nx*ny)

	for j := range ny {
		// Column j contains mode j+1
		mode := j + 1
		for i := range nx {
			data[i*ny+j] = DST1Coefficient(i, mode, nx)
		}
		// After DST, coefficient at mode position should be (N+1)/2
		for k := range nx {
			if k == mode {
				expected[k*ny+j] = float64(nx+1) / 2.0
			}
		}
	}

	if err := plan.ForwardLines(data, shape, 0); err != nil {
		t.Fatalf("ForwardLines failed: %v", err)
	}

	for idx := range data {
		if math.Abs(data[idx]-expected[idx]) > tolerance {
			i, j := idx/ny, idx%ny
			t.Errorf("data[%d,%d] = %v, want %v", i, j, data[idx], expected[idx])
		}
	}
}

func TestDSTPlan_ForwardLines_2D_Axis1(t *testing.T) {
	// 2D array: 4 rows x 7 columns
	// Transform along axis 1 (rows): each row has 7 elements
	nx, ny := 4, 7
	shape := grid.NewShape2D(nx, ny)

	plan, err := NewDSTPlan(ny)
	if err != nil {
		t.Fatalf("NewDSTPlan failed: %v", err)
	}

	// Create test data: each row is a sine mode
	data := make([]float64, nx*ny)
	expected := make([]float64, nx*ny)

	for i := range nx {
		// Row i contains mode i+1
		mode := i + 1
		for j := range ny {
			data[i*ny+j] = DST1Coefficient(j, mode, ny)
		}
		// After DST, coefficient at mode position should be (N+1)/2
		for k := range ny {
			if k == mode {
				expected[i*ny+k] = float64(ny+1) / 2.0
			}
		}
	}

	if err := plan.ForwardLines(data, shape, 1); err != nil {
		t.Fatalf("ForwardLines failed: %v", err)
	}

	for idx := range data {
		if math.Abs(data[idx]-expected[idx]) > tolerance {
			i, j := idx/ny, idx%ny
			t.Errorf("data[%d,%d] = %v, want %v", i, j, data[idx], expected[idx])
		}
	}
}

func TestDSTPlan_RoundTripLines_2D(t *testing.T) {
	nx, ny := 8, 6
	shape := grid.NewShape2D(nx, ny)

	planX, err := NewDSTPlan(nx)
	if err != nil {
		t.Fatalf("NewDSTPlan(nx) failed: %v", err)
	}

	planY, err := NewDSTPlan(ny)
	if err != nil {
		t.Fatalf("NewDSTPlan(ny) failed: %v", err)
	}

	// Create test data
	data := make([]float64, nx*ny)
	original := make([]float64, nx*ny)
	for i := range data {
		data[i] = float64(i + 1)
		original[i] = data[i]
	}

	// Forward along axis 0, then axis 1
	if err := planX.ForwardLines(data, shape, 0); err != nil {
		t.Fatalf("ForwardLines axis 0 failed: %v", err)
	}
	if err := planY.ForwardLines(data, shape, 1); err != nil {
		t.Fatalf("ForwardLines axis 1 failed: %v", err)
	}

	// Inverse along axis 1, then axis 0
	if err := planY.InverseLines(data, shape, 1); err != nil {
		t.Fatalf("InverseLines axis 1 failed: %v", err)
	}
	if err := planX.InverseLines(data, shape, 0); err != nil {
		t.Fatalf("InverseLines axis 0 failed: %v", err)
	}

	// Check round-trip
	for i := range data {
		if math.Abs(data[i]-original[i]) > tolerance {
			t.Errorf("round-trip mismatch at [%d]: got %v, want %v", i, data[i], original[i])
		}
	}
}

func TestDCTPlan_ForwardLines_2D_Axis0(t *testing.T) {
	// 2D array: 8 rows x 4 columns
	// Transform along axis 0 (columns): each column has 8 elements
	nx, ny := 8, 4
	shape := grid.NewShape2D(nx, ny)

	plan, err := NewDCTPlan(nx)
	if err != nil {
		t.Fatalf("NewDCTPlan failed: %v", err)
	}

	// Create test data: constant in each column
	data := make([]float64, nx*ny)
	for j := range ny {
		for i := range nx {
			data[i*ny+j] = 1.0
		}
	}

	if err := plan.ForwardLines(data, shape, 0); err != nil {
		t.Fatalf("ForwardLines failed: %v", err)
	}

	// For constant input, only k=0 coefficient should be non-zero
	for j := range ny {
		for k := 1; k < nx; k++ {
			if math.Abs(data[k*ny+j]) > tolerance {
				t.Errorf("data[%d,%d] = %v, want 0 for constant input", k, j, data[k*ny+j])
			}
		}
	}
}

func TestDCTPlan_ForwardLines_2D_Axis1(t *testing.T) {
	// 2D array: 4 rows x 8 columns
	// Transform along axis 1 (rows): each row has 8 elements
	nx, ny := 4, 8
	shape := grid.NewShape2D(nx, ny)

	plan, err := NewDCTPlan(ny)
	if err != nil {
		t.Fatalf("NewDCTPlan failed: %v", err)
	}

	// Create test data: constant in each row
	data := make([]float64, nx*ny)
	for i := range nx {
		for j := range ny {
			data[i*ny+j] = 1.0
		}
	}

	if err := plan.ForwardLines(data, shape, 1); err != nil {
		t.Fatalf("ForwardLines failed: %v", err)
	}

	// For constant input, only k=0 coefficient should be non-zero
	for i := range nx {
		for k := 1; k < ny; k++ {
			if math.Abs(data[i*ny+k]) > tolerance {
				t.Errorf("data[%d,%d] = %v, want 0 for constant input", i, k, data[i*ny+k])
			}
		}
	}
}

func TestDCTPlan_RoundTripLines_2D(t *testing.T) {
	nx, ny := 8, 6
	shape := grid.NewShape2D(nx, ny)

	planX, err := NewDCTPlan(nx)
	if err != nil {
		t.Fatalf("NewDCTPlan(nx) failed: %v", err)
	}

	planY, err := NewDCTPlan(ny)
	if err != nil {
		t.Fatalf("NewDCTPlan(ny) failed: %v", err)
	}

	// Create test data
	data := make([]float64, nx*ny)
	original := make([]float64, nx*ny)
	for i := range data {
		data[i] = float64(i + 1)
		original[i] = data[i]
	}

	// Forward along axis 0, then axis 1
	if err := planX.ForwardLines(data, shape, 0); err != nil {
		t.Fatalf("ForwardLines axis 0 failed: %v", err)
	}
	if err := planY.ForwardLines(data, shape, 1); err != nil {
		t.Fatalf("ForwardLines axis 1 failed: %v", err)
	}

	// Inverse along axis 1, then axis 0
	if err := planY.InverseLines(data, shape, 1); err != nil {
		t.Fatalf("InverseLines axis 1 failed: %v", err)
	}
	if err := planX.InverseLines(data, shape, 0); err != nil {
		t.Fatalf("InverseLines axis 0 failed: %v", err)
	}

	// Check round-trip
	for i := range data {
		if math.Abs(data[i]-original[i]) > tolerance {
			t.Errorf("round-trip mismatch at [%d]: got %v, want %v", i, data[i], original[i])
		}
	}
}

func TestDSTPlan_ForwardLines_3D(t *testing.T) {
	nx, ny, nz := 7, 5, 4
	shape := grid.NewShape3D(nx, ny, nz)

	plan, err := NewDSTPlan(ny)
	if err != nil {
		t.Fatalf("NewDSTPlan failed: %v", err)
	}

	// Create test data
	data := make([]float64, nx*ny*nz)
	original := make([]float64, nx*ny*nz)
	for i := range data {
		data[i] = float64(i + 1)
		original[i] = data[i]
	}

	// Forward then inverse along axis 1 should recover original
	if err := plan.ForwardLines(data, shape, 1); err != nil {
		t.Fatalf("ForwardLines failed: %v", err)
	}
	if err := plan.InverseLines(data, shape, 1); err != nil {
		t.Fatalf("InverseLines failed: %v", err)
	}

	for i := range data {
		if math.Abs(data[i]-original[i]) > tolerance {
			t.Errorf("3D round-trip mismatch at [%d]: got %v, want %v", i, data[i], original[i])
		}
	}
}

func TestDCTPlan_ForwardLines_3D(t *testing.T) {
	nx, ny, nz := 8, 6, 4
	shape := grid.NewShape3D(nx, ny, nz)

	plan, err := NewDCTPlan(nz)
	if err != nil {
		t.Fatalf("NewDCTPlan failed: %v", err)
	}

	// Create test data
	data := make([]float64, nx*ny*nz)
	original := make([]float64, nx*ny*nz)
	for i := range data {
		data[i] = float64(i + 1)
		original[i] = data[i]
	}

	// Forward then inverse along axis 2 should recover original
	if err := plan.ForwardLines(data, shape, 2); err != nil {
		t.Fatalf("ForwardLines failed: %v", err)
	}
	if err := plan.InverseLines(data, shape, 2); err != nil {
		t.Fatalf("InverseLines failed: %v", err)
	}

	for i := range data {
		if math.Abs(data[i]-original[i]) > tolerance {
			t.Errorf("3D round-trip mismatch at [%d]: got %v, want %v", i, data[i], original[i])
		}
	}
}

func TestForwardLines_SizeMismatch(t *testing.T) {
	shape := grid.NewShape2D(8, 6)

	// DST plan size doesn't match any axis
	dstPlan, _ := NewDSTPlan(10)
	if err := dstPlan.ForwardLines(make([]float64, 48), shape, 0); !errors.Is(err, ErrSizeMismatch) {
		t.Errorf("DST ForwardLines size mismatch: got %v, want ErrSizeMismatch", err)
	}

	// DCT plan size doesn't match any axis
	dctPlan, _ := NewDCTPlan(10)
	if err := dctPlan.ForwardLines(make([]float64, 48), shape, 1); !errors.Is(err, ErrSizeMismatch) {
		t.Errorf("DCT ForwardLines size mismatch: got %v, want ErrSizeMismatch", err)
	}
}

func BenchmarkDSTPlan_ForwardLines_2D(b *testing.B) {
	sizes := []struct{ nx, ny int }{
		{64, 64},
		{128, 128},
		{256, 256},
	}

	for _, sz := range sizes {
		b.Run(sizeStr(sz.nx)+"x"+sizeStr(sz.ny), func(b *testing.B) {
			shape := grid.NewShape2D(sz.nx, sz.ny)
			plan, _ := NewDSTPlan(sz.nx)
			data := make([]float64, sz.nx*sz.ny)

			b.ResetTimer()
			for range b.N {
				_ = plan.ForwardLines(data, shape, 0)
			}
		})
	}
}

func BenchmarkDCTPlan_ForwardLines_2D(b *testing.B) {
	sizes := []struct{ nx, ny int }{
		{64, 64},
		{128, 128},
		{256, 256},
	}

	for _, sz := range sizes {
		b.Run(sizeStr(sz.nx)+"x"+sizeStr(sz.ny), func(b *testing.B) {
			shape := grid.NewShape2D(sz.nx, sz.ny)
			plan, _ := NewDCTPlan(sz.nx)
			data := make([]float64, sz.nx*sz.ny)

			b.ResetTimer()
			for range b.N {
				_ = plan.ForwardLines(data, shape, 0)
			}
		})
	}
}
