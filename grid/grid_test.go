package grid

import "testing"

func TestShape_Dim(t *testing.T) {
	tests := []struct {
		name  string
		shape Shape
		want  int
	}{
		{"1D", NewShape1D(10), 1},
		{"2D", NewShape2D(10, 20), 2},
		{"3D", NewShape3D(10, 20, 30), 3},
		{"2D with nz=1", Shape{10, 20, 1}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.shape.Dim(); got != tt.want {
				t.Errorf("Shape.Dim() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestShape_Size(t *testing.T) {
	tests := []struct {
		name  string
		shape Shape
		want  int
	}{
		{"1D", NewShape1D(10), 10},
		{"2D", NewShape2D(10, 20), 200},
		{"3D", NewShape3D(10, 20, 30), 6000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.shape.Size(); got != tt.want {
				t.Errorf("Shape.Size() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIndex2D(t *testing.T) {
	ny := 5

	tests := []struct {
		i, j int
		want int
	}{
		{0, 0, 0},
		{0, 1, 1},
		{1, 0, 5},
		{2, 3, 13}, // 2*5 + 3 = 13
	}
	for _, tt := range tests {
		got := Index2D(tt.i, tt.j, ny)
		if got != tt.want {
			t.Errorf("Index2D(%d, %d, %d) = %v, want %v", tt.i, tt.j, ny, got, tt.want)
		}
	}
}

func TestIndex3D(t *testing.T) {
	shape := NewShape3D(4, 5, 6)

	tests := []struct {
		i, j, k int
		want    int
	}{
		{0, 0, 0, 0},
		{0, 0, 1, 1},
		{0, 1, 0, 6},
		{1, 0, 0, 30}, // 1 * 5 * 6 = 30
		{1, 2, 3, 45}, // 1*30 + 2*6 + 3 = 45
	}
	for _, tt := range tests {
		got := Index3D(tt.i, tt.j, tt.k, shape)
		if got != tt.want {
			t.Errorf("Index3D(%d, %d, %d, shape) = %v, want %v", tt.i, tt.j, tt.k, got, tt.want)
		}
	}
}

func TestFromIndex2D(t *testing.T) {
	ny := 5

	tests := []struct {
		idx   int
		wantI int
		wantJ int
	}{
		{0, 0, 0},
		{1, 0, 1},
		{5, 1, 0},
		{13, 2, 3},
	}
	for _, tt := range tests {
		gotI, gotJ := FromIndex2D(tt.idx, ny)
		if gotI != tt.wantI || gotJ != tt.wantJ {
			t.Errorf("FromIndex2D(%d, %d) = (%v, %v), want (%v, %v)",
				tt.idx, ny, gotI, gotJ, tt.wantI, tt.wantJ)
		}
	}
}

func TestFromIndex3D(t *testing.T) {
	shape := NewShape3D(4, 5, 6)

	tests := []struct {
		idx   int
		wantI int
		wantJ int
		wantK int
	}{
		{0, 0, 0, 0},
		{1, 0, 0, 1},
		{6, 0, 1, 0},
		{30, 1, 0, 0},
		{45, 1, 2, 3},
	}
	for _, tt := range tests {
		gotI, gotJ, gotK := FromIndex3D(tt.idx, shape)
		if gotI != tt.wantI || gotJ != tt.wantJ || gotK != tt.wantK {
			t.Errorf("FromIndex3D(%d, shape) = (%v, %v, %v), want (%v, %v, %v)",
				tt.idx, gotI, gotJ, gotK, tt.wantI, tt.wantJ, tt.wantK)
		}
	}
}

func TestRowMajorStride(t *testing.T) {
	tests := []struct {
		name  string
		shape Shape
		want  Stride
	}{
		{"1D", NewShape1D(10), Stride{1, 1, 1}},
		{"2D", NewShape2D(4, 5), Stride{5, 1, 1}},
		{"3D", NewShape3D(4, 5, 6), Stride{30, 6, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := RowMajorStride(tt.shape)
			if got != tt.want {
				t.Errorf("RowMajorStride(%v) = %v, want %v", tt.shape, got, tt.want)
			}
		})
	}
}

func TestLineIterator_2D_Axis0(t *testing.T) {
	// 3x4 grid, iterate along axis 0 (rows)
	// Should give 4 lines (one per column j=0,1,2,3)
	shape := NewShape2D(3, 4)
	it := NewLineIterator(shape, 0)

	if it.NumLines() != 4 {
		t.Errorf("NumLines() = %d, want 4", it.NumLines())
	}

	if it.LineLength() != 3 {
		t.Errorf("LineLength() = %d, want 3", it.LineLength())
	}

	// First line starts at index 0
	if it.StartIndex() != 0 {
		t.Errorf("First StartIndex() = %d, want 0", it.StartIndex())
	}

	// Collect all start indices
	starts := []int{it.StartIndex()}
	for it.Next() {
		starts = append(starts, it.StartIndex())
	}

	// Should be [0, 1, 2, 3] - starting indices for each column
	expected := []int{0, 1, 2, 3}
	if len(starts) != len(expected) {
		t.Fatalf("got %d lines, want %d", len(starts), len(expected))
	}

	for i, v := range expected {
		if starts[i] != v {
			t.Errorf("starts[%d] = %d, want %d", i, starts[i], v)
		}
	}
}

func TestLineIterator_2D_Axis1(t *testing.T) {
	// 3x4 grid, iterate along axis 1 (columns)
	// Should give 3 lines (one per row i=0,1,2)
	shape := NewShape2D(3, 4)
	it := NewLineIterator(shape, 1)

	if it.NumLines() != 3 {
		t.Errorf("NumLines() = %d, want 3", it.NumLines())
	}

	if it.LineLength() != 4 {
		t.Errorf("LineLength() = %d, want 4", it.LineLength())
	}

	// Collect all start indices
	starts := []int{it.StartIndex()}
	for it.Next() {
		starts = append(starts, it.StartIndex())
	}

	// Should be [0, 4, 8] - starting indices for each row
	expected := []int{0, 4, 8}
	if len(starts) != len(expected) {
		t.Fatalf("got %d lines, want %d", len(starts), len(expected))
	}

	for i, v := range expected {
		if starts[i] != v {
			t.Errorf("starts[%d] = %d, want %d", i, starts[i], v)
		}
	}
}

func TestIndexRoundTrip2D(t *testing.T) {
	nx, ny := 7, 11
	for i := range nx {
		for j := range ny {
			idx := Index2D(i, j, ny)

			gotI, gotJ := FromIndex2D(idx, ny)
			if gotI != i || gotJ != j {
				t.Errorf("roundtrip failed: (%d,%d) -> %d -> (%d,%d)", i, j, idx, gotI, gotJ)
			}
		}
	}
}

func TestIndexRoundTrip3D(t *testing.T) {
	shape := NewShape3D(5, 7, 11)
	for i := range shape[0] {
		for j := range shape[1] {
			for k := range shape[2] {
				idx := Index3D(i, j, k, shape)

				gotI, gotJ, gotK := FromIndex3D(idx, shape)
				if gotI != i || gotJ != j || gotK != k {
					t.Errorf("roundtrip failed: (%d,%d,%d) -> %d -> (%d,%d,%d)",
						i, j, k, idx, gotI, gotJ, gotK)
				}
			}
		}
	}
}
