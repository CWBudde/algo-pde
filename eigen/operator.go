package eigen

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Operator is a real square linear operator. MulVec must support distinct dst
// and src slices of length Dim().
type Operator interface {
	Dim() int
	MulVec(dst, src []float64)
}

// Preconditioner approximately applies the inverse of an operator.
type Preconditioner interface {
	Apply(dst, src []float64)
}

// Identity is an identity operator of dimension N.
type Identity struct{ N int }

func (i Identity) Dim() int                  { return i.N }
func (i Identity) MulVec(dst, src []float64) { copy(dst, src) }

type entry struct {
	col int
	val float64
}

// SparseSymmetric stores both triangles of an assembled symmetric matrix in
// compressed row form. Construct one with SymmetricBuilder.
type SparseSymmetric struct {
	n      int
	rowPtr []int
	cols   []int
	values []float64
}

func (a *SparseSymmetric) Dim() int { return a.n }

// Nonzeros reports the number of stored entries, counting both triangles.
func (a *SparseSymmetric) Nonzeros() int { return len(a.values) }

// MulVec multiplies dst = A src.
func (a *SparseSymmetric) MulVec(dst, src []float64) {
	if len(dst) != a.n || len(src) != a.n {
		panic("eigen: matrix-vector length mismatch")
	}
	for i := range dst {
		sum := 0.0
		for p := a.rowPtr[i]; p < a.rowPtr[i+1]; p++ {
			sum += a.values[p] * src[a.cols[p]]
		}
		dst[i] = sum
	}
}

// Diagonal returns a copy of the matrix diagonal.
func (a *SparseSymmetric) Diagonal() []float64 {
	d := make([]float64, a.n)
	for i := range d {
		for p := a.rowPtr[i]; p < a.rowPtr[i+1]; p++ {
			if a.cols[p] == i {
				d[i] = a.values[p]
				break
			}
		}
	}
	return d
}

// At returns A(i,j).
func (a *SparseSymmetric) At(i, j int) float64 {
	for p := a.rowPtr[i]; p < a.rowPtr[i+1]; p++ {
		if a.cols[p] == j {
			return a.values[p]
		}
	}
	return 0
}

// SymmetricBuilder accumulates the upper triangle of a sparse symmetric
// matrix. Repeated Add calls to the same entry are summed.
type SymmetricBuilder struct {
	n    int
	rows []map[int]float64
}

func NewSymmetricBuilder(n int) *SymmetricBuilder {
	rows := make([]map[int]float64, n)
	for i := range rows {
		rows[i] = make(map[int]float64)
	}
	return &SymmetricBuilder{n: n, rows: rows}
}

func (b *SymmetricBuilder) Add(i, j int, value float64) {
	if i < 0 || j < 0 || i >= b.n || j >= b.n {
		panic("eigen: builder index out of range")
	}
	if i > j {
		i, j = j, i
	}
	b.rows[i][j] += value
}

func (b *SymmetricBuilder) Build() *SparseSymmetric {
	full := make([][]entry, b.n)
	for i, row := range b.rows {
		for j, value := range row {
			if value == 0 {
				continue
			}
			full[i] = append(full[i], entry{j, value})
			if i != j {
				full[j] = append(full[j], entry{i, value})
			}
		}
	}
	a := &SparseSymmetric{n: b.n, rowPtr: make([]int, b.n+1)}
	for i := range full {
		sort.Slice(full[i], func(p, q int) bool { return full[i][p].col < full[i][q].col })
		a.rowPtr[i] = len(a.cols)
		for _, e := range full[i] {
			a.cols = append(a.cols, e.col)
			a.values = append(a.values, e.val)
		}
	}
	a.rowPtr[b.n] = len(a.cols)
	return a
}

type diagonalPreconditioner struct{ inv []float64 }

// NewDiagonalPreconditioner constructs a Jacobi preconditioner. Entries whose
// magnitude is at most floor use a positive floor.
func NewDiagonalPreconditioner(a *SparseSymmetric, floor float64) Preconditioner {
	if floor <= 0 {
		floor = 1e-14
	}
	d := a.Diagonal()
	for i, v := range d {
		if math.Abs(v) < floor {
			v = floor
		}
		d[i] = 1 / v
	}
	return &diagonalPreconditioner{inv: d}
}

func (p *diagonalPreconditioner) Apply(dst, src []float64) {
	for i := range dst {
		dst[i] = p.inv[i] * src[i]
	}
}

type factorEntry struct {
	index int
	value float64
}

type ic0Preconditioner struct {
	diag []float64
	rows [][]factorEntry
	cols [][]factorEntry
}

// NewIC0Preconditioner constructs a zero-fill incomplete Cholesky factor of
// A+shift*I. It returns an error if a positive pivot cannot be formed.
func NewIC0Preconditioner(a *SparseSymmetric, shift float64) (Preconditioner, error) {
	if shift < 0 || math.IsNaN(shift) || math.IsInf(shift, 0) {
		return nil, errors.New("eigen: IC(0) shift must be finite and non-negative")
	}
	n := a.n
	p := &ic0Preconditioner{diag: make([]float64, n), rows: make([][]factorEntry, n), cols: make([][]factorEntry, n)}
	for i := 0; i < n; i++ {
		for q := a.rowPtr[i]; q < a.rowPtr[i+1]; q++ {
			j := a.cols[q]
			if j >= i {
				break
			}
			s := a.values[q]
			for _, ik := range p.rows[i] {
				if ik.index >= j {
					break
				}
				for _, jk := range p.rows[j] {
					if jk.index == ik.index {
						s -= ik.value * jk.value
						break
					}
					if jk.index > ik.index {
						break
					}
				}
			}
			v := s / p.diag[j]
			p.rows[i] = append(p.rows[i], factorEntry{j, v})
			p.cols[j] = append(p.cols[j], factorEntry{i, v})
		}
		pivot := a.At(i, i) + shift
		for _, e := range p.rows[i] {
			pivot -= e.value * e.value
		}
		if !(pivot > 0) || math.IsInf(pivot, 0) {
			return nil, fmt.Errorf("eigen: IC(0) non-positive pivot at row %d", i)
		}
		p.diag[i] = math.Sqrt(pivot)
	}
	return p, nil
}

func (p *ic0Preconditioner) Apply(dst, src []float64) {
	for i := range dst {
		s := src[i]
		for _, e := range p.rows[i] {
			s -= e.value * dst[e.index]
		}
		dst[i] = s / p.diag[i]
	}
	for i := len(dst) - 1; i >= 0; i-- {
		s := dst[i]
		for _, e := range p.cols[i] {
			s -= e.value * dst[e.index]
		}
		dst[i] = s / p.diag[i]
	}
}
