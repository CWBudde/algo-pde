package plate

import (
	"math"

	"github.com/cwbudde/algo-pde/eigen"
)

type System struct {
	Stiffness     *eigen.SparseSymmetric
	Mass          *eigen.SparseSymmetric
	LossStiffness *eigen.SparseSymmetric
	FreeDOFs      []int
	globalToFree  []int
	areaWeights   []float64
}

func Assemble(model *Model) (*System, error) {
	if err := model.Validate(); err != nil {
		return nil, err
	}
	nGlobal := 3 * len(model.Mesh.Nodes)
	constrained := make([]bool, nGlobal)
	for _, node := range model.Boundary.Clamped {
		constrained[3*node], constrained[3*node+1], constrained[3*node+2] = true, true, true
	}
	for _, node := range model.Boundary.SimplySupported {
		constrained[3*node] = true
	}
	globalToFree := make([]int, nGlobal)
	for i := range globalToFree {
		globalToFree[i] = -1
	}
	var free []int
	for dof := range nGlobal {
		if !constrained[dof] {
			globalToFree[dof] = len(free)
			free = append(free, dof)
		}
	}
	kb := eigen.NewSymmetricBuilder(len(free))
	mb := eigen.NewSymmetricBuilder(len(free))
	lb := eigen.NewSymmetricBuilder(len(free))
	areaWeights := make([]float64, len(model.Mesh.Nodes))

	for triangleIndex, tri := range model.Mesh.Triangles {
		material := model.materialForTriangle(triangleIndex)
		var nodes [3]Node
		for i := range 3 {
			nodes[i] = model.Mesh.Nodes[tri[i]]
		}
		ke, me := plateElement(nodes, material)
		assembleElement(kb, ke, triDOFs(tri), globalToFree, 1)
		assembleElement(mb, me, triDOFs(tri), globalToFree, 1)
		assembleElement(lb, ke, triDOFs(tri), globalToFree, material.LossFactor)
		area := triangleArea(nodes[0], nodes[1], nodes[2])
		for _, node := range tri {
			areaWeights[node] += area / 3
		}
	}
	for _, rib := range model.Ribs {
		ke, me := ribElement(model.Mesh.Nodes[rib.NodeA], model.Mesh.Nodes[rib.NodeB], rib)
		dofs := []int{3 * rib.NodeA, 3*rib.NodeA + 1, 3*rib.NodeA + 2, 3 * rib.NodeB, 3*rib.NodeB + 1, 3*rib.NodeB + 2}
		assembleElement(kb, ke, dofs, globalToFree, 1)
		assembleElement(mb, me, dofs, globalToFree, 1)
		assembleElement(lb, ke, dofs, globalToFree, rib.LossFactor)
	}
	return &System{Stiffness: kb.Build(), Mass: mb.Build(), LossStiffness: lb.Build(), FreeDOFs: free, globalToFree: globalToFree, areaWeights: areaWeights}, nil
}

func triDOFs(tri Triangle) []int {
	dofs := make([]int, 0, 9)
	for _, node := range tri {
		dofs = append(dofs, 3*node, 3*node+1, 3*node+2)
	}
	return dofs
}

func assembleElement(builder *eigen.SymmetricBuilder, element [][]float64, dofs, mapping []int, factor float64) {
	if factor == 0 {
		return
	}
	for i, globalI := range dofs {
		fi := mapping[globalI]
		if fi < 0 {
			continue
		}
		for j := i; j < len(dofs); j++ {
			fj := mapping[dofs[j]]
			if fj >= 0 {
				builder.Add(fi, fj, factor*element[i][j])
			}
		}
	}
}

func plateElement(nodes [3]Node, material OrthotropicMaterial) ([][]float64, [][]float64) {
	twiceArea := (nodes[1].X-nodes[0].X)*(nodes[2].Y-nodes[0].Y) - (nodes[2].X-nodes[0].X)*(nodes[1].Y-nodes[0].Y)
	area := math.Abs(twiceArea) / 2
	dndx := [3]float64{(nodes[1].Y - nodes[2].Y) / twiceArea, (nodes[2].Y - nodes[0].Y) / twiceArea, (nodes[0].Y - nodes[1].Y) / twiceArea}
	dndy := [3]float64{(nodes[2].X - nodes[1].X) / twiceArea, (nodes[0].X - nodes[2].X) / twiceArea, (nodes[1].X - nodes[0].X) / twiceArea}
	d := bendingMatrix(material)
	shear := shearMatrix(material)
	bendingB := make([][]float64, 3)
	for i := range bendingB {
		bendingB[i] = make([]float64, 9)
	}
	shearB := make([][]float64, 2)
	for i := range shearB {
		shearB[i] = make([]float64, 9)
	}
	for i := range 3 {
		base := 3 * i
		bendingB[0][base+1] = dndx[i]
		bendingB[1][base+2] = dndy[i]
		bendingB[2][base+1] = dndy[i]
		bendingB[2][base+2] = dndx[i]
		shearB[0][base] = dndx[i]
		shearB[0][base+1] = -1.0 / 3
		shearB[1][base] = dndy[i]
		shearB[1][base+2] = -1.0 / 3
	}
	ke := zeroMatrix(9)
	addBtDB(ke, bendingB, d, area)
	addBtDB(ke, shearB, shear, area)
	addShearStabilization(ke, shear, area, material.Thickness)
	me := zeroMatrix(9)
	translational := material.Density * material.Thickness * area / 12
	rotary := material.Density * material.Thickness * material.Thickness * material.Thickness * area / 144
	for a := range 3 {
		for b := range 3 {
			coefficient := 1.0
			if a == b {
				coefficient = 2
			}
			me[3*a][3*b] += coefficient * translational
			me[3*a+1][3*b+1] += coefficient * rotary
			me[3*a+2][3*b+2] += coefficient * rotary
		}
	}
	return ke, me
}

// addShearStabilization restores the part of the shear energy that the
// one-point (centroid) shear term drops. Without it the three bending rows and
// the two centroid shear rows span only rank five of the nine element degrees
// of freedom, so a twisting rotation field combined with a matching linear w
// deflection costs no energy at all; that spurious mode shows up as a
// non-positive structural eigenvalue on coarse meshes.
//
// With linear w and rotations the shear strain is gamma(x) = grad(w) - theta(x)
// and its deviation from the element mean is -sum_i (N_i - 1/3) * theta_i, which
// involves the rotation degrees of freedom only. Exact integration gives
//
//	int (N_i - 1/3)(N_j - 1/3) dA = A/18 for i == j and -A/36 otherwise.
//
// Adding that with weight one would be full integration and would reintroduce
// shear locking, so it is scaled by t^2/(t^2+A): vanishing for thin plates on
// coarse meshes where locking is the danger, approaching full integration once
// the element is refined to the plate thickness. The mode is penalized either
// way, only its stiffness is kept small where it has to be.
func addShearStabilization(ke, shear [][]float64, area, thickness float64) {
	weight := thickness * thickness / (thickness*thickness + area)
	for i := range 3 {
		for j := range 3 {
			coefficient := -area / 36
			if i == j {
				coefficient = area / 18
			}
			for p := range 2 {
				for q := range 2 {
					ke[3*i+1+p][3*j+1+q] += weight * coefficient * shear[p][q]
				}
			}
		}
	}
}

func bendingMatrix(m OrthotropicMaterial) [][]float64 {
	nu21 := m.Poisson12 * m.Young2 / m.Young1
	denom := 1 - m.Poisson12*nu21
	q11, q22 := m.Young1/denom, m.Young2/denom
	q12, q66 := m.Poisson12*m.Young2/denom, m.Shear12
	c, s := math.Cos(m.GrainAngleDeg*math.Pi/180), math.Sin(m.GrainAngleDeg*math.Pi/180)
	c2, s2 := c*c, s*s
	c4, s4 := c2*c2, s2*s2
	cs2 := c2 * s2
	qb11 := q11*c4 + 2*(q12+2*q66)*cs2 + q22*s4
	qb22 := q11*s4 + 2*(q12+2*q66)*cs2 + q22*c4
	qb12 := (q11+q22-4*q66)*cs2 + q12*(c4+s4)
	qb16 := (q11-q12-2*q66)*c*c*c*s - (q22-q12-2*q66)*c*s*s*s
	qb26 := (q11-q12-2*q66)*c*s*s*s - (q22-q12-2*q66)*c*c*c*s
	qb66 := (q11+q22-2*q12-2*q66)*cs2 + q66*(c4+s4)
	factor := m.Thickness * m.Thickness * m.Thickness / 12
	return [][]float64{{factor * qb11, factor * qb12, factor * qb16}, {factor * qb12, factor * qb22, factor * qb26}, {factor * qb16, factor * qb26, factor * qb66}}
}

func shearMatrix(m OrthotropicMaterial) [][]float64 {
	g13, g23 := m.Shear13, m.Shear23
	if g13 == 0 {
		g13 = m.Shear12
	}
	if g23 == 0 {
		g23 = m.Shear12
	}
	c, s := math.Cos(m.GrainAngleDeg*math.Pi/180), math.Sin(m.GrainAngleDeg*math.Pi/180)
	factor := m.Shearkappa() * m.Thickness
	return [][]float64{{factor * (g13*c*c + g23*s*s), factor * (g13 - g23) * c * s}, {factor * (g13 - g23) * c * s, factor * (g13*s*s + g23*c*c)}}
}

func ribElement(a, b Node, rib Rib) ([][]float64, [][]float64) {
	dx, dy := b.X-a.X, b.Y-a.Y
	length := math.Hypot(dx, dy)
	tx, ty := dx/length, dy/length
	area, inertia := rib.Width*rib.Height, rib.Width*rib.Height*rib.Height*rib.Height/12
	ke, me := zeroMatrix(6), zeroMatrix(6)
	// Euler-Bernoulli curvature is the line derivative of the plate slope in
	// the rib direction. The plate shear term couples that slope to w.
	curvature := []float64{0, -tx / length, -ty / length, 0, tx / length, ty / length}
	for i := range 6 {
		for j := range 6 {
			ke[i][j] += rib.YoungModulus * inertia * length * curvature[i] * curvature[j]
		}
	}
	translational := rib.Density * area * length / 6
	me[0][0], me[0][3], me[3][0], me[3][3] = 2*translational, translational, translational, 2*translational
	rotary := rib.Density * inertia * length / 6
	for aNode := range 2 {
		for bNode := range 2 {
			coefficient := 1.0
			if aNode == bNode {
				coefficient = 2
			}
			for p, tp := range []float64{tx, ty} {
				for q, tq := range []float64{tx, ty} {
					me[3*aNode+1+p][3*bNode+1+q] += coefficient * rotary * tp * tq
				}
			}
		}
	}
	return ke, me
}

func addBtDB(dst, b, d [][]float64, factor float64) {
	for i := range dst {
		for j := i; j < len(dst); j++ {
			s := 0.0
			for p := range b {
				for q := range b {
					s += b[p][i] * d[p][q] * b[q][j]
				}
			}
			dst[i][j] += factor * s
			dst[j][i] = dst[i][j]
		}
	}
}

func zeroMatrix(n int) [][]float64 {
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n)
	}
	return m
}
