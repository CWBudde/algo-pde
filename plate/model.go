package plate

import (
	"errors"
	"fmt"
	"math"
)

type Node struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

type Triangle [3]int

type Mesh struct {
	Nodes     []Node     `json:"nodes"`
	Triangles []Triangle `json:"triangles"`
}

// OrthotropicMaterial describes one homogeneous plate layer. Axis 1 is
// rotated GrainAngleDeg counter-clockwise from global x.
type OrthotropicMaterial struct {
	Young1          float64 `json:"young_1_pa"`
	Young2          float64 `json:"young_2_pa"`
	Shear12         float64 `json:"shear_12_pa"`
	Poisson12       float64 `json:"poisson_12"`
	Shear13         float64 `json:"shear_13_pa,omitempty"`
	Shear23         float64 `json:"shear_23_pa,omitempty"`
	Density         float64 `json:"density_kg_m3"`
	Thickness       float64 `json:"thickness_m"`
	GrainAngleDeg   float64 `json:"grain_angle_deg,omitempty"`
	LossFactor      float64 `json:"loss_factor,omitempty"`
	ShearCorrection float64 `json:"shear_correction,omitempty"`
}

type Rib struct {
	NodeA        int     `json:"node_a"`
	NodeB        int     `json:"node_b"`
	YoungModulus float64 `json:"young_modulus_pa"`
	Density      float64 `json:"density_kg_m3"`
	Width        float64 `json:"width_m"`
	Height       float64 `json:"height_m"`
	LossFactor   float64 `json:"loss_factor,omitempty"`
}

type Boundary struct {
	Clamped         []int `json:"clamped,omitempty"`
	SimplySupported []int `json:"simply_supported,omitempty"`
}

type WeightedNode struct {
	Node   int     `json:"node"`
	Weight float64 `json:"weight"`
}

type BridgeSource struct {
	ID    string         `json:"id"`
	Nodes []WeightedNode `json:"nodes"`
}

type Model struct {
	Mesh     Mesh                `json:"mesh"`
	Material OrthotropicMaterial `json:"material"`
	Ribs     []Rib               `json:"ribs,omitempty"`
	Boundary Boundary            `json:"boundary"`
	Source   BridgeSource        `json:"source"`
}

func (m *Model) Validate() error {
	if len(m.Mesh.Nodes) < 3 || len(m.Mesh.Triangles) == 0 {
		return errors.New("plate: mesh needs at least three nodes and one triangle")
	}
	finitePositive := func(v float64) bool { return v > 0 && !math.IsInf(v, 0) && !math.IsNaN(v) }
	mat := m.Material
	if !finitePositive(mat.Young1) || !finitePositive(mat.Young2) || !finitePositive(mat.Shear12) ||
		!finitePositive(mat.Density) || !finitePositive(mat.Thickness) {
		return errors.New("plate: material moduli, density, and thickness must be finite and positive")
	}
	if mat.Shearkappa() <= 0 || math.IsNaN(mat.Shearkappa()) || math.IsInf(mat.Shearkappa(), 0) ||
		mat.Poisson12 < -0.99 || mat.Poisson12 >= 0.99 || math.IsNaN(mat.Poisson12) ||
		mat.LossFactor < 0 || math.IsNaN(mat.LossFactor) || math.IsInf(mat.LossFactor, 0) ||
		math.IsNaN(mat.GrainAngleDeg) || math.IsInf(mat.GrainAngleDeg, 0) {
		return errors.New("plate: invalid Poisson ratio, loss factor, or shear correction")
	}
	if (mat.Shear13 != 0 && !finitePositive(mat.Shear13)) || (mat.Shear23 != 0 && !finitePositive(mat.Shear23)) {
		return errors.New("plate: transverse shear moduli must be finite and positive when supplied")
	}
	if 1-mat.Poisson12*mat.Poisson12*mat.Young2/mat.Young1 <= 0 {
		return errors.New("plate: orthotropic plane-stress matrix is not positive definite")
	}
	for i, n := range m.Mesh.Nodes {
		if math.IsNaN(n.X) || math.IsNaN(n.Y) || math.IsInf(n.X, 0) || math.IsInf(n.Y, 0) {
			return fmt.Errorf("plate: node %d has non-finite coordinates", i)
		}
	}
	for i, tri := range m.Mesh.Triangles {
		for _, node := range tri {
			if node < 0 || node >= len(m.Mesh.Nodes) {
				return fmt.Errorf("plate: triangle %d has invalid node %d", i, node)
			}
		}
		if triangleArea(m.Mesh.Nodes[tri[0]], m.Mesh.Nodes[tri[1]], m.Mesh.Nodes[tri[2]]) <= 1e-16 {
			return fmt.Errorf("plate: triangle %d is degenerate", i)
		}
	}
	seen := make(map[int]string)
	for _, node := range m.Boundary.Clamped {
		if node < 0 || node >= len(m.Mesh.Nodes) {
			return fmt.Errorf("plate: invalid clamped node %d", node)
		}
		seen[node] = "clamped"
	}
	for _, node := range m.Boundary.SimplySupported {
		if node < 0 || node >= len(m.Mesh.Nodes) {
			return fmt.Errorf("plate: invalid simply-supported node %d", node)
		}
		if seen[node] == "clamped" {
			return fmt.Errorf("plate: node %d has two boundary types", node)
		}
		seen[node] = "simply-supported"
	}
	if len(seen) == 0 {
		return errors.New("plate: at least one boundary node is required")
	}
	for i, rib := range m.Ribs {
		if rib.NodeA < 0 || rib.NodeA >= len(m.Mesh.Nodes) || rib.NodeB < 0 || rib.NodeB >= len(m.Mesh.Nodes) || rib.NodeA == rib.NodeB {
			return fmt.Errorf("plate: rib %d has invalid endpoints", i)
		}
		if !finitePositive(rib.YoungModulus) || !finitePositive(rib.Density) || !finitePositive(rib.Width) || !finitePositive(rib.Height) ||
			rib.LossFactor < 0 || math.IsNaN(rib.LossFactor) || math.IsInf(rib.LossFactor, 0) {
			return fmt.Errorf("plate: rib %d has invalid properties", i)
		}
	}
	if m.Source.ID == "" || len(m.Source.Nodes) == 0 {
		return errors.New("plate: bridge source needs an id and weighted nodes")
	}
	sum := 0.0
	for _, sourceNode := range m.Source.Nodes {
		if sourceNode.Node < 0 || sourceNode.Node >= len(m.Mesh.Nodes) || math.IsNaN(sourceNode.Weight) || math.IsInf(sourceNode.Weight, 0) {
			return errors.New("plate: bridge source has an invalid weighted node")
		}
		sum += sourceNode.Weight
	}
	if math.Abs(sum-1) > 1e-9 {
		return fmt.Errorf("plate: bridge source weights sum to %g, want 1", sum)
	}
	return nil
}

func (m OrthotropicMaterial) Shearkappa() float64 {
	if m.ShearCorrection == 0 {
		return 5.0 / 6.0
	}
	return m.ShearCorrection
}

func triangleArea(a, b, c Node) float64 {
	return math.Abs((b.X-a.X)*(c.Y-a.Y)-(c.X-a.X)*(b.Y-a.Y)) / 2
}
