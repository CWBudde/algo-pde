package plate

import (
	"encoding/json"
	"fmt"
	"math"
	"testing"
)

func rectangularModel(nx, ny int) *Model {
	nodes := make([]Node, 0, nx*ny)
	for y := range ny {
		for x := range nx {
			nodes = append(nodes, Node{X: float64(x) / float64(nx-1), Y: 0.7 * float64(y) / float64(ny-1)})
		}
	}
	triangles := make([]Triangle, 0, 2*(nx-1)*(ny-1))
	for y := 0; y+1 < ny; y++ {
		for x := 0; x+1 < nx; x++ {
			a, b := y*nx+x, y*nx+x+1
			c, d := (y+1)*nx+x, (y+1)*nx+x+1
			triangles = append(triangles, Triangle{a, b, d}, Triangle{a, d, c})
		}
	}
	boundary := make([]int, 0, 2*nx+2*ny-4)
	for y := range ny {
		for x := range nx {
			if x == 0 || x == nx-1 || y == 0 || y == ny-1 {
				boundary = append(boundary, y*nx+x)
			}
		}
	}
	center := (ny/2)*nx + nx/2
	return &Model{
		Mesh: Mesh{Nodes: nodes, Triangles: triangles},
		Material: OrthotropicMaterial{
			Young1: 11e9, Young2: 0.8e9, Shear12: 0.65e9, Poisson12: 0.35,
			Shear13: 0.65e9, Shear23: 0.3e9, Density: 450, Thickness: 0.008, LossFactor: 0.02,
		},
		Boundary: Boundary{Clamped: boundary},
		Source:   BridgeSource{ID: "bridge", Nodes: []WeightedNode{{Node: center, Weight: 1}}},
	}
}

func TestAssemblePositiveEnergy(t *testing.T) {
	model := rectangularModel(5, 5)
	system, err := Assemble(model)
	if err != nil {
		t.Fatal(err)
	}
	if system.Stiffness.Nonzeros() == 0 || system.Mass.Nonzeros() == 0 {
		t.Fatal("empty assembled matrices")
	}
	v := make([]float64, len(system.FreeDOFs))
	for i := range v {
		v[i] = math.Sin(float64(i+1) * 0.73)
	}
	if energy := quadratic(system.Stiffness, v); !(energy > 0) {
		t.Fatalf("stiffness energy = %g", energy)
	}
	if energy := quadratic(system.Mass, v); !(energy > 0) {
		t.Fatalf("mass energy = %g", energy)
	}
}

func TestRibAddsStiffnessAndMass(t *testing.T) {
	model := rectangularModel(5, 5)
	plain, err := Assemble(model)
	if err != nil {
		t.Fatal(err)
	}
	model.Ribs = []Rib{{NodeA: 11, NodeB: 13, YoungModulus: 10e9, Density: 500, Width: 0.02, Height: 0.04, LossFactor: 0.03}}
	stiffened, err := Assemble(model)
	if err != nil {
		t.Fatal(err)
	}
	v := make([]float64, len(plain.FreeDOFs))
	for i := range v {
		v[i] = math.Sin(float64(i+1) * 0.31)
	}
	if quadratic(stiffened.Stiffness, v) <= quadratic(plain.Stiffness, v) {
		t.Fatal("rib did not add stiffness energy")
	}
	if quadratic(stiffened.Mass, v) <= quadratic(plain.Mass, v) {
		t.Fatal("rib did not add mass energy")
	}
}

func TestSolveModesAndMeshRefinement(t *testing.T) {
	frequencies := make([]float64, 0, 3)
	for _, size := range []int{4, 5, 7} {
		result, err := Solve(rectangularModel(size, size), SolveOptions{ModeCount: 2, Tolerance: 1e-7, MaxIterations: 400, Seed: 3})
		if err != nil {
			t.Fatalf("mesh %d: %v", size, err)
		}
		if result.Modes[0].FrequencyHz <= 0 || result.Modes[0].LossFactor <= 0 {
			t.Fatalf("mesh %d: invalid first mode %+v", size, result.Modes[0])
		}
		if result.Modes[1].FrequencyHz < result.Modes[0].FrequencyHz {
			t.Fatal("modes are not sorted")
		}
		frequencies = append(frequencies, result.Modes[0].FrequencyHz)
	}
	// Reduced shear integration should keep this thin-plate refinement sequence
	// in one physical frequency scale rather than exhibiting catastrophic shear
	// locking. This is deliberately a broad regression bound, not an accuracy
	// claim for the low-order element.
	minFrequency, maxFrequency := frequencies[0], frequencies[0]
	for _, frequency := range frequencies[1:] {
		minFrequency = math.Min(minFrequency, frequency)
		maxFrequency = math.Max(maxFrequency, frequency)
	}
	if maxFrequency/minFrequency > 3 {
		t.Fatalf("first-mode refinement frequencies %v indicate shear locking", frequencies)
	}
}

func TestProjectTransferContractAndDegenerateInvariance(t *testing.T) {
	model := rectangularModel(3, 3)
	areaWeights := make([]float64, len(model.Mesh.Nodes))
	for i := range areaWeights {
		areaWeights[i] = 1
	}
	system := &System{areaWeights: areaWeights}
	v1, v2 := make([]float64, 3*len(model.Mesh.Nodes)), make([]float64, 3*len(model.Mesh.Nodes))
	v1[3*4], v1[3*1] = 1, 0.3
	v2[3*4], v2[3*7] = -0.2, 0.7
	makeResult := func(a, b []float64) *ModalResult {
		return &ModalResult{System: system, Modes: []Mode{{FrequencyHz: 100, LossFactor: 0.02, Shape: a}, {FrequencyHz: 100, LossFactor: 0.02, Shape: b}}}
	}
	first, err := ProjectTransfer(model, makeResult(v1, v2), 0)
	if err != nil {
		t.Fatal(err)
	}
	angle := 0.71
	c, s := math.Cos(angle), math.Sin(angle)
	r1, r2 := make([]float64, len(v1)), make([]float64, len(v1))
	for i := range v1 {
		r1[i] = c*v1[i] + s*v2[i]
		r2[i] = -s*v1[i] + c*v2[i]
	}
	rotated, err := ProjectTransfer(model, makeResult(r1, r2), 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Modes) != 1 || len(rotated.Modes) != 1 || math.Abs(first.Modes[0].Residue-rotated.Modes[0].Residue) > 1e-14 {
		t.Fatalf("degenerate residue changed: %#v vs %#v", first.Modes, rotated.Modes)
	}
	encoded, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &top); err != nil {
		t.Fatal(err)
	}
	wantKeys := []string{"schema_version", "transfer_kind", "model_sha256", "input_unit", "output_unit", "source_id", "modes"}
	if len(top) != len(wantKeys) {
		t.Fatalf("artifact has extra fields: %s", encoded)
	}
	for _, key := range wantKeys {
		if _, ok := top[key]; !ok {
			t.Fatalf("artifact missing %q: %s", key, encoded)
		}
	}
	if len(first.ModelSHA256) != 64 || first.InputUnit != "N*s" || first.OutputUnit != "m/s" || first.TransferKind != TransferKind {
		t.Fatalf("invalid contract constants: %+v", first)
	}
}

func BenchmarkPlateSolveRefinement(b *testing.B) {
	for _, size := range []int{5, 7, 9} {
		model := rectangularModel(size, size)
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			frequency := 0.0
			for range b.N {
				result, err := Solve(model, SolveOptions{ModeCount: 2, Tolerance: 1e-6, MaxIterations: 400, Seed: 1})
				if err != nil {
					b.Fatal(err)
				}
				frequency = result.Modes[0].FrequencyHz
			}
			b.ReportMetric(frequency, "first-mode-Hz")
		})
	}
}
