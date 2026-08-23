package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/algo-pde/plate"
)

func commandModel() plate.Model {
	const n = 4
	nodes := make([]plate.Node, 0, n*n)
	for y := range n {
		for x := range n {
			nodes = append(nodes, plate.Node{X: float64(x) / (n - 1), Y: 0.8 * float64(y) / (n - 1)})
		}
	}
	var triangles []plate.Triangle
	for y := 0; y+1 < n; y++ {
		for x := 0; x+1 < n; x++ {
			a, b, c, d := y*n+x, y*n+x+1, (y+1)*n+x, (y+1)*n+x+1
			triangles = append(triangles, plate.Triangle{a, b, d}, plate.Triangle{a, d, c})
		}
	}
	var boundary []int
	for y := range n {
		for x := range n {
			if x == 0 || y == 0 || x == n-1 || y == n-1 {
				boundary = append(boundary, y*n+x)
			}
		}
	}
	return plate.Model{
		Mesh:     plate.Mesh{Nodes: nodes, Triangles: triangles},
		Material: plate.OrthotropicMaterial{Young1: 10e9, Young2: 1e9, Shear12: 0.6e9, Poisson12: 0.3, Density: 450, Thickness: 0.01, LossFactor: 0.02},
		Boundary: plate.Boundary{Clamped: boundary},
		Source:   plate.BridgeSource{ID: "test-bridge", Nodes: []plate.WeightedNode{{Node: 5, Weight: 1}}},
	}
}

func TestRunWritesStrictArtifactAndReusesIt(t *testing.T) {
	directory := t.TempDir()
	modelPath, outputPath := filepath.Join(directory, "model.json"), filepath.Join(directory, "transfer.json")
	data, err := json.Marshal(commandModel())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(modelPath, data, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	args := []string{"-model", modelPath, "-out", outputPath, "-modes", "2", "-cover-frequency", "0", "-max-iterations", "400", "-tolerance", "1e-7"}
	if err := run(args, &stdout, &stderr); err != nil {
		t.Fatalf("first run: %v; stderr=%s", err, stderr.String())
	}
	artifact, err := readArtifact(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(artifact.Modes) != 2 {
		t.Fatalf("mode count = %d, want 2", len(artifact.Modes))
	}
	encoded, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &top); err != nil {
		t.Fatal(err)
	}
	if len(top) != 7 {
		t.Fatalf("top-level fields = %d, want exactly 7: %s", len(top), encoded)
	}
	stdout.Reset()
	if err := run(args, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stdout.String(), "reused") {
		t.Fatalf("second run did not reuse artifact: %s", stdout.String())
	}
	stdout.Reset()
	changed := append(append([]string(nil), args...), "-seed", "99")
	if err := run(changed, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(stdout.String(), "reused") {
		t.Fatalf("changed solver options incorrectly reused cache: %s", stdout.String())
	}
	if _, err := os.Stat(outputPath + ".cache.json"); err != nil {
		t.Fatalf("cache sidecar: %v", err)
	}
}

func TestReadModelRejectsUnknownField(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.json")
	if err := os.WriteFile(path, []byte(`{"unexpected":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readModel(path); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("error = %v, want unknown field", err)
	}
}

func TestRunRejectsNonFiniteSolverOptionBeforeWriting(t *testing.T) {
	directory := t.TempDir()
	outputPath := filepath.Join(directory, "transfer.json")
	err := run([]string{
		"-model", filepath.Join(directory, "does-not-need-to-exist.json"),
		"-out", outputPath,
		"-tolerance", "+Inf",
	}, &bytes.Buffer{}, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tolerance must be finite") {
		t.Fatalf("error = %v, want non-finite tolerance rejection", err)
	}
	if _, statErr := os.Stat(outputPath); !os.IsNotExist(statErr) {
		t.Fatalf("output was written for invalid options: %v", statErr)
	}
}
