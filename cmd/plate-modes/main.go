// Command plate-modes solves an offline structural plate model and writes the
// strict body-modal-transfer-v1 JSON artifact consumed by algo-piano.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/cwbudde/algo-pde/plate"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("plate-modes", flag.ContinueOnError)
	flags.SetOutput(stderr)
	modelPath := flags.String("model", "", "plate model JSON input")
	outputPath := flags.String("out", "", "body-modal-transfer-v1 JSON output")
	modeCount := flags.Int("modes", 64, "number of lowest structural modes to solve")
	tolerance := flags.Float64("tolerance", 1e-8, "relative eigensolver residual tolerance")
	maxIterations := flags.Int("max-iterations", 500, "maximum block iterations")
	seed := flags.Uint64("seed", 1, "deterministic initial-vector seed")
	icShift := flags.Float64("ic-shift", 0, "non-negative IC(0) diagonal shift")
	coverFrequency := flags.Float64("cover-frequency", 5000, "require the highest mode to reach this frequency in Hz (0 disables)")
	force := flags.Bool("force", false, "recompute instead of reusing a matching output artifact")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" || *outputPath == "" {
		return errors.New("plate-modes: -model and -out are required")
	}
	if *coverFrequency < 0 || math.IsNaN(*coverFrequency) || math.IsInf(*coverFrequency, 0) {
		return errors.New("plate-modes: -cover-frequency must be finite and non-negative")
	}
	model, err := readModel(*modelPath)
	if err != nil {
		return err
	}
	modelHash, err := plate.CanonicalModelHash(model)
	if err != nil {
		return err
	}
	if !*force {
		if cached, cacheErr := readArtifact(*outputPath); cacheErr == nil && cacheMatches(cached, modelHash, model.Source.ID, *modeCount, *coverFrequency) {
			fmt.Fprintf(stdout, "reused %s (%d transfer modes)\n", *outputPath, len(cached.Modes))
			return nil
		}
	}
	result, err := plate.Solve(model, plate.SolveOptions{
		ModeCount: *modeCount, Tolerance: *tolerance, MaxIterations: *maxIterations, Seed: *seed, ICShift: *icShift,
	})
	if err != nil {
		return err
	}
	artifact, err := plate.ProjectTransfer(model, result, 0)
	if err != nil {
		return err
	}
	if err := validateArtifact(artifact); err != nil {
		return err
	}
	if *coverFrequency > 0 && artifact.Modes[len(artifact.Modes)-1].FrequencyHz < *coverFrequency {
		return fmt.Errorf("plate-modes: %d requested modes reach %.3f Hz, below required %.3f Hz; increase -modes", *modeCount, artifact.Modes[len(artifact.Modes)-1].FrequencyHz, *coverFrequency)
	}
	if err := writeArtifact(*outputPath, artifact); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "wrote %s (%d transfer modes, %d iterations)\n", *outputPath, len(artifact.Modes), result.Iterations)
	return nil
}

func readModel(path string) (*plate.Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("plate-modes: read model: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var model plate.Model
	if err := decoder.Decode(&model); err != nil {
		return nil, fmt.Errorf("plate-modes: decode model: %w", err)
	}
	if err := requireEOF(decoder); err != nil {
		return nil, fmt.Errorf("plate-modes: decode model: %w", err)
	}
	if err := model.Validate(); err != nil {
		return nil, err
	}
	return &model, nil
}

func readArtifact(path string) (*plate.ModalTransfer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var artifact plate.ModalTransfer
	if err := decoder.Decode(&artifact); err != nil {
		return nil, err
	}
	if err := requireEOF(decoder); err != nil {
		return nil, err
	}
	if err := validateArtifact(&artifact); err != nil {
		return nil, err
	}
	return &artifact, nil
}

func requireEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

func validateArtifact(artifact *plate.ModalTransfer) error {
	if artifact.SchemaVersion != plate.SchemaVersion || artifact.TransferKind != plate.TransferKind ||
		artifact.InputUnit != "N*s" || artifact.OutputUnit != "m/s" || !lowerHexSHA256(artifact.ModelSHA256) || artifact.SourceID == "" || len(artifact.Modes) == 0 {
		return errors.New("plate-modes: invalid transfer contract")
	}
	lastFrequency := 0.0
	for i, mode := range artifact.Modes {
		if !(mode.FrequencyHz > lastFrequency) || math.IsInf(mode.FrequencyHz, 0) || mode.LossFactor < 0 || math.IsNaN(mode.LossFactor) || math.IsInf(mode.LossFactor, 0) || math.IsNaN(mode.Residue) || math.IsInf(mode.Residue, 0) {
			return fmt.Errorf("plate-modes: invalid transfer mode %d", i)
		}
		lastFrequency = mode.FrequencyHz
	}
	return nil
}

func lowerHexSHA256(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, character := range value {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

func cacheMatches(artifact *plate.ModalTransfer, modelHash, sourceID string, modeCount int, coverFrequency float64) bool {
	if artifact.ModelSHA256 != modelHash || artifact.SourceID != sourceID || len(artifact.Modes) < modeCount {
		return false
	}
	return coverFrequency == 0 || artifact.Modes[len(artifact.Modes)-1].FrequencyHz >= coverFrequency
}

func writeArtifact(path string, artifact *plate.ModalTransfer) error {
	directory := filepath.Dir(path)
	temporary, err := os.CreateTemp(directory, ".plate-modes-*.json")
	if err != nil {
		return fmt.Errorf("plate-modes: create output: %w", err)
	}
	temporaryPath := temporary.Name()
	keep := false
	defer func() {
		_ = temporary.Close()
		if !keep {
			_ = os.Remove(temporaryPath)
		}
	}()
	encoder := json.NewEncoder(temporary)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(artifact); err != nil {
		return fmt.Errorf("plate-modes: encode output: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		return fmt.Errorf("plate-modes: sync output: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("plate-modes: close output: %w", err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("plate-modes: replace output: %w", err)
	}
	keep = true
	return nil
}
