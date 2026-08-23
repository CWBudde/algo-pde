package plate

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
)

const (
	SchemaVersion = 1
	TransferKind  = "bridge_force_to_area_velocity"
)

type TransferMode struct {
	FrequencyHz float64 `json:"frequency_hz"`
	LossFactor  float64 `json:"loss_factor"`
	Residue     float64 `json:"residue"`
}

// ModalTransfer is the strict algo-piano interchange contract. Keep this wire
// shape intentionally small; solver settings and cache metadata do not belong
// in the transfer function.
type ModalTransfer struct {
	SchemaVersion int            `json:"schema_version"`
	TransferKind  string         `json:"transfer_kind"`
	ModelSHA256   string         `json:"model_sha256"`
	InputUnit     string         `json:"input_unit"`
	OutputUnit    string         `json:"output_unit"`
	SourceID      string         `json:"source_id"`
	Modes         []TransferMode `json:"modes"`
}

func CanonicalModelHash(model *Model) (string, error) {
	if err := model.Validate(); err != nil {
		return "", err
	}
	encoded, err := json.Marshal(model)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(encoded)
	return hex.EncodeToString(hash[:]), nil
}

// ProjectTransfer projects a unit distributed bridge impulse to
// area-averaged transverse velocity. Modes within relativeFrequencyTolerance
// are collapsed: summed residues and trace-averaged damping make exactly
// degenerate eigenspaces invariant to eigenvector rotations.
func ProjectTransfer(model *Model, result *ModalResult, relativeFrequencyTolerance float64) (*ModalTransfer, error) {
	if result == nil || result.System == nil || len(result.Modes) == 0 {
		return nil, errors.New("plate: empty modal result")
	}
	if relativeFrequencyTolerance == 0 {
		relativeFrequencyTolerance = 1e-7
	}
	hash, err := CanonicalModelHash(model)
	if err != nil {
		return nil, err
	}
	totalArea := 0.0
	for _, weight := range result.System.areaWeights {
		totalArea += weight
	}
	force := make([]float64, 3*len(model.Mesh.Nodes))
	for _, node := range model.Source.Nodes {
		force[3*node.Node] += node.Weight
	}
	projected := make([]TransferMode, len(result.Modes))
	for i, mode := range result.Modes {
		input, output := 0.0, 0.0
		for node := range model.Mesh.Nodes {
			input += mode.Shape[3*node] * force[3*node]
			output += mode.Shape[3*node] * result.System.areaWeights[node] / totalArea
		}
		projected[i] = TransferMode{FrequencyHz: mode.FrequencyHz, LossFactor: mode.LossFactor, Residue: input * output}
	}
	collapsed := make([]TransferMode, 0, len(projected))
	for start := 0; start < len(projected); {
		end := start + 1
		for end < len(projected) {
			scale := math.Max(projected[start].FrequencyHz, projected[end].FrequencyHz)
			if math.Abs(projected[end].FrequencyHz-projected[start].FrequencyHz) > relativeFrequencyTolerance*scale {
				break
			}
			end++
		}
		cluster := TransferMode{}
		for _, mode := range projected[start:end] {
			cluster.FrequencyHz += mode.FrequencyHz
			cluster.LossFactor += mode.LossFactor
			cluster.Residue += mode.Residue
		}
		count := float64(end - start)
		cluster.FrequencyHz /= count
		cluster.LossFactor /= count
		collapsed = append(collapsed, cluster)
		start = end
	}
	return &ModalTransfer{SchemaVersion: SchemaVersion, TransferKind: TransferKind, ModelSHA256: hash, InputUnit: "N*s", OutputUnit: "m/s", SourceID: model.Source.ID, Modes: collapsed}, nil
}
