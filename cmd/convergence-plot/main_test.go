package main

import (
	"math"
	"strings"
	"testing"
)

// observedOrder is the log-log fit of error against spacing across the extremes.
func observedOrder(hs, errs []float64) float64 {
	return math.Log(errs[len(errs)-1]/errs[0]) / math.Log(hs[len(hs)-1]/hs[0])
}

func TestConvergenceFunctionsAreSecondOrder(t *testing.T) {
	sizes := []int{16, 32, 64, 128}

	cases := []struct {
		name string
		fn   func([]int) ([]float64, []float64, error)
	}{
		{"Dirichlet", convergeDirichlet2D},
		{"Neumann", convergeNeumann2D},
		{"Periodic", convergePeriodic2D},
	}

	for _, c := range cases {
		hs, errs, err := c.fn(sizes)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if len(hs) != len(sizes) || len(errs) != len(sizes) {
			t.Fatalf("%s: expected %d points, got %d/%d", c.name, len(sizes), len(hs), len(errs))
		}
		for i, e := range errs {
			if e <= 0 || math.IsNaN(e) || math.IsInf(e, 0) {
				t.Fatalf("%s: non-positive/finite error at %d: %g", c.name, i, e)
			}
		}
		if order := observedOrder(hs, errs); order < 1.8 {
			t.Errorf("%s: observed order %.3f below 1.8 (expected ~2)", c.name, order)
		}
	}
}

func TestRenderSVGWellFormed(t *testing.T) {
	sizes := []int{16, 32, 64, 128}
	hs, errs, err := convergeDirichlet2D(sizes)
	if err != nil {
		t.Fatal(err)
	}
	svg := renderSVG([]series{{name: "Dirichlet", color: "#2563eb", hs: hs, errs: errs}}, sizes)

	if !strings.HasPrefix(svg, "<svg") {
		t.Fatalf("output does not start with <svg: %.40q", svg)
	}
	if !strings.HasSuffix(strings.TrimSpace(svg), "</svg>") {
		t.Fatal("output is not closed with </svg>")
	}
	if !strings.Contains(svg, "Dirichlet") {
		t.Error("series label missing from SVG")
	}
	if !strings.Contains(svg, "slope 2") {
		t.Error("slope-2 reference label missing from SVG")
	}
}
