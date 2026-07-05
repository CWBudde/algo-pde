// Package docs holds generated documentation artifacts (convergence plots, etc.)
// for the algo-pde project. It contains no runnable code; the go:generate
// directive below regenerates the checked-in SVG plots from the live solver.
//
//go:generate go run ../cmd/convergence-plot -out convergence-2d.svg
package docs
