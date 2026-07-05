// Command convergence-plot regenerates the log-log convergence plots checked
// into docs/. It runs the manufactured-solution convergence studies used by the
// test suite for each boundary condition and emits a self-contained SVG (no
// external plotting dependency, no runtime assets) whose slope-2 reference line
// makes the second-order accuracy of the spectral solver visible at a glance.
//
// Regenerate with:
//
//	go generate ./docs/...
//
// or directly:
//
//	go run ./cmd/convergence-plot -out docs/convergence-2d.svg
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strings"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

// series is one boundary-condition convergence curve: the per-grid spacings and
// the corresponding max-abs errors against the manufactured analytic solution.
type series struct {
	name  string
	color string
	hs    []float64
	errs  []float64
}

func main() {
	out := flag.String("out", "docs/convergence-2d.svg", "output SVG path")
	flag.Parse()

	sizes := []int{16, 32, 64, 128}

	all := []series{
		{name: "Dirichlet", color: "#2563eb", hs: nil, errs: nil},
		{name: "Neumann", color: "#059669", hs: nil, errs: nil},
		{name: "Periodic", color: "#d97706", hs: nil, errs: nil},
	}

	for i := range all {
		var hs, errs []float64
		var err error
		switch all[i].name {
		case "Dirichlet":
			hs, errs, err = convergeDirichlet2D(sizes)
		case "Neumann":
			hs, errs, err = convergeNeumann2D(sizes)
		case "Periodic":
			hs, errs, err = convergePeriodic2D(sizes)
		}
		if err != nil {
			log.Fatalf("%s convergence: %v", all[i].name, err)
		}
		all[i].hs = hs
		all[i].errs = errs
	}

	svg := renderSVG(all, sizes)
	if err := os.WriteFile(*out, []byte(svg), 0o644); err != nil {
		log.Fatalf("writing %s: %v", *out, err)
	}

	fmt.Printf("wrote %s\n", *out)
	for _, s := range all {
		fmt.Printf("  %-10s", s.name)
		for k := range s.hs {
			fmt.Printf(" h=%.4g err=%.3e", s.hs[k], s.errs[k])
			if k+1 < len(s.hs) {
				rate := math.Log(s.errs[k+1]/s.errs[k]) / math.Log(s.hs[k+1]/s.hs[k])
				fmt.Printf(" (rate %.2f)", rate)
			}
		}
		fmt.Println()
	}
}

// convergeDirichlet2D solves -Δu = λu for u = sin(πx/L)sin(πy/L) on a
// vertex-centered grid (nodes at (i+1)h, L = (n+1)h) and returns the spacings
// and max-abs errors against the analytic solution.
func convergeDirichlet2D(sizes []int) (hs, errs []float64, err error) {
	hs = make([]float64, len(sizes))
	errs = make([]float64, len(sizes))
	for idx, n := range sizes {
		h := 1.0 / float64(n+1)
		L := float64(n+1) * h
		hs[idx] = h

		k := math.Pi / L
		lambda := 2.0 * k * k

		u := make([]float64, n*n)
		for i := range n {
			x := float64(i+1) * h
			for j := range n {
				y := float64(j+1) * h
				u[i*n+j] = math.Sin(k*x) * math.Sin(k*y)
			}
		}

		plan, e := poisson.NewPlan(2, []int{n, n}, []float64{h, h},
			[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet})
		if e != nil {
			return nil, nil, e
		}

		rhs := make([]float64, n*n)
		for i := range rhs {
			rhs[i] = lambda * u[i]
		}
		got := make([]float64, n*n)
		if e := plan.Solve(got, rhs); e != nil {
			return nil, nil, e
		}
		errs[idx] = maxAbsDiff(got, u)
	}
	return hs, errs, nil
}

// convergeNeumann2D solves -Δu = λu for u = cos(kx·x)cos(ky·y) on a
// cell-centered grid (nodes at (i+½)h, L = nh); the field has vanishing normal
// derivative on every face, so it is a compatible Neumann problem.
func convergeNeumann2D(sizes []int) (hs, errs []float64, err error) {
	hs = make([]float64, len(sizes))
	errs = make([]float64, len(sizes))
	for idx, n := range sizes {
		h := 1.0 / float64(n)
		L := float64(n) * h
		hs[idx] = h

		k := math.Pi / L
		lambda := 2.0 * k * k

		u := make([]float64, n*n)
		for i := range n {
			x := (float64(i) + 0.5) * h
			for j := range n {
				y := (float64(j) + 0.5) * h
				u[i*n+j] = math.Cos(k*x) * math.Cos(k*y)
			}
		}
		meanU := sliceMean(u)

		plan, e := poisson.NewPlan(2, []int{n, n}, []float64{h, h},
			[]poisson.BCType{poisson.Neumann, poisson.Neumann},
			poisson.WithSubtractMean(), poisson.WithSolutionMean(meanU))
		if e != nil {
			return nil, nil, e
		}

		rhs := make([]float64, n*n)
		for i := range rhs {
			rhs[i] = lambda * u[i]
		}
		got := make([]float64, n*n)
		if e := plan.Solve(got, rhs); e != nil {
			return nil, nil, e
		}
		errs[idx] = maxAbsDiff(got, u)
	}
	return hs, errs, nil
}

// convergePeriodic2D solves -Δu = λu for u = sin(2πx/L)sin(2πy/L) on a periodic
// grid (nodes at i·h, L = nh). The field is mean-zero, so the constant
// nullspace is handled with WithSubtractMean.
func convergePeriodic2D(sizes []int) (hs, errs []float64, err error) {
	hs = make([]float64, len(sizes))
	errs = make([]float64, len(sizes))
	for idx, n := range sizes {
		h := 1.0 / float64(n)
		L := float64(n) * h
		hs[idx] = h

		k := 2.0 * math.Pi / L
		lambda := 2.0 * k * k

		u := make([]float64, n*n)
		for i := range n {
			x := float64(i) * h
			for j := range n {
				y := float64(j) * h
				u[i*n+j] = math.Sin(k*x) * math.Sin(k*y)
			}
		}

		plan, e := poisson.NewPlan(2, []int{n, n}, []float64{h, h},
			[]poisson.BCType{poisson.Periodic, poisson.Periodic},
			poisson.WithSubtractMean())
		if e != nil {
			return nil, nil, e
		}

		rhs := make([]float64, n*n)
		for i := range rhs {
			rhs[i] = lambda * u[i]
		}
		got := make([]float64, n*n)
		if e := plan.Solve(got, rhs); e != nil {
			return nil, nil, e
		}
		errs[idx] = maxAbsDiff(got, u)
	}
	return hs, errs, nil
}

func maxAbsDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

func sliceMean(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

// ---- SVG rendering -------------------------------------------------------

const (
	svgW    = 760
	svgH    = 480
	marginL = 78
	marginR = 168
	marginT = 46
	marginB = 66
)

// plot bounds in pixels.
func plotBox() (x0, y0, x1, y1 float64) {
	return marginL, marginT, svgW - marginR, svgH - marginB
}

// renderSVG draws the log-log convergence plot for every series plus a slope-2
// reference line and a legend. The whole figure is self-contained (an inline
// card background, no external assets) so it renders identically on GitHub in
// light and dark themes.
func renderSVG(all []series, sizes []int) string {
	// Data ranges in log10 space.
	minLogH, maxLogH := math.Inf(1), math.Inf(-1)
	minLogE, maxLogE := math.Inf(1), math.Inf(-1)
	for _, s := range all {
		for _, h := range s.hs {
			lh := math.Log10(h)
			minLogH, maxLogH = math.Min(minLogH, lh), math.Max(maxLogH, lh)
		}
		for _, e := range s.errs {
			le := math.Log10(e)
			minLogE, maxLogE = math.Min(minLogE, le), math.Max(maxLogE, le)
		}
	}
	// Pad the error axis to whole decades so the gridlines land on 10^k.
	minLogE = math.Floor(minLogE) - 0.15
	maxLogE = math.Ceil(maxLogE) + 0.15
	padH := 0.08 * (maxLogH - minLogH)
	minLogH -= padH
	maxLogH += padH

	x0, y0, x1, y1 := plotBox()
	sx := func(lh float64) float64 { return x0 + (lh-minLogH)/(maxLogH-minLogH)*(x1-x0) }
	sy := func(le float64) float64 { return y1 - (le-minLogE)/(maxLogE-minLogE)*(y1-y0) }

	var b strings.Builder
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" `+
		`viewBox="0 0 %d %d" font-family="-apple-system,Segoe UI,Roboto,sans-serif">`+"\n",
		svgW, svgH, svgW, svgH)
	// Card background (light, works under both GitHub themes).
	fmt.Fprintf(&b, `<rect x="0" y="0" width="%d" height="%d" rx="10" fill="#ffffff"/>`+"\n", svgW, svgH)
	fmt.Fprintf(&b, `<rect x="0.5" y="0.5" width="%d" height="%d" rx="10" fill="none" stroke="#e2e8f0"/>`+"\n",
		svgW-1, svgH-1)

	// Title.
	fmt.Fprintf(&b, `<text x="%d" y="26" font-size="17" font-weight="600" fill="#0f172a">`+
		`2D Poisson convergence (max error vs grid spacing h)</text>`+"\n", marginL)

	// Plot frame.
	fmt.Fprintf(&b, `<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#f8fafc" stroke="#cbd5e1"/>`+"\n",
		x0, y0, x1-x0, y1-y0)

	// Horizontal gridlines + y labels at whole decades.
	for k := int(math.Ceil(minLogE)); float64(k) <= maxLogE; k++ {
		yy := sy(float64(k))
		if yy < y0-0.5 || yy > y1+0.5 {
			continue
		}
		fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e2e8f0"/>`+"\n", x0, yy, x1, yy)
		fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" font-size="12" text-anchor="end" fill="#475569">10<tspan `+
			`baseline-shift="super" font-size="9">%d</tspan></text>`+"\n", x0-8, yy+4, k)
	}

	// Vertical gridlines + x labels at each grid spacing, labeled by the actual h
	// value the tick sits at (the axis is scaled in log10(h), not grid size).
	for i := range sizes {
		// Use the Dirichlet series h (they are all close) for tick placement.
		h := all[0].hs[i]
		xx := sx(math.Log10(h))
		fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#eef2f7"/>`+"\n", xx, y0, xx, y1)
		fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" font-size="12" text-anchor="middle" fill="#475569">%.3g</text>`+"\n",
			xx, y1+20, h)
	}

	// Axis titles.
	fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" font-size="13" text-anchor="middle" fill="#334155">grid spacing h</text>`+"\n",
		(x0+x1)/2, y1+44)
	fmt.Fprintf(&b, `<text x="20" y="%.1f" font-size="13" text-anchor="middle" fill="#334155" `+
		`transform="rotate(-90 20 %.1f)">max abs error</text>`+"\n", (y0+y1)/2, (y0+y1)/2)

	// Slope-2 reference line. Anchored to the finest Dirichlet point but shifted
	// down half a decade so it reads as a distinct parallel guide rather than
	// hiding underneath the (genuinely slope-2) data.
	refIdx := len(all[0].hs) - 1
	refH, refE := all[0].hs[refIdx], all[0].errs[refIdx]
	const refShift = 0.55 // decades below the anchor point
	refLine := func(lh float64) float64 {
		return math.Log10(refE) - refShift + 2.0*(lh-math.Log10(refH))
	}
	lhA, lhB := minLogH+padH, maxLogH-padH
	fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#94a3b8" `+
		`stroke-width="1.5" stroke-dasharray="6 5"/>`+"\n",
		sx(lhA), sy(refLine(lhA)), sx(lhB), sy(refLine(lhB)))
	fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" font-size="12" fill="#64748b" font-style="italic">slope 2 `+
		`(O(h²))</text>`+"\n", sx(lhA)+6, sy(refLine(lhA))-6)

	// Data series.
	for _, s := range all {
		var pts strings.Builder
		for k := range s.hs {
			fmt.Fprintf(&pts, "%.1f,%.1f ", sx(math.Log10(s.hs[k])), sy(math.Log10(s.errs[k])))
		}
		fmt.Fprintf(&b, `<polyline points="%s" fill="none" stroke="%s" stroke-width="2.4"/>`+"\n",
			strings.TrimSpace(pts.String()), s.color)
		for k := range s.hs {
			fmt.Fprintf(&b, `<circle cx="%.1f" cy="%.1f" r="4" fill="%s" stroke="#ffffff" stroke-width="1.5"/>`+"\n",
				sx(math.Log10(s.hs[k])), sy(math.Log10(s.errs[k])), s.color)
		}
	}

	// Legend (top-right, inside the right margin).
	lx := x1 + 20
	ly := y0 + 8
	for i, s := range all {
		yy := ly + float64(i)*24
		fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2.4"/>`+"\n",
			lx, yy, lx+26, yy, s.color)
		fmt.Fprintf(&b, `<circle cx="%.1f" cy="%.1f" r="4" fill="%s" stroke="#ffffff" stroke-width="1.5"/>`+"\n",
			lx+13, yy, s.color)
		fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" font-size="13" fill="#0f172a">%s</text>`+"\n",
			lx+34, yy+4, s.name)
	}
	// Observed order annotation under the legend.
	oy := ly + float64(len(all))*24 + 14
	fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" font-size="11" fill="#64748b">observed order:</text>`+"\n", lx, oy)
	for i, s := range all {
		rate := math.Log(s.errs[len(s.errs)-1]/s.errs[0]) /
			math.Log(s.hs[len(s.hs)-1]/s.hs[0])
		fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" font-size="11" fill="%s">%s ≈ %.2f</text>`+"\n",
			lx, oy+float64(i+1)*16, s.color, s.name, rate)
	}

	b.WriteString("</svg>\n")
	return b.String()
}
