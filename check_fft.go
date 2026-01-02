package main

import (
	"fmt"

	algofft "github.com/MeKo-Christian/algo-fft"
)

func main() {
	n := 9
	p, _ := algofft.NewPlan64(n)
	src := make([]complex128, n)
	for i := range n {
		src[i] = complex(float64(i+1), 0)
	}

	tmp := make([]complex128, n)
	p.Forward(tmp, src)

	dst := make([]complex128, n)
	p.Inverse(dst, tmp)

	fmt.Printf("n=%d\n", n)
	fmt.Printf("src[0]=%v\n", src[0])
	fmt.Printf("dst[0]=%v\n", dst[0])

	n2 := 8
	p2, _ := algofft.NewPlan64(n2)
	src2 := make([]complex128, n2)
	for i := range n2 {
		src2[i] = complex(float64(i+1), 0)
	}
	tmp2 := make([]complex128, n2)
	p2.Forward(tmp2, src2)
	dst2 := make([]complex128, n2)
	p2.Inverse(dst2, tmp2)
	fmt.Printf("n=%d\n", n2)
	fmt.Printf("src[0]=%v\n", src2[0])
	fmt.Printf("dst[0]=%v\n", dst2[0])
}
