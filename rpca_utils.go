package rpca

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

func signum(f float64) float64 {
	switch {
	case f < 0:
		return -1
	case f == 0:
		return 0
	default:
		return 1
	}
}

func setDiag(mat mat64.Mutable, d []float64) {
	for i, v := range d {
		mat.Set(i, i, v)
	}
}

func add(mat mat64.Mutable, addend float64) {
	r, c := mat.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			mat.Set(i, j, mat.At(i, j)+addend)
		}
	}
}

func sum(vals []float64) float64 {
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum
}

func l1Norm(mat mat64.RawMatrixer) float64 {
	sum := 0.0
	for _, v := range mat.RawMatrix().Data {
		sum += math.Abs(v)
	}
	return sum
}

func buildMatrix(series []float64, frequency int) rPCAable {
	lenSeries := len(series)
	if frequency <= 0 {
		panic("Frequency less than or equal to zero")
	}
	if lenSeries%frequency != 0 {
		panic("Time series not evenly divisible by frequency")
	}
	rows, cols := frequency, lenSeries/frequency
	return mat64.DenseCopyOf(mat64.NewDense(cols, rows, series).T())
}

func matrixData(mat mat64.Matrix) []float64 {
	r, c := mat.Dims()
	data := make([]float64, r*c)
	for j := 0; j < c; j++ {
		for i := 0; i < r; i++ {
			data[i+j*r] = mat.At(i, j)
		}
	}
	return data
}

func diff(x []float64) []float64 {
	y := make([]float64, len(x)-1)
	for i := 0; i < len(x)-1; i++ {
		y[i] = x[i+1] - x[i]
	}
	return y
}
