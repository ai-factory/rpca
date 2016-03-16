package RPCA

import (
	"fmt"
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"math"
)

const (
	MAX_ITERS int     = 1000
	LPENALTY  float64 = 1
	SPENALTY  float64 = 1.4
	SCALE     bool    = true
)

type DecomposedMatrix struct {
	L, S, E    mat64.Matrix
	converged  bool
	iterations int
}
type RPCAComponent struct {
	component *mat64.Dense
	norm      float64
}
type RPCAable interface {
	mat64.Mutable
	mat64.RawMatrixer
	Scale(f float64, a mat64.Matrix)
}

func buildMatrix(series []float64, frequency int) RPCAable {
	lenSeries := len(series)
	if lenSeries%frequency != 0 {
		panic("Time series not evenly divisible by frequency")
	}
	rows, cols := frequency, lenSeries/frequency
	data := make([]float64, lenSeries)
	for i, v := range series {
		row, col := i%rows, i/rows
		j := row*cols + col
		data[j] = v
	}
	return mat64.NewDense(rows, cols, data)
}

func ComputeRPCA(series []float64, frequency int) DecomposedMatrix {

	mat := buildMatrix(series, frequency)
	rows, cols := mat.Dims()

	if SCALE {
		mean, stdDev := stat.MeanStdDev(mat.RawMatrix().Data, nil)
		add(mat, -mean)
		mat.Scale(1.0/stdDev, mat)
	}

	// Get initial mu, which is our convergence rate
	mu := float64(cols*rows) / (4.0 * l1Norm(mat))
	l := mat64.NewDense(rows, cols, nil)
	s := mat64.NewDense(rows, cols, nil)
	e := mat64.NewDense(rows, cols, nil)

	// Initialize objective
	previousObjective := 0.5 * math.Pow(mat64.Norm(mat, 2), 2)
	objective := previousObjective

	total := 1e-8 * previousObjective
	difference := 2 * total

	println("Rows x cols:", rows, cols)
	println("Objective initial:", objective)
	println("Total initial:", total)
	println("Diff initial:", difference)
	println("l1norm:", l1Norm(mat))
	println("Mu initial:", mu)
	fa := mat64.Formatted(mat, mat64.Prefix("    "))
	fmt.Printf("Matrix:\n%v\n\n", fa)

	iter := 0
	converged := false

	for difference > total && iter < MAX_ITERS {
		thisLPenalty := mu * LPENALTY
		thisSPenalty := mu * SPENALTY

		sComp := computeS(mat, l, thisSPenalty)
		s = sComp.component
		lComp := computeL(mat, s, thisLPenalty)
		l = lComp.component
		eComp := computeE(mat, l, s)
		e = eComp.component

		objective = computeObjective(lComp.norm, sComp.norm, eComp.norm)
		difference = math.Abs(previousObjective - objective)
		previousObjective = objective

		mu = computeDynamicMu(e)
		//TODO These are all wrong
		println("L S E norms:", lComp.norm, sComp.norm, eComp.norm)
		println("Objective function: ", previousObjective, " on previous iteration ", iter)
		println("Objective function: ", objective, " on iteration ", iter-1)
		println("Mu on iteration ", iter, ": ", mu)
		iter++
	}
	if iter < MAX_ITERS {
		converged = true
	}
	return DecomposedMatrix{l, s, e, converged, iter}
}

func computeDynamicMu(e *mat64.Dense) float64 {
	r, c := e.Dims()
	eStdDev := stat.StdDev(e.RawMatrix().Data, nil)
	mu := eStdDev * math.Sqrt(2*math.Max(float64(r), float64(c)))
	return math.Max(0.01, mu)
}

func computeS(mat, l mat64.Matrix, penalty float64) RPCAComponent {
	r, c := mat.Dims()
	residual := mat64.NewDense(r, c, nil)
	residual.Sub(mat, l)
	s := softThresholdMat(residual, penalty)
	norm := l1Norm(s) * penalty
	return RPCAComponent{s.(*mat64.Dense), norm}
}

func computeObjective(lNorm, sNorm, eNorm float64) float64 {
	return (0.5 * eNorm) + lNorm + sNorm
}

func computeL(mat, s mat64.Matrix, penalty float64) RPCAComponent {
	r, c := mat.Dims()
	var svd mat64.SVD
	l := mat64.NewDense(r, c, nil)

	diff := mat64.NewDense(r, c, nil)
	diff.Sub(mat, s)
	svd.Factorize(diff, matrix.SVDFull)
	penalizedValues := softThresholdVec(svd.Values(nil), penalty)
	penalizedDiag := mat64.NewDense(r, c, nil)
	setDiag(penalizedDiag, penalizedValues)
	u := mat64.NewDense(r, r, nil)
	u.UFromSVD(&svd)
	v := mat64.NewDense(c, c, nil)
	v.VFromSVD(&svd)
	vT := v.T()
	l.Mul(u, penalizedDiag)
	l.Mul(l, vT)
	return RPCAComponent{l, mat64.Sum(penalizedDiag) * penalty}
}

func computeE(mat, l, s mat64.Matrix) RPCAComponent {
	r, c := mat.Dims()
	e := mat64.NewDense(r, c, nil)
	e.Sub(mat, l)
	e.Sub(e, s)
	return RPCAComponent{e, math.Pow(mat64.Norm(e, 2), 2)}
}

//TODO Make these one function
func softThresholdMat(mat mat64.Matrix, penalty float64) RPCAable {
	r, c := mat.Dims()
	thresholdedMat := mat64.NewDense(r, c, nil)
	penalize := func(i, j int, v float64) float64 {
		return signum(v) * math.Max(math.Abs(v)-penalty, 0)
	}
	thresholdedMat.Apply(penalize, mat)
	return thresholdedMat
}

func softThresholdVec(v []float64, penalty float64) []float64 {
	var thresholded []float64
	penalize := func(v float64) float64 {
		return signum(v) * math.Max(math.Abs(v)-penalty, 0)
	}
	for _, v := range v {
		thresholded = append(thresholded, penalize(v))
	}
	return thresholded
}

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
