package rpca

import (
	"fmt"
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"math"
)

const MAX_ITERS int = 1000

type Anomalies struct {
	Positions []bool
	Values    []float64
}

func FindAnomalies(series []float64, options ...func(*RPCAConfig) error) Anomalies {
	conf := RPCAConfig{
		frequency: 7,
		autodiff:  true,
		forcediff: false,
		scale:     true,
		lPenalty:  1.0,
		sPenalty:  1.4,
		verbose:   false,
	}

	// Apply because we need to know the frequency
	for _, option := range options {
		option(&conf)
	}

	floatFreq := float64(conf.frequency)
	conf.sPenalty = 1.4 / math.Sqrt(math.Max(floatFreq, float64(len(series))/floatFreq))

	// Apply again in case user provided S penalty
	for _, option := range options {
		option(&conf)
	}

	mat := buildMatrix(series, conf.frequency)
	decomposed := computeRPCA(mat, &conf)
	return decomposedToAnomalies(&decomposed)
}

func decomposedToAnomalies(decomp *decomposedMatrix) Anomalies {
	anomalyMat := mat64.DenseCopyOf(decomp.S.T())
	anomalies := anomalyMat.RawMatrix().Data
	positions := make([]bool, len(anomalies))
	for i, v := range anomalies {
		positions[i] = v != 0
	}
	return Anomalies{positions, anomalies}
}

type decomposedMatrix struct {
	L, S, E    rPCAable
	converged  bool
	iterations int
}
type rPCAComponent struct {
	matrix *mat64.Dense
	norm   float64
}
type rPCAable interface {
	mat64.Mutable
	mat64.RawMatrixer
	Scale(f float64, a mat64.Matrix)
}

func computeRPCA(mat rPCAable, conf *RPCAConfig) decomposedMatrix {
	rows, cols := mat.Dims()
	var mean, stdDev float64

	if conf.scale {
		mean, stdDev = stat.MeanStdDev(mat.RawMatrix().Data, nil)
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

	if conf.verbose {
		println("Rows x cols:", rows, cols)
		println("Objective initial:", objective)
		println("Total initial:", total)
		println("Diff initial:", difference)
		println("Mu initial:", mu)
		println("l1norm:", l1Norm(mat))
		fa := mat64.Formatted(mat)
		fmt.Printf("Matrix:\n%v\n\n", fa)
	}

	iter := 0
	converged := false

	for difference > total && iter < MAX_ITERS {
		thisLPenalty := mu * conf.lPenalty
		thisSPenalty := mu * conf.sPenalty

		if conf.verbose {
			println("S penalty:", thisSPenalty)
			println("L penalty:", thisLPenalty)
		}

		sComp := computeS(mat, l, thisSPenalty)
		s = sComp.matrix
		lComp := computeL(mat, s, thisLPenalty)
		l = lComp.matrix
		eComp := computeE(mat, l, s)
		e = eComp.matrix

		objective = computeObjective(lComp.norm, sComp.norm, eComp.norm)
		difference = math.Abs(previousObjective - objective)
		previousObjective = objective

		mu = computeDynamicMu(e)
		if conf.verbose {
			println("L S E norms:", lComp.norm, sComp.norm, eComp.norm)
			println("Objective function: ", previousObjective, " on previous iteration ", iter)
			println("Objective function: ", objective, " on iteration ", iter-1)
			println("Mu on iteration ", iter, ": ", mu)
		}
		iter++
	}
	if iter < MAX_ITERS {
		converged = true
	}
	if conf.scale {
		if conf.verbose {
			println("Mean, StdDev: ", mean, stdDev)
		}
		l.Scale(stdDev, l)
		add(l, mean)
		s.Scale(stdDev, s)
		e.Scale(stdDev, e)
	}
	return decomposedMatrix{l, s, e, converged, iter}
}

func computeDynamicMu(e *mat64.Dense) float64 {
	r, c := e.Dims()
	eStdDev := stat.StdDev(e.RawMatrix().Data, nil)
	mu := eStdDev * math.Sqrt(2*math.Max(float64(r), float64(c)))
	return math.Max(0.01, mu)
}

func computeS(mat, l mat64.Matrix, penalty float64) rPCAComponent {
	r, c := mat.Dims()
	residual := mat64.NewDense(r, c, nil)
	residual.Sub(mat, l)
	s := softThresholdMat(residual, penalty)
	norm := l1Norm(s) * penalty
	return rPCAComponent{s.(*mat64.Dense), norm}
}

func computeObjective(lNorm, sNorm, eNorm float64) float64 {
	return (0.5 * eNorm) + lNorm + sNorm
}

func computeL(mat, s mat64.Matrix, penalty float64) rPCAComponent {
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
	return rPCAComponent{l, mat64.Sum(penalizedDiag) * penalty}
}

func computeE(mat, l, s mat64.Matrix) rPCAComponent {
	r, c := mat.Dims()
	e := mat64.NewDense(r, c, nil)
	e.Sub(mat, l)
	e.Sub(e, s)
	return rPCAComponent{e, math.Pow(mat64.Norm(e, 2), 2)}
}

//TODO Make these one function
func softThresholdMat(mat mat64.Matrix, penalty float64) rPCAable {
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
