/*
Package rpca implements anomaly detection using Robust Principle Component
Analysis
(http://techblog.netflix.com/2015/02/rad-outlier-detection-on-big-data.html).
It is a port of RPCA provided by Netflix as part of their Surus project
(https://github.com/Netflix/Surus). It takes bits and pieces of Netflix's RAD
implementations written in R, C++, Java, and Javascript.
*/
package rpca

import (
	"github.com/berkmancenter/adf"
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"

	"fmt"
	"math"
)

// The maximum number of iterations before we give up trying to converge.
const MAX_ITERS int = 1000

type Anomalies struct {
	// A slice of booleans indicating which values in the provided time series
	// were anomalous.
	Positions []bool

	// Values is a slice of floats indicating exactly how anomlous each point in
	// the provided time series was. Points that were not anomalous have a value
	// of zero. Points that were anomalously low have negative values, while
	// points that were anomalously high have positive values.
	Values []float64

	// Part of the RPCA process requires normalizing the given time series by
	// subtracting the mean and dividing by the standard deviation (Z scoring)
	// before detecting anomalies. The anomalousness of each point is computed in
	// this Z-scored space before being transformed back into the domain of the
	// given time series. Sometimes, it's useful to have the normalized values,
	// for example, when comparing anomalies across time series.
	NormedValues []float64
}

/*
FindAnomalies is the primary function to use when using this package. It takes
a slice of floats and any number of options. Passing options may look a little
funny. This is because this package uses functional arguments to make the API
easier to use (more on functional arguments here:
http://dave.cheney.net/2014/10/17/functional-options-for-friendly-apis).
Basically, all options have default values, and to change that value, pass
options like so:

	anoms := rpca.FindAnomalies(series, rpca.Frequency(7), rpca.AutoDiff(true))

The interface is designed to match that of Netflix's anomaly detection
R package.
*/
func FindAnomalies(series []float64, options ...func(*rpcaConfig) error) Anomalies {
	conf := rpcaConfig{
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
	normedAnomalyMat := mat64.DenseCopyOf(decomp.SNormed.T())
	normedAnomalies := normedAnomalyMat.RawMatrix().Data
	positions := make([]bool, len(anomalies))
	for i, v := range anomalies {
		positions[i] = v != 0
	}
	return Anomalies{positions, anomalies, normedAnomalies}
}

type decomposedMatrix struct {
	L, S, SNormed, E rPCAable
	converged        bool
	iterations       int
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

func computeRPCA(mat rPCAable, conf *rpcaConfig) decomposedMatrix {
	var mean, stdDev float64
	rows, cols := mat.Dims()
	needsDiff := false
	adf := adf.New(mat.RawMatrix().Data, 0, -1)

	if conf.autodiff {
		adf.Run()
		needsDiff = !adf.IsStationary()
	}

	if needsDiff || conf.forcediff {
		diffed := diff(matrixData(mat))
		diffed = append([]float64{0}, diffed...)
		mat = buildMatrix(diffed, conf.frequency)
	}

	if conf.scale {
		mean, stdDev = stat.MeanStdDev(mat.RawMatrix().Data, nil)
		if conf.verbose {
			println("Mean, StdDev: ", mean, stdDev)
		}
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
		fmt.Printf("Matrix:\n%v\n\n", mat64.Formatted(mat))
	}

	iter := 0
	converged := false

	for difference > total && iter < MAX_ITERS {
		thisLPenalty := mu * conf.lPenalty
		thisSPenalty := mu * conf.sPenalty

		if conf.verbose {
			println("S penalty (", conf.sPenalty, ") with mu ", mu, ":", thisSPenalty)
			println("L penalty with mu ", mu, ":", thisLPenalty)
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
			println("Mu after iteration ", iter, ": ", mu, "\n")
		}
		iter++
	}
	if iter < MAX_ITERS {
		converged = true
	}
	sNormed := mat64.DenseCopyOf(s)
	if conf.scale {
		if conf.verbose {
			println("Mean, StdDev: ", mean, stdDev)
		}
		l.Scale(stdDev, l)
		add(l, mean)
		s.Scale(stdDev, s)
		e.Scale(stdDev, e)
	}
	// Not sure why this is required, but it is.
	if needsDiff || conf.forcediff {
		l = mat64.NewDense(rows, cols, matrixData(l))
		s = mat64.NewDense(rows, cols, matrixData(s))
		sNormed = mat64.NewDense(rows, cols, matrixData(sNormed))
		e = mat64.NewDense(rows, cols, matrixData(e))
	}
	return decomposedMatrix{l, s, sNormed, e, converged, iter}
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
