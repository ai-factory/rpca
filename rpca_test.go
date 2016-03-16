package RPCA

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"testing"
)

func fmtMat(mat mat64.Matrix) fmt.Formatter {
	return mat64.Formatted(mat, mat64.Excerpt(2), mat64.Squeeze())
}

func TestComputeRPCA(t *testing.T) {
	for _, test := range rpcaTestCases {
		if test.skip {
			continue
		}
		observed := computeRPCA(test.timeSeries, test.options)
		if !mat64.EqualApprox(observed.L, test.expected.L, 0.01) {
			t.Errorf("Failed matching low-rank representation (L) with %v.\n"+
				"Expected:\n%v\n\nbut got\n\n%v",
				test.description, fmtMat(test.expected.L), fmtMat(observed.L))
		}
		if !mat64.EqualApprox(observed.S, test.expected.S, 0.01) {
			t.Errorf("Failed matching sparse matrix (S) with %v.\n"+
				"Expected:\n%v\n\nbut got\n\n%v",
				test.description, fmtMat(test.expected.S), fmtMat(observed.S))
		}
		if !mat64.EqualApprox(observed.E, test.expected.E, 0.01) {
			t.Errorf("Failed matching error matrix (E) with " + test.description)
		}
		if observed.converged != test.expected.converged {
			t.Errorf("Failed to match convergence with %v. Expected %v but got %v",
				test.description, test.expected.converged, observed.converged)
		}
		if observed.iterations != test.expected.iterations {
			t.Errorf("Failed to match iterations with %v. Expected %v but got %v",
				test.description, test.expected.iterations, observed.iterations)
		}
	}
}

func TestFindAnomalies(t *testing.T) {
}
