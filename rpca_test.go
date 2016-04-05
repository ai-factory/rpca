package rpca

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
			t.Errorf("Failed '%v' on matching low-rank representation (L).\n"+
				"Expected:\n%v\n\nbut got\n\n%v",
				test.description, fmtMat(test.expected.L), fmtMat(observed.L))
		}
		if !mat64.EqualApprox(observed.S, test.expected.S, 0.01) {
			t.Errorf("Failed '%v' on matching sparse matrix (S).\n"+
				"Expected:\n%v\n\nbut got\n\n%v",
				test.description, fmtMat(test.expected.S), fmtMat(observed.S))
		}
		if !mat64.EqualApprox(observed.E, test.expected.E, 0.01) {
			t.Errorf("Failed '%v' on matching error matrix (E).\n"+
				"Expected:\n%v\n\nbut got\n\n%v",
				test.description, fmtMat(test.expected.E), fmtMat(observed.E))
		}
		if observed.converged != test.expected.converged {
			t.Errorf("Failed '%v' on matching convergence. Expected %v but got %v",
				test.description, test.expected.converged, observed.converged)
		}
		if observed.iterations != test.expected.iterations {
			t.Errorf("Failed '%v' on matching iterations. Expected %v but got %v",
				test.description, test.expected.iterations, observed.iterations)
		}
	}
}

func BenchmarkComputeRPCA(b *testing.B) {
	testCase := rpcaTestCases[2]
	for i := 0; i < b.N; i++ {
		computeRPCA(testCase.timeSeries, testCase.options)
	}
}
