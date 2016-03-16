package RPCA

import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

func TestComputeRPCA(t *testing.T) {
	for _, test := range testCases {
		observed := ComputeRPCA(test.timeSeries, test.frequency)
		if !mat64.EqualApprox(observed.S, test.expected.S, 0.2) {
			t.Fatalf("ComputeRPCA(%v) = %v, want %v",
				test.timeSeries, observed.S, test.expected.S)
		}
	}
}
