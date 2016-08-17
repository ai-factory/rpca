package rpca

type rpcaConfig struct {
	frequency int
	autodiff  bool
	forcediff bool
	scale     bool
	lPenalty  float64
	sPenalty  float64
	verbose   bool
}

// Frequency informs the algorithm of the major frequency of the time series to
// use for analysis. For example, if you have 56 points of daily measurements,
// the major frequency is likely 7, which would capture the weekly trend. Note
// that due to the nature of the algorithm, the length of the provided time
// series must be divisible by the frequency.
func Frequency(freq int) func(*rpcaConfig) error {
	return func(conf *rpcaConfig) error {
		conf.frequency = freq
		return nil
	}
}

// Whether or not to detect if the given time series contains a significant
// global trend that should be removed before anomaly detection. Trend
// detection is done with the Augmented Dickey-Fuller test. Note that
// auto-differencing will change the nature of the detected anomalies. If the
// time series is not detrended, a lasting mean-shift in the time series (for
// example, a large, sustained increase) will result in a number of consecutive
// points after the shift being identified as anomalous. If the time series is
// detrended, only the single point that marks the beginning of the shift will
// be identified as anomalous.
func AutoDiff(active bool) func(*rpcaConfig) error {
	return func(conf *rpcaConfig) error {
		conf.autodiff = active
		return nil
	}
}

// If true, skip the Augmented Dickey-Fuller test and always auto-difference
// the given time series.
func ForceDiff(active bool) func(*rpcaConfig) error {
	return func(conf *rpcaConfig) error {
		conf.forcediff = active
		return nil
	}
}

// If false, do not normalize the time series before running anomaly detection.
// This could result in the algorithm not converging on a nice solution.
func Scale(active bool) func(*rpcaConfig) error {
	return func(conf *rpcaConfig) error {
		conf.scale = active
		return nil
	}
}

// A scalar for the amount of thresholding to use when determining the low rank
// approximation of the given time series.  The default values are chosen to
// correspond to the smart thresholding values described in Zhou's Stable
// Principal Component Pursuit.
func LPenalty(penalty float64) func(*rpcaConfig) error {
	return func(conf *rpcaConfig) error {
		conf.lPenalty = penalty
		return nil
	}
}

// A scalar for the amount of thresholding to use when determining the
// separation between noise and sparse outliers.  The default values are chosen
// to correspond to the smart thresholding values described in Zhou's Stable
// Principal Component Pursuit.
func SPenalty(penalty float64) func(*rpcaConfig) error {
	return func(conf *rpcaConfig) error {
		conf.sPenalty = penalty
		return nil
	}
}

// If true, print lots of information about each iteration of the algorithm.
func Verbose(active bool) func(*rpcaConfig) error {
	return func(conf *rpcaConfig) error {
		conf.verbose = active
		return nil
	}
}
