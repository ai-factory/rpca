package rpca

type RPCAConfig struct {
	frequency int
	autodiff  bool
	forcediff bool
	scale     bool
	lPenalty  float64
	sPenalty  float64
	verbose   bool
}

func Frequency(freq int) func(*RPCAConfig) error {
	return func(conf *RPCAConfig) error {
		conf.frequency = freq
		return nil
	}
}
func AutoDiff(active bool) func(*RPCAConfig) error {
	return func(conf *RPCAConfig) error {
		conf.autodiff = active
		return nil
	}
}
func ForceDiff(active bool) func(*RPCAConfig) error {
	return func(conf *RPCAConfig) error {
		conf.forcediff = active
		return nil
	}
}
func Scale(active bool) func(*RPCAConfig) error {
	return func(conf *RPCAConfig) error {
		conf.scale = active
		return nil
	}
}
func LPenalty(penalty float64) func(*RPCAConfig) error {
	return func(conf *RPCAConfig) error {
		conf.lPenalty = penalty
		return nil
	}
}
func SPenalty(penalty float64) func(*RPCAConfig) error {
	return func(conf *RPCAConfig) error {
		conf.sPenalty = penalty
		return nil
	}
}
func Verbose(active bool) func(*RPCAConfig) error {
	return func(conf *RPCAConfig) error {
		conf.verbose = active
		return nil
	}
}
