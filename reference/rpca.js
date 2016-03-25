X = [];
L = [];
S = [];
E = [];
lpenalty = 0;
spenalty = 0;
MAX_ITERS = 500;

X_Array = [];
L_Array = [];
S_Array = [];
E_Array = [];

mu_Array = [];
diff_Array = [];

function RSVD(data,_lpenalty,_spenalty) {


	thisModel = {};
	thisModel.X = data;
  
	lpenalty = parseFloat(d3.select("#lpenalty").property("value"));
	spenalty = parseFloat(d3.select("#spenalty").property("value"));
	
	//Added for form
	X = data;
	
	initMatrices();
	computeRSVD();
}

function initMatrices() {
	L = X.map(function(d) {return d.map(function(d) {return 0})});
	S = X.map(function(d) {return d.map(function(d) {return 0})});
	E = X.map(function(d) {return d.map(function(d) {return 0})});

}

function computeRSVD() {
	mu = X.length * X[0].length / (4 * l1norm(X));
	objPrev = 0.5 * numeric.norm2Squared(X);
	obj = objPrev;
	tol = objPrev * 1e-8;
	diff = 2 * tol;
	
	iter = 0;

	while (diff > tol && iter < MAX_ITERS) {
    total_iterations++;
		
    nuclearNorm = computeS(mu);

		l1Norm = computeL(mu);

		l2Norm = computeE();

		obj = computeObjective(nuclearNorm, l1Norm, l2Norm);

		diff = Math.abs(objPrev - obj);
		objPrev = obj;
		mu = computeDynamicMu();

//    console.log("Iteration:", iter)
//    console.log("ComputeL:", l1norm)
//    console.log("MU:", mu)
		
    L_Array.push(L);
    S_Array.push(S);
    E_Array.push(E);

    mu_Array.push(mu);
    diff_Array.push(diff);

		iter += 1;
	}
	
}

function computeDynamicMu() {

  var _m = E.length;
  var _n = E[0].length;
  var E_sd = standardDeviationArray(E);

  var _mu = E_sd * Math.sqrt(2 * Math.max(_m,_n));

  return Math.max(.01,_mu);

	// _MAD = MedianAbsoluteDeviation(E);
	// return Math.max(.01, _MAD);
}

function standardDeviationArray(_array) {
    var _flatArray = [];
  for (var x = 0; x<_array.length;x++) {
    for (var y = 0; y<_array[x].length;y++) {
      _flatArray.push(_array[x][y])
    }
  }

  return standardDeviation(_flatArray);
}

function MedianAbsoluteDeviation(_array) {
	_values = [];
	
	for(var x = 0;x<_array.length;x++) {
		for(var y = 0;y<_array[x].length;y++) {
			_values.push(_array[x][y]);
		}
	}
	_median = median(_values);
	
	_values_abs = [];
	for(var x = 0;x<_array.length;x++) {
		for(var y = 0;y<_array[x].length;y++) {
			_values_abs.push(Math.abs(_array[x][y]) - _median);
		}
	}
	
	_AbsMedian = median(_values_abs);
	
	return _AbsMedian;

	function median(values) {

    values.sort( function(a,b) {return a - b;} );

    var half = Math.floor(values.length/2);

    if(values.length % 2)
        return values[half];
    else
        return (values[half-1] + values[half]) / 2.0;
	}

}

function computeObjective(_nuclearnorm, _l1norm, _l2norm) {
	return 0.5 * _l2norm + _nuclearnorm + _l1norm;
}


function computeE() {

//	E = numeric.sub(X,(numeric.sub(L,S)));
    E = numeric.sub((numeric.sub(X,L)),S);
	return numeric.norm2Squared(E);
	}

function computeL(_mu) {
	LPenalty = lpenalty * _mu;

//  console.log("LPenalty:", LPenalty)

	svd = numeric.svd(numeric.sub(X,S));
	
	penalizedD = softThresholdSingle(svd.S, LPenalty);
	D_matrix = numeric.diag(penalizedD);
	L = numeric.dot(svd.U,numeric.dot(D_matrix,numeric.transpose(svd.V)))

	return numeric.sum(penalizedD) * LPenalty;
}

function computeS(_mu) {
	SPenalty = spenalty * _mu;

	penalizedS = softThreshold(numeric.sub(X,L),SPenalty);
	S = penalizedS;

	return l1norm(penalizedS) * SPenalty;
}

function softThreshold(_array,_penalty) {
	for(var x = 0;x<_array.length;x++) {
		for(var y = 0;y<_array[x].length;y++) {
			_array[x][y] = signum(_array[x][y]) * Math.max(Math.abs(_array[x][y]) - _penalty, 0);
		}
	}
	return _array;
}

function softThresholdSingle(_array,_penalty) {
	for(var x = 0;x<_array.length;x++) {
			_array[x] = signum(_array[x]) * Math.max(Math.abs(_array[x]) - _penalty, 0);
	}
	return _array;
}


function signum(_x) {
	return _x?_x<0?-1:1:0;
}

function arraySubtract(_array1,_array2) {
	_resultArray = _array1.map(function(d) {return d.map(function(){return 0})})
	for(var x = 0;x<_array1.length;x++) {
		for(var y = 0;y<_array1[x].length;y++) {
			_resultArray[x][y] = _array1[x][y] - _array2[x][y];
		}
	}
	return _resultArray;
}

function l1norm(_data) {
	var l1norm = 0;
	for (var x = 0; x < _data.length;x++) {
		for (var y = 0; y < _data[x].length;y++) {
			l1norm += Math.abs(_data[x][y]);
		}	
	}

	return l1norm;
}
