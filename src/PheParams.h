#ifndef PHEPARAMS_H_
#define PHEPARAMS_H_

#include <map>
#include <string>
#include <armadillo>
#include <boost/math/special_functions/digamma.hpp>

using namespace std;
using namespace arma;

struct PheParams {

	// 1 x K
	rowvec phi;
	rowvec phi_normalized;

	double beta;

	PheParams() {

		beta = 0.01;
	};
};

#endif /* PHEPARAMS_H_ */
