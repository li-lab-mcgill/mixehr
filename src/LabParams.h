#ifndef LABPARAMS_H_
#define LABPARAMS_H_

#include <map>
#include <string>
#include <armadillo>
#include <boost/math/special_functions/digamma.hpp>

using namespace std;
using namespace arma;

struct LabParams {

	int V;

	// V x K
	mat eta;
	mat eta_normalized;

	// 1 x K
	rowvec psi;

	// expected observed count when lab is in topic k
	rowvec observedCnt; // 1 x K

	// expected missing count when lab is in topic k
	rowvec missingCnt; // 1 x K

	// hyperparameters for psi (beta distribution)
	double a;
	double b;

	// V x 1 hyperparameters for eta (lab states)
	vec zeta;

	LabParams() {
		V=2;
		a=1;
		b=1;
		observedCnt.zeros();
		missingCnt.zeros();
	};
};

#endif /* LABPARAMS_H_ */
