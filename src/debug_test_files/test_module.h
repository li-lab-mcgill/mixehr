/*
 * test_module.h
 *
 *  Created on: Aug. 26, 2020
 *      Author: yueli
 */

#ifndef SRC_DEBUG_TEST_FILES_TEST_MODULE_H_
#define SRC_DEBUG_TEST_FILES_TEST_MODULE_H_

#include <string>

#include <armadillo>

#include "pairkeyhash.h"

#include <map>
#include <iostream>
using namespace std;
using namespace arma;

class Test_Module {

public:
	// 1 x K
	unordered_map<pair<int, int>, vector<pair<int, int>>> myDict;

	rowvec metaphe; // 1 x K

	double beta;

	Test_Module(double my_beta);

	void foo();

	void bar();
};


#endif /* SRC_DEBUG_TEST_FILES_TEST_MODULE_H_ */
