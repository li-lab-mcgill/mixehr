#ifndef PATIENT_H_
#define PATIENT_H_

#include <map>
#include <armadillo>

#include "pairkeyhash.h"
#include "PheParams.h"
#include "LabParams.h"

using namespace std;
using namespace arma;

class Patient {
public:

	int patId;

	bool isTestPat;

	rowvec metaphe; // 1 x K
	rowvec metaphe_normalized; // 1 x K

	// key: <typeId,pheId>
	// value: freq
	unordered_map<pair<int, int>, int> pheDict;


	// key: <typeId,labId>
	// value: <stateId, freq>
	// patient can have both normal and abnormal states
	unordered_map<pair<int, int>, vector<pair<int, int>>> labDict;

	unordered_map<pair<int, int>, bool> obsDict;

	// key: <typeId, pheId>
	// value: 1 x K
	unordered_map<pair<int, int>, rowvec> gamma;


	// key: <typeId, labId>
	// value: V x K
	unordered_map<pair<int, int>, mat> lambda;


	Patient(int id,
			unordered_map<pair<int, int>, int> pheMap,
			unordered_map<pair<int, int>, vector<pair<int, int>>> labMap,
			int K,
			unordered_map<pair<int,int>, LabParams*> labParams);

	~Patient();

	void assignTargetPhenotypes();
	void assignTargetLabTests(bool missingLabOnly, bool observedLabOnly);
	void assignTargetView(int targetViewId);

	double tarPheFrac;
	double tarLabFrac;
	int Cj;
	int Cj_tar;
	int Cj_train;

	unordered_map<pair<int,int>, bool> isTestPhe;
	unordered_map<pair<int,int>, bool> isTestLab;
};

#endif /* PATIENT_H_ */

