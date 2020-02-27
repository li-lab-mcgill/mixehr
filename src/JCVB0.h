#ifndef JCVB0_H_
#define JCVB0_H_

#include <map>
#include <vector>
#include <iostream>
#include <armadillo>

#include "Patient.h"
#include "PheParams.h"
#include "LabParams.h"


using namespace std;
using namespace arma;

class JCVB0 {
public:

	int K; //Num of Topics

	bool mar;
	bool svi;
	bool imp;

	unordered_map<int,int> numOfPhes;
	unordered_map<int,int> numOfLabs;

	int iterations;

	int C_train; // total no of EHR tokens in training patient set
	int C_tar; // total no of *target* tokens in test patient set

	// key: typeId,pheId
	// value: PhenoParams struct
	unordered_map<pair<int,int>, PheParams*> pheParams;
	unordered_map<pair<int,int>, LabParams*> labParams;


	// key: typeId
	// value: 1 x K for K topics
	unordered_map<int, rowvec> topicCountsPerPheType; // T x K total topic counts per phe type

	// key: typeId,labId
	// value: 1 x K sum of eta for K topics
	unordered_map<pair<int,int>, rowvec> stateCountsPerLabType;


	// sum of hyperparameters saved for latter computations
	rowvec alpha;
	double alphaSum;


	unordered_map<int, double> betaSum;
	unordered_map<int, double> sumLogGammaBeta;

	unordered_map<pair<int,int>, double> zetaSum;
	unordered_map<pair<int,int>, double> sumLogGammaZeta;

	int D_train;
	int D_test;
	int D_impute; // number of patients to impute

	vector<Patient> *trainPats;
	vector<Patient> *testPats;
	vector<Patient> *imputeTargetPats;

	void initialize(
			unordered_map<int,int> pheCnt,
			unordered_map<int,int> labCnt,
			int numOfTopics, int maxiter,
			unordered_map<pair<int,int>, PheParams*> pheParamsMap,
			unordered_map<pair<int,int>, LabParams*> labParamsMap);

	int inferTestPatMetaphe_maxiter_intermediateRun; // during training for monitoring purpose (fast the better)
	int inferTestPatMetaphe_maxiter_finalRun; // after training for evaluation purpose (accurate the better)

	void inferPatParams(vector<Patient>::iterator pat, int maxiter);

	void inferAllPatParams(int trainIter=1);
	virtual void updateParamSums();
	virtual void updateMainParams();

	void inferTestPatParams_finalRun();

	// not needed for learning because psi is integrated out
	// useful for topic discovery
	void updatePsi();


	// update hyperparams
	virtual void updateAlpha();
	virtual void updateBeta();
	virtual void updateZeta();
	virtual void updatePsiHyper();
	virtual void updateHyperParams();


	map<int, vector<int>> pheIds;
	map<int, vector<int>> labIds;
	void sortID(); // sorted id for outputs

	void showParams();
	void normalizeParams();

	virtual ~JCVB0();
	virtual void train(bool updateHyper);
	virtual double trainLogLik();

	virtual rowvec trainLogLik_breakdowns(); // model diagnostics

	double predictLogLik();
};

#endif /* JCVB0_H_ */







