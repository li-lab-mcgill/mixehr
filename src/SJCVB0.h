#ifndef SJCVB0_H_
#define SJCVB0_H_

#include "JCVB0.h"
#include "PatientBatch.h"

class SJCVB0: public JCVB0 {
public:

	// stochastic inference tuning parameters
	int numOfBurnins;
	int batchsize;
	int batchNum;
	int batchId; // current training batch

	int rho_t;
	int tau;
	double kappa;
	double rho;

	int D_train;

	// key: typeId,pheId
	// value: PhenoParams struct
	unordered_map<pair<int,int>, PheParams*> pheParamsHat;
	unordered_map<pair<int,int>, LabParams*> labParamsHat;

	vector<PatientBatch*> *patientBatches;

	// key: typeId
	// value: 1 x K for K topics
	unordered_map<int, rowvec> topicCountsPerPheTypeHat; // T x K total topic counts per phe type

	// key: typeId,labId
	// value: 1 x K sum of eta for K topics
	unordered_map<pair<int,int>, rowvec> stateCountsPerLabTypeHat;

	SJCVB0();

	void train(bool updateHyper);
	void inferBatchPatParams();
	void updateMainParams();
	void updateParamSumsSVI();


	// update hyperparams
	void updateAlpha();
	void updateBeta();
	void updateZeta();
	void updatePsiHyper();
	void updateHyperParams();

	double trainLogLik();

	void testhash();
};

#endif /* SJCVB0_H_ */
