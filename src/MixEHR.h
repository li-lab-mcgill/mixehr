#ifndef MixEHR_H_
#define MixEHR_H_

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <armadillo>
#include <sstream>
#include <random>
#include <iterator>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace arma;

#include "PheParams.h"
#include "LabParams.h"
#include "Patient.h"
#include "PatientBatch.h"
#include "JCVB0.h"
#include "SJCVB0.h"


class MixEHR {
public:

	int numOfPats;
	int numOfTopics;
	int numOfIters;
	int numOfPheTypes;
	int numOfLabTypes;
	int k_nearest_neighbors;

	unordered_map<int,int> numOfPhes;
	unordered_map<int,int> numOfLabs;

	string metaFile;
	string trainDataFile;
	string testDataFile;
	string imputeTargetsFile; // for impute targets in new patients
	string imputePatDataFile; // new patients data for imputation

	unordered_map<pair<int,int>, PheParams*>* pheParamsMap;
	unordered_map<pair<int,int>, LabParams*>* labParamsMap;

	vector<pair<int,int>> pheImputeTargets;
	vector<pair<int,int>> labImputeTargets;

	unordered_map<int,bool> isLabType;

	string newDatafile;

	string trainedModelPrefix;

	bool inferNewPatientMetaphe;

	bool outputIntermediates;

	bool imputeNewPatientData; // impute new patient data

	int inferPatParams_maxiter;

	bool inferTrainPatMetaphe_only; // infer train patient mix for imputation
	string trainPatMetapheFile; // file to save train patient mix
	string trainPatIdFile; // train patient ID to match with the trainPatMetaphe matrix rows

	// inference method
	string inference;

	// SJCVB0
	int numOfBurnins;
	int batchsize;
	double kappaStepsize;

	// evaluation
	double testPatsFrac;

	bool mar;
	bool evalLabOnly;
	bool missingLabOnly;
	bool observedLabOnly;

	int targetTypeId;
	bool evalTargetTypeOnly;

	vec logTrainLik;
	mat logTrainLik_breakdowns; // (model diagnostics) elbo breakdowns to check convergence issues

	vec logPredLik;

	vec trainTime;

	string outPrefix_trainData;
	string outPrefix_testData;

	string output_dir;

	int maxcores;

	double loglikdiff_thres;

	MixEHR(string datafile_train,
			string datafile_test,
			string metafile,
			int topics,
			int maxiter, double diff_thres,
			string inferMethod,
			double testSetFrac,
			int batchSize, int numOfBurnIns, double stepsize,
			string newDataPath, string modelPrefix,
			bool inferNewPatientParams,

			bool inferTrainPatientMix,
			string trainPatMixFile,
			string trainPatIdFile,

			string imputeTargetsFilePath,
			string imputePatDataFilePath,
			int knn,
			bool saveIntermediates,
			bool missingAtRandom,
			bool evaluateLabOnly,
			bool testMissingLabOnly,
			bool testObservedLabOnly,
			int targetViewId,
			bool evalTargetViewOnly,
			int inferTestPatThetaMaxiter,
			string output_path,
			int maxthreads);

	void parseMetaInfo();
	JCVB0* initialize_infer();
	void parseTrainData(JCVB0* jcvb0); // change JCVB0 within function
	void parseTestData(JCVB0* jcvb0); // change JCVB0 within function

	void parseImputeTargetsFile();
	void parseImputePatDataFile(JCVB0* jcvb0);

	// parse model files
	void parsePhi();
	void parseEta();

	// parse hyperparameter files
	rowvec parseAlpha();
	void parseBeta();
	void parseZeta();
	void parsePsiHyper();
	void parseCounts();

	SJCVB0* parseTrainDataBatches();
	//	void trainOnline();

	void exportResults(JCVB0* jcvb0, int iter, bool verbose);

	void exportLogLik(int iter);

	void exportTrainTime(int iter);

	void exportLogLik_breakdowns(); // output all elbo breakdown elements per iteration

	void exportTestPatData(JCVB0* jcvb0);

	JCVB0* parseTrainedModelFiles();

	void parseImputeTargetList(); // get a list of impute targets (i.e., a subset of variables in metainfo)

	JCVB0* parseNewData();

	void inferNewPatMetaphe(JCVB0* jcvb0, bool output_to_file=false);

	void inferTrainPatMetaphe(); // infer and save the train pat mix for imputing test patients
	void imputeNewPheData(JCVB0* jcvb0, int nearestNeighborK);
	void imputeNewLabData(JCVB0* jcvb0, int nearestNeighborK);
	void imputeNewPatData(int nearestNeighborK);
};

#endif
















