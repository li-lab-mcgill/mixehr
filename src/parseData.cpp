#include "MixEHR.h"
#include "Patient.h"
#include "JCVB0.h"
#include "PheParams.h"

#include <armadillo>

using namespace std;
using namespace arma;


void MixEHR::parseMetaInfo() {

	// parse meta data file
	ifstream metafile(metaFile.c_str());

	int typeId,pheId,stateNum;

	numOfPheTypes = 0;
	numOfLabTypes = 0;

	int oldTypeId = -1;

	while(metafile >> typeId >> pheId >> stateNum) {

//		cout << "typeId: " << typeId << "; pheId: " << pheId << "; stateNum: " << stateNum << endl;

		if(oldTypeId!=typeId) { // new data type found

			if(stateNum > 1) {
				numOfLabTypes++;
				isLabType[typeId] = true;
			} else {
				numOfPheTypes++;
				isLabType[typeId] = false;
			}

			oldTypeId=typeId;
		}

		if(stateNum > 1) {

			if(oldTypeId!=typeId) { // new data type found
				numOfLabTypes++;
				isLabType[typeId] = true;
			}

			LabParams *labPar = new LabParams();

			labPar->V = stateNum;

			labPar->eta = randu<mat>(stateNum, numOfTopics);

			// lab state prob
			labPar->zeta = ones<vec>(stateNum)/stateNum;

			// missing indicator prob
			labPar->observedCnt = zeros<rowvec>(numOfTopics);
			labPar->missingCnt = zeros<rowvec>(numOfTopics);

			(*labParamsMap)[make_pair(typeId,pheId)] = labPar;

			if(numOfLabs.find(typeId)==numOfLabs.end()) {

				numOfLabs[typeId] = 1;

			} else {

				numOfLabs[typeId]++;
			}
		} else {
			if(oldTypeId!=typeId) { // new data type found
				numOfPheTypes++;
				isLabType[typeId] = false;
			}

			PheParams *phePar = new PheParams();

			phePar->phi = randu<rowvec>(numOfTopics);

			(*pheParamsMap)[make_pair(typeId, pheId)] = phePar;

			if(numOfPhes.find(typeId)==numOfPhes.end()) {

				numOfPhes[typeId] = 1;

			} else {

				numOfPhes[typeId]++;
			}
		}
		oldTypeId=typeId;
	}

	metafile.close();
}


void MixEHR::parseTrainData(JCVB0* jcvb0) {

	// parse patient data file
	ifstream datafile(trainDataFile.c_str());

	int typeId,pheId;

	int patId,stateId,freq;

	bool eof = false;

	if(!(datafile >> patId >> typeId >> pheId >> stateId >> freq)) {
		eof = true;
	}

//	printf("patId: %d; typeId: %d; pheId: %d; stateId: %d; freq: %d\n", patId,typeId,pheId,stateId,freq);

	while(!eof) {

		unordered_map<pair<int, int>, int>* pheMap = new unordered_map<pair<int, int>, int>();

		unordered_map<pair<int, int>, vector<pair<int, int>>>* labMap = new unordered_map<pair<int, int>, vector<pair<int, int>>>();

		int oldPatId = patId;

		while(patId == oldPatId) {

			if(isLabType[typeId]) {

				if(jcvb0->labParams[make_pair(typeId,pheId)]->V <= stateId) {

					cout << "typeId: " << typeId << "; labId: " << pheId << endl;
					cout << "jcvb0->labParams[make_pair(typeId,pheId)]->V: " << jcvb0->labParams[make_pair(typeId,pheId)]->V << endl;
					cout << "stateId: " << stateId << endl;

					throw::runtime_error("invalid stateId in train patient data");
				}

				(*labMap)[make_pair(typeId,pheId)].push_back(make_pair(stateId,freq));

			} else {

				(*pheMap)[make_pair(typeId,pheId)] = freq;
			}

			if(!(datafile >> patId >> typeId >> pheId >> stateId >> freq)) {

				eof = true;
				break;
			}

//			printf("patId: %d; typeId: %d; pheId: %d; stateId: %d; freq: %d\n", patId,typeId,pheId,stateId,freq);
		}

		Patient* newPat = new Patient(oldPatId,
				*pheMap,
				*labMap,
				numOfTopics,
				jcvb0->labParams);

		jcvb0->trainPats->push_back(*newPat);

		jcvb0->C_train += newPat->Cj_train;

		// printf("%.3f\n", (double)(jcvb0->trainPats->size() + jcvb0->testPats->size())/numOfPats);
	}

	datafile.close();

	jcvb0->D_train = jcvb0->trainPats->size();

	cout << "numOfPheTypes: " << numOfPheTypes << endl;
	cout << "numOfLabTypes: " << numOfLabTypes << endl;
	cout << "numOfPhenotypes: " << jcvb0->pheParams.size() << endl;
	cout << "numOfLabTests: " << jcvb0->labParams.size() << endl;
	cout << "numOfPats: " << jcvb0->D_train << endl;
	cout << "C_train: " << jcvb0->C_train << endl;
	printf("Training data file parsing completed.\n");
	cout << "--------------------" << endl;

//	throw::runtime_error("parseData");
}


void MixEHR::parseTestData(JCVB0* jcvb0) {

	// parse patient data file
	ifstream datafile(testDataFile.c_str());

	int patId,typeId,pheId,stateId,freq,obs;

	bool eof = false;


	if(!(datafile >> patId >> typeId >> pheId >> stateId >> freq >> obs)) {
		eof = true;
	}

//	printf("patId: %d; typeId: %d; pheId: %d; stateId: %d; freq: %d; obs: %d\n", patId,typeId,pheId,stateId,freq,obs);

	while(!eof) {

		unordered_map<pair<int, int>, int>* pheMap = new unordered_map<pair<int, int>, int>();
		unordered_map<pair<int, int>, vector<pair<int, int>>>* labMap = new unordered_map<pair<int, int>, vector<pair<int, int>>>();
		unordered_map<pair<int, int>, bool>* obsMap = new unordered_map<pair<int, int>, bool>();

		for(unordered_map<pair<int,int>, LabParams*>::iterator iter = jcvb0->labParams.begin(); iter != jcvb0->labParams.end(); iter++) {
			(*obsMap)[iter->first] = false;
		}

		int oldPatId = patId;

		while(patId == oldPatId) {

			if(isLabType[typeId]) {

				if(jcvb0->labParams[make_pair(typeId,pheId)]->V <= stateId) {

					cout << "In MixEHR::parseTestData" << endl;
					cout << "jcvb0->labParams[make_pair(typeId,pheId)]->V: " << jcvb0->labParams[make_pair(typeId,pheId)]->V << endl;
					cout << "stateId: " << stateId << endl;

					throw::runtime_error("invalid stateId in train patient data");
				}

				(*labMap)[make_pair(typeId,pheId)].push_back(make_pair(stateId,freq));
				(*obsMap)[make_pair(typeId,pheId)] = obs;

			} else {

				(*pheMap)[make_pair(typeId,pheId)] = freq;
			}
			if(!(datafile >> patId >> typeId >> pheId >> stateId >> freq >> obs)) {

				eof = true;
				break;
			}

//			printf("patId: %d; typeId: %d; pheId: %d; stateId: %d; freq: %d; obs: %d\n", patId,typeId,pheId,stateId,freq,obs);
		}

		Patient* newPat = new Patient(oldPatId,
				*pheMap,
				*labMap,
				numOfTopics,
				jcvb0->labParams);


		newPat->isTestPat = true;
		newPat->obsDict = *obsMap;

		if(evalTargetTypeOnly) {
			newPat->assignTargetView(targetTypeId);
		} else {
			if(!evalLabOnly) {
				newPat->assignTargetPhenotypes();
			}
			newPat->assignTargetLabTests(missingLabOnly, observedLabOnly);
		}
		jcvb0->testPats->push_back(*newPat);
		jcvb0->C_tar += newPat->Cj_tar;
	}

	datafile.close();

	jcvb0->D_test = jcvb0->testPats->size();

	printf("testPats: %d\n", (int)jcvb0->D_test);
	cout << "C_tar: " << jcvb0->C_tar << endl;
	printf("Testing data file parsing completed.\n");
	cout << "--------------------" << endl;
}


void MixEHR::parsePhi() {

	// parse trained state eta parameters
	string trainedModelState = trainedModelPrefix + "_phi.csv";

	cout << "--------------------" << endl;
	cout << "Parsing phi: " << trainedModelState << " ... ";

	// parse trained phi file
	ifstream modelfile(trainedModelState.c_str());

	if(!modelfile.good()) {
		cout << trainedModelState << endl;
		throw runtime_error("file does not exist");
	}

	string tmp;

	getline(modelfile, tmp, ',');
	int typeId = atoi(tmp.c_str());
	getline(modelfile, tmp, ',');
	int pheId = atoi(tmp.c_str());

	while(modelfile.good()) {

		for(int k=0; k<numOfTopics-1; k++) {

			getline(modelfile, tmp, ',');

			(*pheParamsMap)[make_pair(typeId,pheId)]->phi(k) = atof(tmp.c_str());

//			if(typeId == 1 && pheId == 30774) {
//				cout << "typeId: " << typeId << "; " << "pheId: " << pheId << "; " << "k: " <<
//						k << "; " << (*pheParamsMap)[make_pair(typeId,pheId)]->phi(k) << endl;
//			}
		}

//		if(typeId == 1 && pheId == 30774)
//			throw runtime_error("stop here");

		getline(modelfile, tmp, '\n');

		(*pheParamsMap)[make_pair(typeId,pheId)]->phi(numOfTopics-1) = atof(tmp.c_str());

//		cout << "typeId: " << typeId << "; " << "pheId: " << pheId << "; " << "k: " <<
//				numOfTopics-1 << "; " << (*pheParamsMap)[make_pair(typeId,pheId)]->phi(numOfTopics-1) << endl;

		getline(modelfile, tmp, ',');
		typeId = atoi(tmp.c_str());
		getline(modelfile, tmp, ',');
		pheId = atoi(tmp.c_str());
	}

	modelfile.close();

	cout << "done." << endl;
}



void MixEHR::parseEta() {

	// parse trained state eta parameters
	string trainedModelState = trainedModelPrefix + "_eta.csv";

	cout << "--------------------" << endl;
	cout << "Parsing eta: " << trainedModelState << " ... ";

	// parse trained phi file
	ifstream modelfile(trainedModelState.c_str());

	if(!modelfile.good()) {
		cout << trainedModelState << endl;
		throw runtime_error("file does not exist");
	}

	string tmp;

	getline(modelfile, tmp, ',');
	int typeId = atoi(tmp.c_str());
	getline(modelfile, tmp, ',');
	int labId = atoi(tmp.c_str());

	while(modelfile.good()) {

		getline(modelfile, tmp, ',');

		int stateId = atoi(tmp.c_str());

		for(int k=0; k<numOfTopics-1; k++) {

			getline(modelfile, tmp, ',');

			(*labParamsMap)[make_pair(typeId,labId)]->eta(stateId, k) = atof(tmp.c_str());
		}

		getline(modelfile, tmp, '\n');

		(*labParamsMap)[make_pair(typeId,labId)]->eta(stateId, numOfTopics-1) = atof(tmp.c_str());

		getline(modelfile, tmp, ',');
		typeId = atoi(tmp.c_str());
		getline(modelfile, tmp, ',');
		labId = atoi(tmp.c_str());
	}

	modelfile.close();

	cout << "done." << endl;
}

rowvec MixEHR::parseAlpha() {

	string tmp;

	// parse trained alpha parameter file
	string trainedModelAlpha = trainedModelPrefix + "_alpha.csv";

	cout << "--------------------" << endl;
	cout << "Parsing alpha: " << trainedModelAlpha << " ... ";

	ifstream modelfile(trainedModelAlpha.c_str());

	if(!modelfile.good()) {
		cout << trainedModelAlpha << endl;
		throw runtime_error("file does not exist");
	}

	rowvec alpha = zeros<rowvec>(numOfTopics);

	for(int k=0; k<numOfTopics-1; k++) {

		getline(modelfile, tmp, ',');

		alpha(k) = atof(tmp.c_str());
	}

	getline(modelfile, tmp, '\n');

	alpha(numOfTopics-1) = atof(tmp.c_str());

	modelfile.close();

	cout << "done." << endl;

	return alpha;
}


void MixEHR::parseBeta() {

	// parse trained alpha parameter file
	string trainedModelBeta= trainedModelPrefix + "_beta.csv";

	cout << "--------------------" << endl;
	cout << "Parsing beta: " << trainedModelBeta << " ... ";

	ifstream modelfile(trainedModelBeta.c_str());

	if(!modelfile.good()) {
		cout << trainedModelBeta << endl;
		throw runtime_error("file does not exist");
	}

	string tmp;

	getline(modelfile, tmp, ',');
	int typeId = atoi(tmp.c_str());
	getline(modelfile, tmp, ',');
	int pheId = atoi(tmp.c_str());

	while(modelfile.good()) {

		getline(modelfile, tmp, '\n');

		(*pheParamsMap)[make_pair(typeId,pheId)]->beta = atof(tmp.c_str());

		getline(modelfile, tmp, ',');
		typeId = atoi(tmp.c_str());
		getline(modelfile, tmp, ',');
		pheId = atoi(tmp.c_str());
	}

	modelfile.close();

	cout << "done." << endl;
}


void MixEHR::parseZeta() {

	// parse trained state eta parameters
	string trainedModelState = trainedModelPrefix + "_zeta.csv";

	cout << "--------------------" << endl;
	cout << "Parsing zeta: " << trainedModelState << " ... ";

	// parse trained phi file
	ifstream modelfile(trainedModelState.c_str());

	if(!modelfile.good()) {
		cout << trainedModelState << endl;
		throw runtime_error("file does not exist");
	}

	//	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParamsMap->begin(); iter != labParamsMap->end(); iter++)
	//		iter->second->zeta.zeros();

	string tmp;

	getline(modelfile, tmp, ',');
	int typeId = atoi(tmp.c_str());
	getline(modelfile, tmp, ',');
	int labId = atoi(tmp.c_str());
	getline(modelfile, tmp, ',');
	int stateId = atoi(tmp.c_str());


	while(modelfile.good()) {

		getline(modelfile, tmp, '\n');

		(*labParamsMap)[make_pair(typeId,labId)]->zeta(stateId) = atof(tmp.c_str());

		getline(modelfile, tmp, ',');
		typeId = atoi(tmp.c_str());
		getline(modelfile, tmp, ',');
		labId = atoi(tmp.c_str());
		getline(modelfile, tmp, ',');
		stateId = atoi(tmp.c_str());
	}

	modelfile.close();

	cout << "done." << endl;
}

// file format:typeId, labId, a, b
void MixEHR::parsePsiHyper() {

	string trainedModelState = trainedModelPrefix + "_psiHyper.csv";

	cout << "--------------------" << endl;
	cout << "Parsing psi-hyper: " << trainedModelState << " ... ";

	ifstream modelfile(trainedModelState.c_str());

	if(!modelfile.good()) {
		cout << trainedModelState << endl;
		throw runtime_error("file does not exist");
	}

	string tmp;

	getline(modelfile, tmp, ',');
	int typeId = atoi(tmp.c_str());
	getline(modelfile, tmp, ',');
	int labId = atoi(tmp.c_str());

	while(modelfile.good()) {

		getline(modelfile, tmp, ',');

		(*labParamsMap)[make_pair(typeId,labId)]->a = atof(tmp.c_str());

		getline(modelfile, tmp, '\n');

		(*labParamsMap)[make_pair(typeId,labId)]->b = atof(tmp.c_str());

		getline(modelfile, tmp, ',');
		typeId = atoi(tmp.c_str());
		getline(modelfile, tmp, ',');
		labId = atoi(tmp.c_str());
	}

	modelfile.close();

	cout << "done." << endl;
}


// parse two files: _observedCnt.csv, _missingCnt.csv
void MixEHR::parseCounts() {

	string trainedModelState;

	for(int psiPars = 1; psiPars <= 2; psiPars++) {

		if(psiPars==1) {

			trainedModelState = trainedModelPrefix + "_obscnt.csv";
			cout << "--------------------" << endl;
			cout << "Parsing obscnt: " << trainedModelState << " ... ";

		} else if(psiPars==2) {

			trainedModelState = trainedModelPrefix + "_miscnt.csv";
			cout << "--------------------" << endl;
			cout << "Parsing miscnt: " << trainedModelState << " ... ";
		}

		ifstream modelfile(trainedModelState.c_str());

		if(!modelfile.good()) {
			cout << trainedModelState << endl;
			throw runtime_error("file does not exist");
		}

		string tmp;

		getline(modelfile, tmp, ',');
		int typeId = atoi(tmp.c_str());
		getline(modelfile, tmp, ',');
		int labId = atoi(tmp.c_str());

		while(modelfile.good()) {

			for(int k=0; k<numOfTopics-1; k++) {

				getline(modelfile, tmp, ',');

				if(psiPars==1) {
					(*labParamsMap)[make_pair(typeId,labId)]->observedCnt(k) = atof(tmp.c_str());
				} else if(psiPars==2) {
					(*labParamsMap)[make_pair(typeId,labId)]->missingCnt(k) = atof(tmp.c_str());
				}
			}

			getline(modelfile, tmp, '\n');

			if(psiPars==1) {
				(*labParamsMap)[make_pair(typeId,labId)]->observedCnt(numOfTopics-1) = atof(tmp.c_str());
			} else if(psiPars==2) {
				(*labParamsMap)[make_pair(typeId,labId)]->missingCnt(numOfTopics-1) = atof(tmp.c_str());
			}

			getline(modelfile, tmp, ',');
			typeId = atoi(tmp.c_str());
			getline(modelfile, tmp, ',');
			labId = atoi(tmp.c_str());
		}

		modelfile.close();

		cout << "done." << endl;
	}
}



JCVB0* MixEHR::parseTrainedModelFiles() {

	if(trainedModelPrefix.length() == 0) {
		throw runtime_error("trainedModelPrefix is undefined");
	}

	parsePhi();

	parseBeta();

	parseEta();

	parseZeta();

	parsePsiHyper();

	parseCounts();

	JCVB0* jcvb0 = new JCVB0();

	jcvb0->initialize(
			numOfPhes, numOfLabs,
			numOfTopics, numOfIters,
			*pheParamsMap, *labParamsMap);

	jcvb0->alpha = parseAlpha();

	jcvb0->updateParamSums();

	jcvb0->normalizeParams();

	jcvb0->mar = mar;

	jcvb0->imp = imputeNewPatientData || inferTrainPatMetaphe_only;

	return jcvb0;
}


JCVB0* MixEHR::parseNewData() {

	JCVB0* jcvb0 = parseTrainedModelFiles();

	cout << "--------------------" << endl;
	printf("Trained model data files parsing completed.\n");
	cout << "--------------------" << endl;

//	jcvb0->showParams();

	// parse patient data file
	ifstream datafile(newDatafile.c_str());

	if(!datafile.is_open()) {
		cout << newDatafile << endl;
		throw::runtime_error("Cannot open file");
	}

	int patId,typeId,pheId,stateId,freq;

	bool eof = false;

	if(!(datafile >> patId >> typeId >> pheId >> stateId >> freq)) {
		eof = true;
	}

//	printf("patId: %d; typeId: %d; pheId: %d; stateId: %d; freq: %d\n", patId,typeId,pheId,stateId,freq);

	while(!eof) {

		unordered_map<pair<int, int>, int>* pheMap = new unordered_map<pair<int, int>, int>();

		unordered_map<pair<int, int>, vector<pair<int, int>>>* labMap = new unordered_map<pair<int, int>, vector<pair<int, int>>>();

		int oldPatId = patId;

		while(patId == oldPatId) {

			if(isLabType[typeId]) {

				(*labMap)[make_pair(typeId,pheId)].push_back(make_pair(stateId,freq));

			} else {

				(*pheMap)[make_pair(typeId,pheId)] = freq;
			}

			if(!(datafile >> patId >> typeId >> pheId >> stateId >> freq)) {

				eof = true;
				break;
			}
		}

		Patient* newPat = new Patient(oldPatId,
				*pheMap,
				*labMap,
				numOfTopics,
				jcvb0->labParams);

		newPat->isTestPat = true;

		jcvb0->testPats->push_back(*newPat);

		jcvb0->C_train += newPat->Cj_train;
	}

	datafile.close();

	cout << "--------------------" << endl;
	printf("New patient data files parsing completed.\n");
	cout << "--------------------" << endl;

	cout << "numOfPheTypes: " << numOfPheTypes << endl;
	cout << "numOfLabTypes: " << numOfLabTypes << endl;
	cout << "numOfPhenotypes: " << jcvb0->pheParams.size() << endl;
	cout << "numOfLabTests: " << jcvb0->labParams.size() << endl;

	printf("testPats: %d\n", (int)jcvb0->testPats->size());
	cout << "C_train: " << jcvb0->C_train<< endl;

	cout << "--------------------" << endl;

	return jcvb0;
}


void MixEHR::parseImputeTargetsFile() {

	// parse meta data file
	ifstream targetsfile(imputeTargetsFile.c_str());

	int typeId,pheId;


	while(targetsfile >> typeId >> pheId) {

//		cout << "typeId: " << typeId << "; pheId: " << pheId << endl;

		unordered_map<pair<int,int>, PheParams*>::const_iterator phe_hasit = pheParamsMap->find(make_pair(typeId,pheId));
		unordered_map<pair<int,int>, LabParams*>::const_iterator lab_hasit = labParamsMap->find(make_pair(typeId,pheId));

		if(phe_hasit != pheParamsMap->end()) {
			pheImputeTargets.push_back(make_pair(typeId, pheId));
		} else if (lab_hasit != labParamsMap->end()) {
			labImputeTargets.push_back(make_pair(typeId, pheId));
		} else {
			cout << "typeId: " << typeId << "; pheId: " << pheId << endl;
			throw::runtime_error("target phenotype is not defined in the metainfo file");
		}

	}

	targetsfile.close();
}








void MixEHR::parseImputePatDataFile(JCVB0* jcvb0) {

	// parse patient data file
	ifstream datafile(imputePatDataFile.c_str());

	int patId,typeId,pheId,stateId,freq;

	bool eof = false;

	if(!(datafile >> patId >> typeId >> pheId >> stateId >> freq)) {
		eof = true;
	}

//	printf("patId: %d; typeId: %d; pheId: %d; stateId: %d; freq: %d\n", patId,typeId,pheId,stateId,freq);

	while(!eof) {

		unordered_map<pair<int, int>, int>* pheMap = new unordered_map<pair<int, int>, int>();
		unordered_map<pair<int, int>, vector<pair<int, int>>>* labMap = new unordered_map<pair<int, int>, vector<pair<int, int>>>();

		int oldPatId = patId;

		while(patId == oldPatId) {

			if(isLabType[typeId]) {

				if(jcvb0->labParams[make_pair(typeId,pheId)]->V <= stateId) {

					cout << "In MixEHR::parseTestData" << endl;
					cout << "jcvb0->labParams[make_pair(typeId,pheId)]->V: " << jcvb0->labParams[make_pair(typeId,pheId)]->V << endl;
					cout << "stateId: " << stateId << endl;

					throw::runtime_error("invalid stateId in train patient data");
				}

				(*labMap)[make_pair(typeId,pheId)].push_back(make_pair(stateId,freq));

			} else {

				(*pheMap)[make_pair(typeId,pheId)] = freq;
			}
			if(!(datafile >> patId >> typeId >> pheId >> stateId >> freq)) {
				eof = true;
				break;
			}

//			printf("patId: %d; typeId: %d; pheId: %d; stateId: %d; freq: %d\n", patId,typeId,pheId,stateId,freq);
		}

		Patient* newPat = new Patient(oldPatId,
				*pheMap,
				*labMap,
				numOfTopics,
				jcvb0->labParams);

		jcvb0->imputeTargetPats->push_back(*newPat);
	}

	datafile.close();

	jcvb0->D_impute = jcvb0->imputeTargetPats->size();

	printf("imputeTargetPats: %d\n", (int)jcvb0->D_impute);
	printf("Impute target data file parsing completed.\n");
	cout << "--------------------" << endl;
}




















