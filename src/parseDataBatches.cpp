#include "MixEHR.h"

using namespace std;
using namespace arma;

SJCVB0* MixEHR::parseTrainDataBatches() {

	// reading patient data from file
	ifstream datafile(trainDataFile.c_str());

	int typeId,pheId;

	SJCVB0* sjcvb0 = new SJCVB0();

	sjcvb0->initialize(
			numOfPhes, numOfLabs,
			numOfTopics, numOfIters,
			*pheParamsMap, *labParamsMap);

	sjcvb0->pheParamsHat = unordered_map<pair<int,int>, PheParams*>();
	sjcvb0->labParamsHat = unordered_map<pair<int,int>, LabParams*>();

	for(unordered_map<pair<int,int>, PheParams*>::iterator iter = sjcvb0->pheParams.begin(); iter != sjcvb0->pheParams.end(); iter++) {
		pair<int,int> pheId = iter->first;
		sjcvb0->pheParamsHat[pheId] = new PheParams();
		sjcvb0->pheParamsHat[pheId]->phi = iter->second->phi; // 1 x K
	}

	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = sjcvb0->labParams.begin(); iter != sjcvb0->labParams.end(); iter++) {

		pair<int,int> labId = iter->first;
		sjcvb0->labParamsHat[iter->first] = new LabParams();
		sjcvb0->labParamsHat[labId]->eta = iter->second->eta; // V x K
		sjcvb0->labParamsHat[labId]->zeta = iter->second->zeta; // V x 1
		sjcvb0->labParamsHat[labId]->observedCnt = iter->second->observedCnt; // 1 x K
		sjcvb0->labParamsHat[labId]->missingCnt = iter->second->missingCnt; // 1 x K
	}


	sjcvb0->mar = mar;

	sjcvb0->svi = true;

	sjcvb0->inferTestPatMetaphe_maxiter_finalRun = inferPatParams_maxiter;

	sjcvb0->patientBatches = new vector<PatientBatch*>();

	sjcvb0->numOfBurnins = numOfBurnins;

	sjcvb0->kappa = kappaStepsize;

	sjcvb0->topicCountsPerPheTypeHat = sjcvb0->topicCountsPerPheType;

	sjcvb0->stateCountsPerLabTypeHat = sjcvb0->stateCountsPerLabType;

	int patId,stateId,freq;

	bool eof = false;

	if(!(datafile >> patId >> typeId >> pheId >> stateId >> freq)) {
		eof = true;
	}


//	printf("patId: %d; typeId: %d; pheId: %d; stateId: %d; freq: %d\n", patId,typeId,pheId,stateId,freq);

	while(!eof) {

		PatientBatch* patBatch = new PatientBatch();

		patBatch->M = 0;

		int batchCounter = 1;

		while(batchCounter <= batchsize && !eof) {

			unordered_map<pair<int, int>, int>* pheMap = new unordered_map<pair<int, int>, int>();

			unordered_map<pair<int, int>, vector<pair<int, int>>>* labMap = new unordered_map<pair<int, int>, vector<pair<int, int>>>();

			int oldPatId = patId;

			while(patId == oldPatId) {

				if(isLabType[typeId]) {

					if(sjcvb0->labParams[make_pair(typeId,pheId)]->V <= stateId) {

						cout << "jcvb0->labParams[make_pair(typeId,pheId)]->V: " <<
								sjcvb0->labParams[make_pair(typeId,pheId)]->V << endl;

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

//				printf("patId: %d; typeId: %d; pheId: %d; stateId: %d; freq: %d\n", patId,typeId,pheId,stateId,freq);
			}

			Patient* newPat = new Patient(oldPatId,
					*pheMap,
					*labMap,
					numOfTopics,
					sjcvb0->labParams);

			sjcvb0->D_train++;

			patBatch->M += newPat->Cj_train;

			patBatch->patVector->push_back(*newPat);

			sjcvb0->C_train += newPat->Cj_train;

			batchCounter++;
		}

		if(patBatch->patVector->size() > 0) {

			sjcvb0->patientBatches->push_back(patBatch);
		}
	}

	datafile.close();

	cout << "numOfPheTypes: " << numOfPheTypes << endl;
	cout << "numOfLabTypes: " << numOfLabTypes << endl;
	cout << "numOfPhenotypes: " << sjcvb0->pheParams.size() << endl;
	cout << "numOfLabTests: " << sjcvb0->labParams.size() << endl;
	cout << "--------------------" << endl;
	cout << "trainPats: " << sjcvb0->D_train << endl;
	cout << "C_train: " << sjcvb0->C_train << endl;
	cout << "train batchNum: " << sjcvb0->patientBatches->size() << endl;
	cout << "C_tar: " << sjcvb0->C_tar << endl;
	printf("Data file parsing completed.\n");
	cout << "--------------------" << endl;
	cout << "batchsize: " << batchsize << endl;
	cout << "numOfBurnins: " << sjcvb0->numOfBurnins << endl;
	cout << "kappaStepsize: " << kappaStepsize<< endl;
	cout << "--------------------" << endl;


	return sjcvb0;
}




















