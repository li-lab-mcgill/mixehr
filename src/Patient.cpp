#include <armadillo>

#include "Patient.h"

using namespace arma;

Patient::Patient(int id,
		unordered_map<pair<int, int>, int> pheMap,
		unordered_map<pair<int, int>, vector<pair<int, int>>> labMap,
		int K,
		unordered_map<pair<int,int>, LabParams*> labParams)
{
	patId = id;
	pheDict = pheMap;
	labDict = labMap;
	tarPheFrac = 0.5;
	tarLabFrac = 0.5;

	metaphe = randu<rowvec>(K);
	metaphe_normalized = metaphe/accu(metaphe);

	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++) {

		obsDict[iter->first] = false;
	}

	Cj = 0;

	isTestPat = false;

	for(unordered_map<pair<int, int>, int>::iterator iter = pheMap.begin(); iter != pheMap.end(); iter++) {

		isTestPhe[iter->first] = false;

		Cj += iter->second;
	}

	for(unordered_map<pair<int, int>, vector<pair<int,int>>>::iterator iter = labMap.begin(); iter != labMap.end(); iter++) {

		obsDict[iter->first] = true;

		isTestLab[iter->first] = false;

		for(vector<pair<int,int>>::iterator iter2 = iter->second.begin(); iter2 != iter->second.end(); iter2++) {

			Cj += iter2->second;
		}
	}


	Cj_train = Cj;
	Cj_tar = 0;
}

void Patient::assignTargetView(int targetViewId) {

	if(pheDict.size() > 0) {

		for(unordered_map<pair<int, int>, int>::iterator iter = pheDict.begin(); iter != pheDict.end(); iter++) {

			if(iter->first.first == targetViewId) {

				isTestPhe[iter->first] = true;

				Cj_tar += iter->second;

			} else {

				isTestPhe[iter->first] = false;
			}
		}
	}

	if(labDict.size() > 0) {

		for(unordered_map<pair<int, int>, vector<pair<int, int>>>::iterator iter = labDict.begin(); iter != labDict.end(); iter++) {

			if(iter->first.first == targetViewId) {

				isTestLab[iter->first] = true;

				for(vector<pair<int, int>>::iterator iter2 = iter->second.begin(); iter2 != iter->second.end(); iter2++) {

					Cj_tar += iter2->second;
				}

			} else {

				isTestLab[iter->first] = false;
			}
		}
	}
}


void Patient::assignTargetPhenotypes() {

	if(pheDict.size() > 0) {

		vector<pair<int,int>> obsPhe;

		for(unordered_map<pair<int, int>, int>::iterator iter = pheDict.begin(); iter != pheDict.end(); iter++) {

			obsPhe.push_back(iter->first);

			isTestPhe[iter->first] = false;
		}

		std::random_shuffle(obsPhe.begin(), obsPhe.end());

		vector<pair<int,int> >::const_iterator first_tarPhe = obsPhe.begin();

		vector<pair<int,int> >::const_iterator last_tarPhe = obsPhe.begin() + floor(tarPheFrac * obsPhe.size());

		vector<pair<int,int> > tmp(first_tarPhe, last_tarPhe);

		for(vector<pair<int, int>>::iterator iter = tmp.begin(); iter != tmp.end(); iter++) {

			Cj_tar += pheDict[*iter];

			isTestPhe[*iter] = true;
		}

		Cj_train -= Cj_tar;
	}
}


void Patient::assignTargetLabTests(bool missingLabOnly, bool observedLabOnly) {

	if(labDict.size() > 0) {

		if(missingLabOnly) { // assign test lab as missing lab

			for(unordered_map<pair<int, int>, vector<pair<int, int>>>::iterator iter = labDict.begin(); iter != labDict.end(); iter++) {

				if(!obsDict[iter->first]) {

					isTestLab[iter->first] = true;

					for(vector<pair<int, int>>::iterator iter2 = iter->second.begin(); iter2 != iter->second.end(); iter2++) {

						Cj_tar += iter2->second;
					}

				} else {
					isTestLab[iter->first] = false;
				}
			}

		} else {

			vector<pair<int,int>> obsLab;

			for(unordered_map<pair<int, int>, vector<pair<int, int>>>::iterator iter = labDict.begin(); iter != labDict.end(); iter++) {

				if(observedLabOnly) { // consider only observed lab for comparing nmar and mar model

					if(obsDict[iter->first]) {

						obsLab.push_back(iter->first);
					}

				} else { // consider all lab tests

					obsLab.push_back(iter->first);
				}

				isTestLab[iter->first] = false;
			}

			std::random_shuffle(obsLab.begin(), obsLab.end());

			vector<pair<int,int> >::const_iterator first_tarLab = obsLab.begin();

			vector<pair<int,int> >::const_iterator last_tarLab = obsLab.begin() + floor(tarLabFrac * obsLab.size());

			vector<pair<int,int> > tmp(first_tarLab, last_tarLab);

			for(vector<pair<int, int>>::iterator iter = tmp.begin(); iter != tmp.end(); iter++) {

				isTestLab[*iter] = true;

				for(vector<pair<int,int>>::iterator iter2=labDict[*iter].begin(); iter2!=labDict[*iter].end(); iter2++) {

					Cj_tar += iter2->second;
				}
			}
		}

		Cj_train -= Cj_tar;
	}
}


Patient::~Patient() {

	pheDict.clear();
	labDict.clear();
	obsDict.clear();
	gamma.clear();
	lambda.clear();
	isTestPhe.clear();
	isTestLab.clear();
}






































