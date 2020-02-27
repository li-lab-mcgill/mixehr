#include "JCVB0.h"
#include "PheParams.h"
#include "LabParams.h"

using namespace std;
using namespace arma;

void JCVB0::initialize(
		unordered_map<int,int> pheCnt,
		unordered_map<int,int> labCnt,
		int numOfTopics, int maxiter,
		unordered_map<pair<int,int>, PheParams*> pheParamsMap,
		unordered_map<pair<int,int>, LabParams*> labParamsMap)
{
	K = numOfTopics;

	numOfPhes = unordered_map<int,int>();
	numOfLabs = unordered_map<int,int>();

	C_train=0;
	C_tar=0;

	iterations = maxiter;

	svi = false;
	mar = false;
	imp = false;

	trainPats = new vector<Patient>();
	testPats = new vector<Patient>();
	imputeTargetPats = new vector<Patient>();

	// initialize alpha
	alpha = zeros<rowvec>(K);
	alpha.fill(0.1);
	alphaSum = accu(alpha);

	pheParams = pheParamsMap;

	labParams = labParamsMap;

	betaSum = unordered_map<int,double>();

	sumLogGammaBeta = unordered_map<int,double>();

	topicCountsPerPheType = unordered_map<int, rowvec>();

	stateCountsPerLabType = unordered_map<pair<int,int>, rowvec>();

	inferTestPatMetaphe_maxiter_intermediateRun = 2;

	// update by phe types
	for(unordered_map<int,int>::iterator iter=pheCnt.begin(); iter!=pheCnt.end(); iter++) {

		int t = iter->first;

		numOfPhes[t] = pheCnt[t];

		topicCountsPerPheType[t] = zeros<rowvec>(K);

		betaSum[t] = 0;

		sumLogGammaBeta[t] = 0;
	}


	// update by lab types
	for(unordered_map<int,int>::iterator iter=labCnt.begin(); iter!=labCnt.end(); iter++) {

		int u = iter->first;

		numOfLabs[u] = labCnt[u];
	}

	for(unordered_map<pair<int,int>, PheParams*>::iterator iter = pheParams.begin(); iter != pheParams.end(); iter++) {

		int t = iter->first.first;

		betaSum[t] += iter->second->beta;

		sumLogGammaBeta[t] += lgamma(iter->second->beta);
	}

	for(unordered_map<pair<int,int>,LabParams*>::iterator iter=labParams.begin(); iter!=labParams.end(); iter++) {

		stateCountsPerLabType[iter->first] = zeros<rowvec>(K);
	}


	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++) {

		zetaSum[iter->first] = accu(iter->second->zeta); // sum per test over test values

		sumLogGammaZeta[iter->first] = accu(lgamma(iter->second->zeta));
	}

	updateParamSums();

	normalizeParams();

	updatePsi();

	sortID();
}


JCVB0::~JCVB0() {

	pheParams.clear();
	labParams.clear();

	betaSum.clear();
	sumLogGammaBeta.clear();

	zetaSum.clear();
	sumLogGammaZeta.clear();

	for(vector<Patient>::iterator pat = trainPats->begin(); pat != trainPats->end(); pat++) {
		pat->~Patient();
	}
	trainPats->clear();

	for(vector<Patient>::iterator pat = testPats->begin(); pat != testPats->end(); pat++) {
		pat->~Patient();
	}
	testPats->clear();
}


// Update variational parameters by JCVB0
void JCVB0::train(bool updateHyper) {

	//	cout << "E-step" << endl;

	// E-step
	inferAllPatParams();

	//	cout << "M-step" << endl;

	// M-step
	updateMainParams();

	//	cout << "EB-step" << endl;

	if(updateHyper) updateHyperParams();
}


// E-step
void JCVB0::inferAllPatParams(int trainIter) {
	int j = 0;
	vector<Patient>::iterator pat0 = trainPats->begin();

#pragma omp parallel for shared(j)
	for(j=0; j < (int) trainPats->size(); j++) {
		inferPatParams(pat0 + j, trainIter);
	}
}



void JCVB0::inferTestPatParams_finalRun() {
	normalizeParams();
	int j = 0;
	vector<Patient>::iterator pat0 = testPats->begin();

#pragma omp parallel for shared(j)
	for(j=0; j < (int) testPats->size(); j++) {
		inferPatParams(pat0 + j, inferTestPatMetaphe_maxiter_finalRun);
	}
}

// E-step
void JCVB0::inferPatParams(vector<Patient>::iterator pat, int maxiter) {


	for(unordered_map<pair<int, int>, int>::iterator iter = pat->pheDict.begin(); iter != pat->pheDict.end(); iter++) {

		if(!pat->isTestPhe[iter->first]) {

			pat->gamma[iter->first] = zeros<rowvec>(K); // 1 x K
		}
	}

	// latent for all lab tests
	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++) {

		pat->lambda[iter->first] = zeros<mat>(iter->second->V, K); // V x K
	}

	double diff = 1;
	int iter = 1;

	while(diff > 1e-2 && iter <= maxiter) {

		rowvec patMetaphe_j_prev = pat->metaphe;

		// iterate latents for ONLY the observed pat-phe
		for(unordered_map<pair<int,int>, int>::iterator patiter = pat->pheDict.begin(); patiter != pat->pheDict.end(); patiter++) {

			pair<int,int> pheId = patiter->first;

			if(!pat->isTestPhe[pheId]) {

				// removing the current patient-phe token
				rowvec pheWeight_ij = patiter->second * pat->gamma[pheId];

				pat->gamma[pheId] = (pat->metaphe - pheWeight_ij + alpha) %
						(pheParams[pheId]->phi - pheWeight_ij + pheParams[pheId]->beta) /
						(topicCountsPerPheType[pheId.first] - pheWeight_ij + betaSum[pheId.first]);

				if(any(pat->gamma[pheId] < 0)) {

					if(pat->isTestPat || svi || imp) {

						pat->gamma[pheId] = (pat->metaphe + alpha) %
								(pheParams[pheId]->phi + pheParams[pheId]->beta) /
								(topicCountsPerPheType[pheId.first] + betaSum[pheId.first]);

					} else {

						// DEBUG S impossible for training unless there is a bug
						cout << "pat->patId: " << pat->patId << "; typeId: " <<
								pheId.first << "; pheId: " << pheId.second << endl;

						cout << "iter: " << iter << endl;

						cout << "pat->gamma[pheId]: " << pat->gamma[pheId] << endl;

						cout << "pheWeight_ij: " << pheWeight_ij << endl;
						cout << "pat->metaphe - pheWeight_ij + alpha: " << pat->metaphe - pheWeight_ij + alpha << endl;


						cout << "pheParams[pheId]->phi: " << pheParams[pheId]->phi << endl;
						cout << "pheWeight_ij: " << pheWeight_ij << endl;

						cout << "pheParams[pheId]->beta: " << pheParams[pheId]->beta << endl << endl;


						cout << "(pheParams[pheId]->phi - pheWeight_ij + pheParams[pheId]->beta)" <<
								(pheParams[pheId]->phi - pheWeight_ij + pheParams[pheId]->beta) << endl;

						cout << "(topicCountsPerPheType[pheId.first] - pheWeight_ij + betaSum[pheId.first]): "<<
								(topicCountsPerPheType[pheId.first] - pheWeight_ij + betaSum[pheId.first]) << endl;

						throw::runtime_error("inferPatParams(): pat->gamma[pheId] has negative values for training patients");
						// DEBUG E
					}
				}

				// not needed because normalizing over phe tokens produce much better results (below)
				pat->gamma[pheId] = pat->gamma[pheId]/accu(pat->gamma[pheId]);
			}
		}

		// iterate latents for ALL pat-lab including missing lab tests
		for(unordered_map<pair<int,int>, LabParams*>::iterator patiter = labParams.begin(); patiter != labParams.end(); patiter++) {

			pair<int,int> labId = patiter->first;

			int V = labParams[labId]->V;

			if(!pat->obsDict[labId] && !mar) { // true missing lab test

				//				cout << "lab " << labId.second << " missing" << endl;

				pat->lambda[labId].zeros();

				rowvec labWeight_lj = sum(pat->lambda[labId]);

				for(int v=0; v < V; v++) {

					//					cout << "v " << v << endl;

					rowvec labWeight_ljv = pat->lambda[labId].row(v);

					// 3 weighting factors: 1. metaphe; 2. lab value; 3. missing indicator
					pat->lambda[labId].row(v) = (pat->metaphe - labWeight_lj + alpha) %

							(labParams[labId]->eta.row(v) - labWeight_ljv + labParams[labId]->zeta(v)) /
							(stateCountsPerLabType[patiter->first] - labWeight_ljv + zetaSum[labId]) %

							((labParams[labId]->missingCnt - labWeight_lj + labParams[labId]->b) /
									(labParams[labId]->observedCnt + labParams[labId]->a +
											labParams[labId]->missingCnt - labWeight_lj + labParams[labId]->b));


					// lab never seen before (possible for small training set or mar assumption
					if(any(pat->lambda[labId].row(v) < 0)) {

						if(pat->isTestPat || svi || imp) {

							pat->lambda[labId].row(v) = (pat->metaphe + alpha) %

									(labParams[labId]->eta.row(v) + labParams[labId]->zeta(v)) /
									(stateCountsPerLabType[patiter->first] + zetaSum[labId]) %

									((labParams[labId]->missingCnt + labParams[labId]->b) /
											(labParams[labId]->observedCnt + labParams[labId]->a +
													labParams[labId]->missingCnt + labParams[labId]->b));
						} else {

							// DEBUG S (impossible for training unless there is a bug)
							cout << "typeId: " << labId.first << "; labId: " << labId.second << endl;
							cout << "pat->lambda[labId]: " << pat->lambda[labId] << endl;
							throw::runtime_error("inferPatParams(): pat->lambda[labId] for missing lab test has negative values");
							// DEBUG E
						}
					}
				}

				//				cout << "done" << endl;

			} else if(pat->obsDict[labId] && pat->isTestLab[labId] && !mar) { // observed target lab test

				//				cout << "lab " << labId.second << " test" << endl;

				pat->lambda[labId].zeros();

				rowvec labWeight_lj = sum(pat->lambda[labId]);

				for(int v=0; v < V; v++) {

					rowvec labWeight_ljv = pat->lambda[labId].row(v);

					pat->lambda[labId].row(v) = (pat->metaphe - labWeight_lj + alpha) %

							(labParams[labId]->eta.row(v) - labWeight_ljv + labParams[labId]->zeta(v)) /
							(stateCountsPerLabType[patiter->first] - labWeight_ljv + zetaSum[labId]) %

							((labParams[labId]->observedCnt - labWeight_lj + labParams[labId]->a) /
									(labParams[labId]->observedCnt + labParams[labId]->a +
											labParams[labId]->missingCnt - labWeight_lj + labParams[labId]->b));


					// lab never seen before (possible for small training set or mar assumption
					if(any(pat->lambda[labId].row(v) < 0)) {

						pat->lambda[labId].row(v) = (pat->metaphe + alpha) %

								(labParams[labId]->eta.row(v) + labParams[labId]->zeta(v)) /
								(stateCountsPerLabType[patiter->first] + zetaSum[labId]) %

								((labParams[labId]->observedCnt + labParams[labId]->a) /
										(labParams[labId]->observedCnt + labParams[labId]->a +
												labParams[labId]->missingCnt + labParams[labId]->b));
					}
				}
			} else if(pat->obsDict[labId] && !pat->isTestLab[labId]) { // observed non-target lab test for theta inference

				//				cout << "lab " << labId.second << " observed" << endl;

				rowvec labWeight_lj = zeros<rowvec>(K);

				for(vector<pair<int,int>>::iterator iter2 = pat->labDict[labId].begin(); iter2 != pat->labDict[labId].end(); iter2++) {

					labWeight_lj += iter2->second * pat->lambda[labId].row(iter2->first);
				}

				for(vector<pair<int,int>>::iterator iter2 = pat->labDict[labId].begin(); iter2 != pat->labDict[labId].end(); iter2++) {

					int v = iter2->first;

					rowvec labWeight_ljv = iter2->second * pat->lambda[labId].row(v);

					if(!mar) {

						pat->lambda[labId].row(v) = (pat->metaphe - labWeight_lj + alpha) %

								(patiter->second->eta.row(v) - labWeight_ljv + patiter->second->zeta(v)) /
								(stateCountsPerLabType[patiter->first] - labWeight_ljv + zetaSum[labId]) %

								((labParams[labId]->observedCnt - labWeight_lj + labParams[labId]->a) /
										(labParams[labId]->observedCnt - labWeight_lj + labParams[labId]->a +
												labParams[labId]->missingCnt + labParams[labId]->b));

					} else {

						pat->lambda[labId].row(v) = (pat->metaphe - labWeight_lj + alpha) %

								(patiter->second->eta.row(v) - labWeight_ljv + patiter->second->zeta(v)) /
								(stateCountsPerLabType[patiter->first] - labWeight_ljv + zetaSum[labId]);
					}


					// lab never seen before (possible for small training set or mar assumption
					if(any(pat->lambda[labId].row(v) < 0)) {

						if(pat->isTestPat || svi || imp) {

							if(!mar) {

								pat->lambda[labId].row(v) = (pat->metaphe + alpha) %

										(patiter->second->eta.row(v) + patiter->second->zeta(v)) /
										(stateCountsPerLabType[patiter->first] + zetaSum[labId]) %

										((labParams[labId]->observedCnt + labParams[labId]->a) /
												(labParams[labId]->observedCnt + labParams[labId]->a +
														labParams[labId]->missingCnt + labParams[labId]->b));
							} else {

								pat->lambda[labId].row(v) = (pat->metaphe + alpha) %

										(patiter->second->eta.row(v) + patiter->second->zeta(v)) /
										(stateCountsPerLabType[patiter->first] + zetaSum[labId]);
							}
						} else {

							// DEBUG S (impossible for training unless there is a bug)
							cout << "iter: " << iter << endl;

							cout << "patId: " << pat->patId;
							cout << "; typeId: " << labId.first << "; labId: " << labId.second << "; stateId: " << v << endl;

							cout << "pat->lambda[labId]: " << pat->lambda[labId] << endl;

							cout << "labWeight_ljv: " << labWeight_ljv << endl;

							cout << "(pat->metaphe - labWeight_lj + alpha): " << (pat->metaphe - labWeight_lj + alpha) << endl;

							cout << "iter->second->eta.row(v) - labWeight_ljv + iter->second->zeta(v): " <<
									patiter->second->eta.row(v) - labWeight_ljv + patiter->second->zeta(v) << endl;

							cout << "stateCountsPerLabType[iter->first] - labWeight_ljv + zetaSum[labId]: " <<
									stateCountsPerLabType[patiter->first] - labWeight_ljv + zetaSum[labId] << endl;


							cout << "(labParams[labId]->observedCnt - labWeight_lj + labParams[labId]->a): " <<
									(labParams[labId]->observedCnt - labWeight_lj + labParams[labId]->a) << endl;

							cout << "labParams[labId]->observedCnt: " << labParams[labId]->observedCnt << endl;
							cout << "labWeight_lj: " << labWeight_lj << endl;
							cout << "labParams[labId]->a: " << labParams[labId]->a << endl;


							cout << "labParams[labId]->observedCnt - labWeight_lj + labParams[labId]->a + ..." <<
									"labParams[labId]->missingCnt + labParams[labId]->b" << endl;
							cout << labParams[labId]->observedCnt - labWeight_lj + labParams[labId]->a +
									labParams[labId]->missingCnt + labParams[labId]->b << endl;


							throw::runtime_error("inferPatParams(): pat->lambda[labId] for observed lab test has negative values");
							// DEBUG E
						}
					}
				}
			}

			// V x K
			pat->lambda[labId] = normalise(pat->lambda[labId],1,1); // normalize for each row across K
		}

		pat->metaphe.zeros();


		// infer patient metaphe
		// update phe params
		// iterate latents for ONLY the observed pat-phe
		for(unordered_map<pair<int, int>, int>::iterator iter = pat->pheDict.begin(); iter != pat->pheDict.end(); iter++) {

			pair<int,int> pheId = iter->first;

			if(!pat->isTestPhe[pheId]) {

				pat->metaphe += iter->second * pat->gamma[pheId];
			}
		}


		// update lab params
		// iterate latents for ALL pat-lab including missing lab tests
		for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++) {

			pair<int,int> labId = iter->first;

			if(!pat->obsDict[labId] && !mar) { // true missing lab test

				pat->metaphe += sum(pat->lambda[labId]);

			} else {

				if(pat->isTestLab[labId] && !mar) {

					pat->metaphe += sum(pat->lambda[labId]);

				} else {

					for(vector<pair<int,int>>::iterator iter2=pat->labDict[labId].begin(); iter2!=pat->labDict[labId].end(); iter2++) {

						pat->metaphe += iter2->second * pat->lambda[labId].row(iter2->first);
					}
				}
			}
		}

		diff = accu(abs(pat->metaphe - patMetaphe_j_prev))/K;

		//		cout << "pat: " << pat->patId << " iter: " << iter << ": " << "diff: " << diff << "; pat->metaphe: " << pat->metaphe;

		iter++;
	}

	//	printf("inferPatParams iter %d; diff %.3f\n", iter, diff);

	pat->metaphe_normalized = alpha + pat->metaphe;
	pat->metaphe_normalized = pat->metaphe_normalized/accu(pat->metaphe_normalized);
}


































