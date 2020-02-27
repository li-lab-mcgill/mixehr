#include "JCVB0.h"

using namespace std;
using namespace arma;

// marginal likelihood estimated by the
// variational evidence lower bound (elbo)
// NB: bc in JCVB0, variance is not estimated
// this entails not including teh qlog(q) term
// in the original ELBO
double JCVB0::trainLogLik() {

	int D_train = trainPats->size();

	// theta elbo
	double elbo = D_train * (lgamma(alphaSum) - accu(lgamma(alpha)));

	for(std::vector<Patient>::iterator patj = trainPats->begin(); patj!=trainPats->end(); patj++) {

		elbo += accu(lgamma(alpha + patj->metaphe)) - lgamma(alphaSum + accu(patj->metaphe));
	}

	//	cout << "theta elbo: " << elbo << endl;

	// phi elbo
	double betaPrior = 0;

	for(unordered_map<int,double>::iterator iter=betaSum.begin(); iter!=betaSum.end(); iter++) {

		int t = iter->first;

		//		cout << "betaSum[t]: " << betaSum[t] << endl;
		//		cout << "sumLogGammaBeta[t]: " << sumLogGammaBeta[t] << endl;

		betaPrior += lgamma(betaSum[t]) - sumLogGammaBeta[t];
	}


	double betaLike = 0;

	for(unordered_map<pair<int,int>, PheParams*>::iterator phePar = pheParams.begin(); phePar != pheParams.end(); phePar++) {

		betaLike += accu(lgamma(phePar->second->beta + phePar->second->phi));
	}

	double betaNorm = 0;

	for(unordered_map<int,double>::iterator iter=betaSum.begin(); iter!=betaSum.end(); iter++) {

		int t = iter->first;

		for(int k = 0; k < K; k++) {

			betaNorm += lgamma(betaSum[t] + topicCountsPerPheType[t](k));
		}
	}

	elbo += K * betaPrior + betaLike - betaNorm;

	//	cout << "phi elbo: " << K * betaPrior + betaLike - betaNorm << endl;
	//	cout << "betaPrior: " << betaPrior << endl;
	//	cout << "betaLike: " << betaLike << endl;
	//	cout << "betaNorm: " << betaNorm << endl;
	//	throw::runtime_error("");


	// eta elbo
	double zetaPrior = 0;
	double zetaLike = 0;
	double zetaNorm = 0;

	for(unordered_map<pair<int,int>, LabParams*>::iterator labPar = labParams.begin(); labPar != labParams.end(); labPar++) {

		zetaPrior += lgamma(zetaSum[labPar->first]) - sumLogGammaZeta[labPar->first];

		for(int k=0; k<K; k++) {

			for(int v=0; v<labPar->second->V; v++) {

				zetaLike += 	lgamma(labPar->second->zeta(v) + labPar->second->eta(v,k));
			}

			zetaNorm += lgamma(accu(labPar->second->zeta) + accu(labPar->second->eta.col(k)));
		}
	}


	elbo += K * zetaPrior + zetaLike - zetaNorm;

	//	cout << "eta elbo: " << K * zetaPrior + zetaLike - zetaNorm << endl;

	// psi elbo
	if(!mar) {

		double psiLike = 0;

		for(unordered_map<pair<int,int>, LabParams*>::iterator labPar = labParams.begin(); labPar != labParams.end(); labPar++) {

			// sum across values
			psiLike += K*(lgamma(labPar->second->a + labPar->second->b) - lgamma(labPar->second->a) - lgamma(labPar->second->b)) +

					accu(lgamma(labPar->second->a + labPar->second->observedCnt) + lgamma(labPar->second->b + labPar->second->missingCnt) -

							lgamma(labPar->second->a + labPar->second->observedCnt + labPar->second->b + labPar->second->missingCnt));
		}

		elbo += psiLike;
	}


	//	cout << "psi elbo: " << K * zetaPrior + zetaLike - zetaNorm << endl;

	vector<Patient>::iterator pat0 = trainPats->begin();

	int j = 0;

	vec logEq = zeros<vec>(trainPats->size());

	// Eq[logq(z,h)]
#pragma omp parallel for shared(j)
	for(j=0; j < (int) trainPats->size(); j++) {

		vector<Patient>::iterator patj = pat0 + j;

		for(unordered_map<pair<int, int>, int>::iterator iter = patj->pheDict.begin(); iter != patj->pheDict.end(); iter++) {

			pair<int,int> pheId = iter->first;

			logEq(j) += accu(patj->gamma[pheId] % log(patj->gamma[pheId]));
		}

		for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++) {

			pair<int,int> labId = iter->first;

			if(patj->labDict.find(labId)!=patj->labDict.end()) {

				for(vector<pair<int,int>>::iterator iter2=patj->labDict[labId].begin(); iter2!=patj->labDict[labId].end(); iter2++) {

					logEq(j) += accu(patj->lambda[labId].row(iter2->first) % log(patj->lambda[labId].row(iter2->first)));
				}

			} else {

				logEq(j) += accu(patj->lambda[labId] % log(patj->lambda[labId]));
			}
		}

		// free up the memory allocated for the patient gamma hash
		patj->gamma.clear();

		// free up the memory allocated for the patient gamma hash
		patj->lambda.clear();
	}

	elbo -= accu(logEq);

	return elbo/C_train;
}


// first predict theta using x% of test terms
// then predict the 1-x% target terms using theta
double JCVB0::predictLogLik() {

	int j = 0;

	vec llk_vec = zeros<vec>(testPats->size());

	vector<Patient>::iterator pat0 = testPats->begin();

#pragma omp parallel for shared(j)
	for(j=0; j < (int) testPats->size(); j++) {

		vector<Patient>::iterator pat = pat0 + j;

//		cout << "testPat " << pat->patId << endl;

		inferPatParams(pat, inferTestPatMetaphe_maxiter_intermediateRun);

//		cout << "done" << endl;

		// infer patient target phes
		for(unordered_map<pair<int,int>, int>::iterator iter = pat->pheDict.begin(); iter != pat->pheDict.end(); iter++) {

			pair<int,int> pheId = iter->first;

			if(pat->isTestPhe[pheId]) {

				double llk_ij = accu(pat->metaphe_normalized % pheParams[pheId]->phi_normalized);

				llk_vec(j) += iter->second * log(llk_ij);
			}
		}

//		cout << "phe loglik done" << endl;

		// infer patient target labs
		for(unordered_map<pair<int,int>, vector<pair<int,int>>>::iterator iter = pat->labDict.begin(); iter != pat->labDict.end(); iter++) {

			pair<int,int> labId = iter->first;

			if(pat->isTestLab[labId]) {

//				cout << "labId: " << labId.second << endl;

				double labCnt = 0;

				// predictive distribution on lab results
				for(vector<pair<int,int>>::iterator iter2=iter->second.begin(); iter2!=iter->second.end(); iter2++) {

					double llk_ljv = accu(pat->metaphe_normalized % labParams[labId]->eta_normalized.row(iter2->first));

					llk_vec(j) += iter2->second * log(llk_ljv);

					labCnt += iter2->second;
				}

				// predictive distribution on missing indicator
				double llk_lj = 0;

				if(pat->obsDict[labId]) {

					llk_lj = accu(pat->metaphe_normalized % labParams[labId]->psi);

				} else {

					llk_lj = accu(pat->metaphe_normalized % (1 - labParams[labId]->psi));
				}

				llk_vec(j) += labCnt * log(llk_lj);
			}
		}

//		cout << "lab loglik done" << endl;
	}

	//	cout << "accu(llk_vec): " << accu(llk_vec) << endl;
	//	cout << "C_tar: " << C_tar << endl;

	return accu(llk_vec)/C_tar;
}



// DEBUG oNLY
rowvec JCVB0::trainLogLik_breakdowns() {

	rowvec elbo_bkdw = zeros<rowvec>(4+(!mar));

	int D_train = trainPats->size();
	int i = 0;

	// theta elbo
	double thetaELBO = D_train * (lgamma(alphaSum) - accu(lgamma(alpha)));

	for(std::vector<Patient>::iterator patj = trainPats->begin(); patj!=trainPats->end(); patj++) {

		thetaELBO += accu(lgamma(alpha + patj->metaphe)) - lgamma(alphaSum + accu(patj->metaphe));
	}

	elbo_bkdw(i) = thetaELBO;
	i++;


	// phi elbo
	double betaPrior = 0;

	for(unordered_map<int,double>::iterator iter=betaSum.begin(); iter!=betaSum.end(); iter++) {

		int t = iter->first;

		betaPrior += lgamma(betaSum[t]) - sumLogGammaBeta[t];
	}

	double betaLike = 0;

	for(unordered_map<pair<int,int>, PheParams*>::iterator phePar = pheParams.begin(); phePar != pheParams.end(); phePar++) {

		betaLike += accu(lgamma(phePar->second->beta + phePar->second->phi));
	}

	double betaNorm = 0;

	for(unordered_map<int,double>::iterator iter=betaSum.begin(); iter!=betaSum.end(); iter++) {

		int t = iter->first;

		for(int k = 0; k < K; k++) {

			betaNorm += lgamma(betaSum[t] + topicCountsPerPheType[t](k));
		}
	}

	elbo_bkdw(i) = K * betaPrior + betaLike - betaNorm;
	i++;


	// eta elbo
	double zetaPrior = 0;
	double zetaLike = 0;
	double zetaNorm = 0;

	for(unordered_map<pair<int,int>, LabParams*>::iterator labPar = labParams.begin(); labPar != labParams.end(); labPar++) {

		zetaPrior += lgamma(zetaSum[labPar->first]) - sumLogGammaZeta[labPar->first];

		for(int k=0; k<K; k++) {

			for(int v=0; v<labPar->second->V; v++) {

				zetaLike += 	lgamma(labPar->second->zeta(v) + labPar->second->eta(v,k));
			}

			zetaNorm += lgamma(accu(labPar->second->zeta) + accu(labPar->second->eta.col(k)));
		}
	}


	elbo_bkdw(i) = K * zetaPrior + zetaLike - zetaNorm;
	i++;

	//	cout << "eta elbo: " << K * zetaPrior + zetaLike - zetaNorm << endl;

	if(!mar) {
		// psi elbo
		double psiLike = 0;

		for(unordered_map<pair<int,int>, LabParams*>::iterator labPar = labParams.begin(); labPar != labParams.end(); labPar++) {

			psiLike += accu( lgamma(labPar->second->a + labPar->second->b) - lgamma(labPar->second->a) - lgamma(labPar->second->b) +

					lgamma(labPar->second->a + labPar->second->observedCnt) + lgamma(labPar->second->b + labPar->second->missingCnt) -

					lgamma(labPar->second->a + labPar->second->observedCnt + labPar->second->b + labPar->second->missingCnt));
		}

		elbo_bkdw(i) = psiLike;
		i++;
	}


	//	cout << "psi elbo: " << K * zetaPrior + zetaLike - zetaNorm << endl;

	double elbo_logq = 0;

	// Eq[logq(z,h)]
	for(std::vector<Patient>::iterator patj = trainPats->begin(); patj!=trainPats->end(); patj++) {

		for(unordered_map<pair<int, int>, int>::iterator iter = patj->pheDict.begin(); iter != patj->pheDict.end(); iter++) {

			pair<int,int> pheId = iter->first;

			elbo_logq += accu(patj->gamma[pheId] % log(patj->gamma[pheId]));
		}

		for(unordered_map<pair<int, int>, int>::iterator iter = patj->pheDict.begin(); iter != patj->pheDict.end(); iter++) {

			pair<int,int> labId = iter->first;

			elbo_logq += accu(patj->lambda[labId] % log(patj->lambda[labId]));
		}
	}

	elbo_bkdw(i) = elbo_logq;

	return elbo_bkdw/C_train;
}



























