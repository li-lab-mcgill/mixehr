#include "SJCVB0.h"
#include "PatientBatch.h"

using namespace std;
using namespace arma;

namespace bm = boost::math;

SJCVB0::SJCVB0(){

	numOfBurnins = 1;
	batchsize = 0;
	batchNum = 0;
	batchId = 0;

	tau = 1;
	kappa = 0.5;
	rho = 0;
	rho_t = 0;

	D_train = 0;

	patientBatches = new vector<PatientBatch*>();
}

void SJCVB0::train(bool updateHyper) {

	if(rho_t==0) {
		rho = 1;
	} else {
		rho = pow(rho_t + tau, -kappa);
	}

	batchId = rand() % patientBatches->size();

//	cout << "E-step" << endl;
	inferBatchPatParams();

//	cout << "M-step" << endl;
	updateMainParams();

//	cout << "EB-step" << endl;
	if(updateHyper) updateHyperParams();

	//	rho_t = (*patientBatches)[batchId]->patVector->size();
	//	rho_t += (*patientBatches)[batchId]->patVector->size();

	rho_t++;
}


// E-step
void SJCVB0::inferBatchPatParams() {

	PatientBatch* patBatch = (*patientBatches)[batchId];

	vector<Patient> *patVector = patBatch->patVector;

	int j = 0;

	vector<Patient>::iterator pat0 = patVector->begin();

#pragma omp parallel for shared(j)
	for(j=0; j < (int) patVector->size(); j++) {

		inferPatParams(pat0 + j, numOfBurnins);
	}
}


// M-step:
void SJCVB0::updateMainParams() {

	PatientBatch* patBatch = (*patientBatches)[batchId];

	vector<Patient> *patVector = patBatch->patVector;


	for(unordered_map<pair<int,int>, PheParams*>::iterator iter = pheParamsHat.begin(); iter != pheParamsHat.end(); iter++) {
		iter->second->phi.zeros(); // 1 x K
	}

	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParamsHat.begin(); iter != labParamsHat.end(); iter++) {
		iter->second->eta.zeros(); // V x K
		iter->second->observedCnt.zeros(); // 1 x K
		iter->second->missingCnt.zeros(); // 1 x K
	}

	for(vector<Patient>::iterator pat=patVector->begin(); pat!=patVector->end(); pat++) {

		// update phe params
		// iterate latents for ONLY the observed pat-phe
		for(unordered_map<pair<int, int>, int>::iterator iter = pat->pheDict.begin(); iter != pat->pheDict.end(); iter++) {

			pair<int,int> pheId = iter->first;

			pheParamsHat[pheId]->phi += iter->second * pat->gamma[pheId];
		}

		// update lab params
		// iterate latents for ALL pat-lab including missing lab tests
		for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParamsHat.begin(); iter != labParamsHat.end(); iter++) {

			pair<int,int> labId = iter->first;


			if(!pat->obsDict[labId] && !mar) { // missing lab test

				for(int v=0; v < iter->second->V; v++) {

					iter->second->eta.row(v) += pat->lambda[labId].row(v);

					iter->second->missingCnt += pat->lambda[labId].row(v);
				}

			} else { // observed lab test

				for(vector<pair<int,int>>::iterator iter2=pat->labDict[labId].begin(); iter2!=pat->labDict[labId].end(); iter2++) {

					int v = iter2->first;

					iter->second->eta.row(v) += iter2->second * pat->lambda[labId].row(v);

					iter->second->observedCnt += iter2->second * pat->lambda[labId].row(v);
				}
			}
		}
	}


	// Update global parameters with approximate stochastic gradients
	for(unordered_map<pair<int,int>, PheParams*>::iterator iter = pheParamsHat.begin(); iter != pheParamsHat.end(); iter++) {

		pheParams[iter->first]->phi = (1-rho) * pheParams[iter->first]->phi + rho * iter->second->phi;
	}


	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParamsHat.begin(); iter != labParamsHat.end(); iter++) {

		labParams[iter->first]->eta = (1-rho) * labParams[iter->first]->eta + rho * iter->second->eta;

		labParams[iter->first]->observedCnt = (1-rho) * labParams[iter->first]->observedCnt + rho * iter->second->observedCnt;

		labParams[iter->first]->missingCnt = (1-rho) * labParams[iter->first]->missingCnt + rho * iter->second->missingCnt;
	}

	updateParamSumsSVI();

	normalizeParams();

	updatePsi();
}



void SJCVB0::updateParamSumsSVI() {

	// update topicCountsPerPheType
	for(unordered_map<int, rowvec>::iterator iter=topicCountsPerPheTypeHat.begin(); iter!=topicCountsPerPheTypeHat.end(); iter++)
		topicCountsPerPheTypeHat[iter->first].zeros();

	for(unordered_map<pair<int,int>, PheParams*>::iterator iter = pheParamsHat.begin(); iter != pheParamsHat.end(); iter++)
		topicCountsPerPheTypeHat[iter->first.first] += iter->second->phi;


	for(unordered_map<int, rowvec>::iterator iter=topicCountsPerPheTypeHat.begin(); iter!=topicCountsPerPheTypeHat.end(); iter++)
		topicCountsPerPheType[iter->first] = (1-rho) * topicCountsPerPheType[iter->first] + rho * topicCountsPerPheTypeHat[iter->first];


	// update stateCountsPerLabType
	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParamsHat.begin(); iter != labParamsHat.end(); iter++)
		stateCountsPerLabTypeHat[iter->first] = sum(iter->second->eta);

	for(unordered_map<pair<int,int>, rowvec> ::iterator iter = stateCountsPerLabTypeHat.begin(); iter != stateCountsPerLabTypeHat.end(); iter++)
		stateCountsPerLabType[iter->first] = (1-rho) * stateCountsPerLabType[iter->first] + rho * stateCountsPerLabTypeHat[iter->first];
}


void SJCVB0::updateHyperParams() {

//	cout << "updating alpha" << endl;

	updateAlpha();

//	cout << "alpha updated" << endl;

//	cout << "updating beta" << endl;

	updateBeta();

//	cout << "beta updated" << endl;

//	cout << "updating zeta" << endl;

	updateZeta();

//	cout << "zeta updated" << endl;

//	cout << "updating psi" << endl;

	if(!mar) updatePsiHyper();

//	cout << "psi updated" << endl;
}


void SJCVB0::updateAlpha() {

	PatientBatch* patBatch = (*patientBatches)[batchId];

	vector<Patient> *patVector = patBatch->patVector;

	alphaSum = accu(alpha);

	// update alpha by Minka's fixed point iteration
	rowvec alpha_numer = zeros<rowvec>(K);

	double alpha_denom = 0;

	for(vector<Patient>::iterator pat=patVector->begin(); pat!=patVector->end(); pat++) {

		for(int k=0; k<K; k++) {

			alpha_numer(k) += bm::digamma(pat->metaphe(k) + alpha(k));
		}

		alpha_denom += bm::digamma(accu(pat->metaphe) + alphaSum);
	}

	for(int k=0; k<K; k++) {
		alpha_numer(k) -= patVector->size() * bm::digamma(alpha(k));
	}

	alpha_denom -= patVector->size() * bm::digamma(alphaSum);

	// having 10 and 100 imply a gamma(2,20) with mean at 0.1
	alpha = (1-rho)*alpha + rho*(alpha % alpha_numer)/(10 + alpha_denom);

	if(any(alpha < 0)) {

		cout << "alpha: " << alpha << endl;
		cout << "alpha_numer: " << alpha_numer << endl;
		cout << "alpha_denom: " << alpha_denom << endl;

		throw::runtime_error("alpha become negative");
	}

	alphaSum = accu(alpha);
}


void SJCVB0::updateBeta() {

	unordered_map<int, double> beta_denom = unordered_map<int, double>();

		for(unordered_map<int,double>::iterator iter=betaSum.begin(); iter!=betaSum.end(); iter++) {

			int t = iter->first;

			beta_denom[t] = 0;

			betaSum[t] = 0;
			sumLogGammaBeta[t] = 0;
		}

		for(int k=0; k<K; k++) {

			unordered_map<int, double> beta_denom_k = unordered_map<int, double>();

			for(unordered_map<int,double>::iterator iter=betaSum.begin(); iter!=betaSum.end(); iter++) {

				beta_denom_k[iter->first] = 0;
			}

			for(unordered_map<pair<int,int>, PheParams*>::iterator iter = pheParams.begin(); iter != pheParams.end(); iter++) {

				int t = iter->first.first;

				beta_denom_k[t] += iter->second->beta + iter->second->phi(k);
			}

			for(unordered_map<int,double>::iterator iter=betaSum.begin(); iter!=betaSum.end(); iter++) {

				int t = iter->first;

				beta_denom[t] += bm::digamma(beta_denom_k[t]);
			}

			beta_denom_k.clear();
		}



		unordered_map<int, double> beta_denom_t = unordered_map<int, double>();

		for(unordered_map<int,double>::iterator iter=betaSum.begin(); iter!=betaSum.end(); iter++) {

			beta_denom_t[iter->first] = 0;
		}


		for(unordered_map<pair<int,int>, PheParams*>::iterator iter = pheParams.begin(); iter != pheParams.end(); iter++) {

			int t = iter->first.first;

			beta_denom_t[t] += iter->second->beta;
		}


		for(unordered_map<int,double>::iterator iter=betaSum.begin(); iter!=betaSum.end(); iter++) {

			int t = iter->first;

			beta_denom[t] -= K * bm::digamma(beta_denom_t[t]);
		}

		beta_denom_t.clear();


		for(unordered_map<pair<int,int>, PheParams*>::iterator iter = pheParams.begin(); iter != pheParams.end(); iter++) {

			int t = iter->first.first;

			double beta_numer = 0;

			for(int k=0; k<K; k++) {

				beta_numer += bm::digamma(iter->second->beta + iter->second->phi(k));
			}

			beta_numer -= K*bm::digamma(iter->second->beta);

			iter->second->beta = (1-rho) * iter->second->beta +

					rho * (1 + iter->second->beta * beta_numer) / (100 + beta_denom[t]); // fixed point update

			betaSum[t] += iter->second->beta;

			sumLogGammaBeta[t] += lgamma(iter->second->beta);
		}

		beta_denom.clear();
}


void SJCVB0::updateZeta() {

	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParamsHat.begin(); iter != labParamsHat.end(); iter++) {

		vec zeta_numer = zeros<vec>(iter->second->V);
		vec zeta_denom = zeros<vec>(iter->second->V);

		for(int v=0; v < iter->second->V; v++) {

			for(int k=0; k < K; k++) {

				zeta_numer(v) += bm::digamma(labParams[iter->first]->zeta(v) + iter->second->eta(v,k));

				zeta_denom(v) += bm::digamma(accu(labParams[iter->first]->zeta) + accu(iter->second->eta.col(k)));
			}


			zeta_numer(v) -= K*bm::digamma(labParams[iter->first]->zeta(v));

			zeta_denom(v) -= K*bm::digamma(accu(labParams[iter->first]->zeta));
		}

		// apriori zeta follows gamma with (2, 2V) shape and rate implying equal prob for each state
		vec zeta_new = (1e-3 + labParams[iter->first]->zeta % zeta_numer)/(1e-3 + zeta_denom);

		labParams[iter->first]->zeta = (1-rho)*labParams[iter->first]->zeta + rho * zeta_new;

		zetaSum[iter->first] = accu(labParams[iter->first]->zeta); // sum per test over test values

		sumLogGammaZeta[iter->first] = accu(lgamma(labParams[iter->first]->zeta));
	}
}


void SJCVB0::updatePsiHyper() {

	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParamsHat.begin(); iter != labParamsHat.end(); iter++) {

		double norm = 0;
		double a_numer = 0;
		double b_numer = 0;

		for(int k = 0; k<K; k++) {

			norm += accu(bm::digamma(labParams[iter->first]->a + iter->second->observedCnt(k) + labParams[iter->first]->b + iter->second->missingCnt(k)));
			a_numer += bm::digamma(labParams[iter->first]->a + iter->second->observedCnt(k));
			b_numer += bm::digamma(labParams[iter->first]->b + iter->second->missingCnt(k));
		}

		norm -= K * bm::digamma(iter->second->a + iter->second->b);

		// assuming a and b follows gamma prior with parameter 2 and 1
		// to prevent zero when observedCnt/missingCnt equal to zeros
		// which makes the numerator zero
		double a_new = (1e-3 + labParams[iter->first]->a * (a_numer - K*bm::digamma(labParams[iter->first]->a)))/(1e-3 + norm);
		double b_new = (1e-3 + labParams[iter->first]->b * (b_numer - K*bm::digamma(labParams[iter->first]->b)))/(1e-3 + norm);

		labParams[iter->first]->a = (1-rho)*labParams[iter->first]->a + rho * a_new;
		labParams[iter->first]->b = (1-rho)*labParams[iter->first]->b + rho * b_new;


		if(labParams[iter->first]->a < 0 || labParams[iter->first]->b < 0) {
			showParams();
			cout << "labParams[iter->first]->a: " << labParams[iter->first]->a << endl;
			cout << "labParams[iter->first]->b: " << labParams[iter->first]->b << endl;
			throw::runtime_error("a or b became negative");
		}
	}
}



double SJCVB0::trainLogLik() {

	vector<Patient> *patVector = (*patientBatches)[batchId]->patVector;

	int D_train_batch = patVector->size();

	// theta elbo
	double elbo = D_train_batch * (lgamma(alphaSum) - accu(lgamma(alpha)));

	for(std::vector<Patient>::iterator patj = patVector->begin(); patj!=patVector->end(); patj++) {

		elbo += accu(lgamma(alpha + patj->metaphe)) - lgamma(alphaSum + accu(patj->metaphe));
	}

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

	elbo += K * betaPrior + betaLike - betaNorm;


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
	double psiLike = 0;

	for(unordered_map<pair<int,int>, LabParams*>::iterator labPar = labParams.begin(); labPar != labParams.end(); labPar++) {

		// sum across values
		psiLike += K*(lgamma(labPar->second->a + labPar->second->b) - lgamma(labPar->second->a) - lgamma(labPar->second->b)) +

				accu(lgamma(labPar->second->a + labPar->second->observedCnt) + lgamma(labPar->second->b + labPar->second->missingCnt) -

						lgamma(labPar->second->a + labPar->second->observedCnt + labPar->second->b + labPar->second->missingCnt));
	}

	elbo += psiLike;

	//	cout << "psi elbo: " << K * zetaPrior + zetaLike - zetaNorm << endl;

	vector<Patient>::iterator pat0 = patVector->begin();

	int j = 0;

	vec logEq = zeros<vec>(patVector->size());

	// Eq[logq(z,h)]
#pragma omp parallel for shared(j)
	for(j=0; j < (int) patVector->size(); j++) {

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






























