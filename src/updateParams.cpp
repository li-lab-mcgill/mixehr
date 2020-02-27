#include "JCVB0.h"
#include "PheParams.h"
#include "LabParams.h"

using namespace std;
using namespace arma;

namespace bm = boost::math;


void JCVB0::updatePsi() {

	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++) {

		iter->second->psi = (iter->second->a + iter->second->observedCnt) /
				(iter->second->a + iter->second->b + iter->second->observedCnt + iter->second->missingCnt);
	}
}


void JCVB0::updateAlpha() {

	alphaSum = accu(alpha);

	// update alpha by Minka's fixed point iteration
	rowvec alpha_numer = zeros<rowvec>(K);

	double alpha_denom = 0;

	for(vector<Patient>::iterator pat = trainPats->begin(); pat != trainPats->end(); pat++) {

		for(int k=0; k<K; k++) {

			alpha_numer(k) += bm::digamma(pat->metaphe(k) + alpha(k));
		}

		alpha_denom += bm::digamma(accu(pat->metaphe) + alphaSum);
	}

	for(int k=0; k<K; k++) {
		alpha_numer(k) -= D_train * bm::digamma(alpha(k));
	}

	alpha_denom -= D_train * bm::digamma(alphaSum);

	// having 10 and 100 imply a gamma(2,20) with mean at 0.1
	alpha = alpha % alpha_numer/(10+alpha_denom);

//	cout << "alpha: " << alpha << endl;

	alphaSum = accu(alpha);
}


void JCVB0::updateBeta() {

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

		iter->second->beta = (1 + iter->second->beta * beta_numer) / (100 + beta_denom[t]); // fixed point update

		if(iter->second->beta==0) {
			throw::runtime_error("iter->second->beta become zeros");
		}

		betaSum[t] += iter->second->beta;

		sumLogGammaBeta[t] += lgamma(iter->second->beta);
	}

	beta_denom.clear();
}


void JCVB0::updateZeta() {

	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++) {

		vec zeta_numer = zeros<vec>(iter->second->V);
		vec zeta_denom = zeros<vec>(iter->second->V);

		for(int v=0; v < iter->second->V; v++) {

			for(int k=0; k < K; k++) {

				zeta_numer(v) += bm::digamma(iter->second->zeta(v) + iter->second->eta(v,k));

				zeta_denom(v) += bm::digamma(accu(iter->second->zeta) + accu(iter->second->eta.col(k)));
			}


			zeta_numer(v) -= K*bm::digamma(iter->second->zeta(v));

			zeta_denom(v) -= K*bm::digamma(accu(iter->second->zeta));

//			cout << "zeta_numer(v): " << zeta_numer(v) << endl;
//			cout << "zeta_denom(v): " << zeta_denom(v) << endl;
		}

		// apriori zeta follows gamma with (2, 2V) shape and rate implying equal prob for each state
//		iter->second->zeta = (1 + iter->second->zeta % zeta_numer)/(2*iter->second->V + zeta_denom);
//		iter->second->zeta = (100 + iter->second->zeta % zeta_numer)/(100*iter->second->V + zeta_denom);

		iter->second->zeta = (1e-3 + iter->second->zeta % zeta_numer)/(1e-3 + zeta_denom);

		if(any(iter->second->zeta==0)) {
			cout << iter->second->zeta << endl;
			throw::runtime_error("iter->second->zeta has zeros");
		}

//		cout << "iter->second->zeta: " << iter->second->zeta.t() << endl;

		zetaSum[iter->first] = accu(iter->second->zeta); // sum per test over test values

		sumLogGammaZeta[iter->first] = accu(lgamma(iter->second->zeta));
	}
}


void JCVB0::updatePsiHyper() {

	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++) {

		double norm = 0;
		double a_numer = 0;
		double b_numer = 0;

		for(int k = 0; k<K; k++) {

			norm += accu(bm::digamma(iter->second->a + iter->second->observedCnt(k) + iter->second->b + iter->second->missingCnt(k)));

			a_numer += bm::digamma(iter->second->a + iter->second->observedCnt(k));

			b_numer += bm::digamma(iter->second->b + iter->second->missingCnt(k));
		}

		norm -= K * bm::digamma(iter->second->a + iter->second->b);

		// assuming a and b follows gamma prior with parameter 2 and 1
		// to prevent zero when observedCnt/missingCnt equal to zeros
		// which makes the numerator zero
//		iter->second->a = (1 + iter->second->a * (a_denom - K*bm::digamma(iter->second->a)))/(2+norm);
//		iter->second->b = (1 + iter->second->b * (b_denom - K*bm::digamma(iter->second->b)))/(2+norm);

		iter->second->a = (1e-3 + iter->second->a * (a_numer - K*bm::digamma(iter->second->a)))/(1e-3 + norm);
		iter->second->b = (1e-3 + iter->second->b * (b_numer - K*bm::digamma(iter->second->b)))/(1e-3 + norm);


		if(iter->second->a==0 || iter->second->b==0) {
			cout << "iter->second->a: " << iter->second->a << endl;
			cout << "iter->second->b: " << iter->second->b << endl;
			throw::runtime_error("iter->second->a or b is zero");
		}

//		cout << "iter->second->a: " << iter->second->a << endl;
//		cout << "iter->second->b: " << iter->second->b << endl;
	}
}


void JCVB0::updateHyperParams() {

	updateAlpha();

//	cout << "alpha updated" << endl;

	updateBeta();

//	cout << "beta updated" << endl;

	updateZeta();

//	cout << "zeta updated" << endl;

	if(!mar) updatePsiHyper();

//	cout << "psi updated" << endl;
}


void JCVB0::updateParamSums() {

	// update topicCountsPerPheType
	unordered_map<int, rowvec> topicCountsPerPheType_new = unordered_map<int, rowvec>();

	for(unordered_map<int, rowvec>::iterator iter=topicCountsPerPheType.begin(); iter!=topicCountsPerPheType.end(); iter++)
		topicCountsPerPheType_new[iter->first] = zeros<rowvec>(K);

	for(unordered_map<pair<int,int>, PheParams*>::iterator iter = pheParams.begin(); iter != pheParams.end(); iter++)
		topicCountsPerPheType_new[iter->first.first] += iter->second->phi;

	for(unordered_map<int, rowvec>::iterator iter=topicCountsPerPheType.begin(); iter!=topicCountsPerPheType.end(); iter++)
		topicCountsPerPheType[iter->first] = topicCountsPerPheType_new[iter->first];


	// update stateCountsPerLabType
	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++)
		stateCountsPerLabType[iter->first] = sum(iter->second->eta);

	topicCountsPerPheType_new.clear();
}

void JCVB0::normalizeParams() {

	for(unordered_map<pair<int,int>, PheParams*>::iterator iter = pheParams.begin(); iter != pheParams.end(); iter++) {

		iter->second->phi_normalized = (iter->second->beta + iter->second->phi)/

				(betaSum[iter->first.first] + topicCountsPerPheType[iter->first.first]);
	}


	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++) {

		iter->second->eta_normalized = (repmat(iter->second->zeta,1,K) + iter->second->eta)/

				(accu(iter->second->zeta) + repmat(stateCountsPerLabType[iter->first],iter->second->V,1));
	}
}


void JCVB0::updateMainParams() {

	for(unordered_map<pair<int,int>, PheParams*>::iterator iter = pheParams.begin(); iter != pheParams.end(); iter++) {
		iter->second->phi.zeros(); // 1 x K
	}

	for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++) {
		iter->second->eta.zeros(); // V x K
		iter->second->observedCnt.zeros(); // 1 x K
		iter->second->missingCnt.zeros(); // 1 x K
	}

	// main update
	for(vector<Patient>::iterator pat = trainPats->begin(); pat != trainPats->end(); pat++) {

		// update phe params
		// iterate latents for ONLY the observed pat-phe
		for(unordered_map<pair<int, int>, int>::iterator iter = pat->pheDict.begin(); iter != pat->pheDict.end(); iter++) {

			pair<int,int> pheId = iter->first;

			pheParams[pheId]->phi += iter->second * pat->gamma[pheId];
		}

		// update lab params
		// iterate latents for ALL pat-lab including missing lab tests
		for(unordered_map<pair<int,int>, LabParams*>::iterator iter = labParams.begin(); iter != labParams.end(); iter++) {

			pair<int,int> labId = iter->first;

//			cout << "labId: " << labId.second << endl;

			if(!pat->obsDict[labId] && !mar) { // missing lab test

//				cout << "missing" << endl;

				for(int v=0; v < iter->second->V; v++) {

					iter->second->eta.row(v) += pat->lambda[labId].row(v);

					iter->second->missingCnt += pat->lambda[labId].row(v);
				}

			} else { // observed lab test

				for(vector<pair<int,int>>::iterator iter2=pat->labDict[labId].begin(); iter2!=pat->labDict[labId].end(); iter2++) {

					int v = iter2->first;

					iter->second->eta.row(v) += iter2->second * pat->lambda[labId].row(v);

					if(!mar) {
						iter->second->observedCnt += iter2->second * pat->lambda[labId].row(v);
					}
				}
			}
		}
	}

	updateParamSums();

	normalizeParams();

	if(!mar) updatePsi();
}


// sorted id for outputs
void JCVB0::sortID() {

	// sorted phe id
	vector<int> typeIds;

	for(unordered_map<pair<int,int>, PheParams*>::iterator phePar = pheParams.begin(); phePar != pheParams.end(); phePar++) {

		int t = phePar->first.first;
		int w = phePar->first.second;

		if(pheIds.find(t)==pheIds.end()) {

			typeIds.push_back(t);

			pheIds[t].push_back(w);

		} else {

			pheIds[t].push_back(w);
		}
	}

	sort(typeIds.begin(), typeIds.end());

	for(vector<int>::iterator iter=typeIds.begin(); iter!=typeIds.end(); iter++) {

		sort(pheIds[*iter].begin(), pheIds[*iter].end());
	}


	// sorted lab id
	typeIds.clear();

	for(unordered_map<pair<int,int>, LabParams*>::iterator labPar = labParams.begin(); labPar != labParams.end(); labPar++) {

		int t = labPar->first.first;
		int l = labPar->first.second;

		if(labIds.find(t)==labIds.end()) {

			typeIds.push_back(t);

			labIds[t].push_back(l);

		} else {

			labIds[t].push_back(l);
		}
	}

	sort(typeIds.begin(), typeIds.end());

	for(vector<int>::iterator iter=typeIds.begin(); iter!=typeIds.end(); iter++) {

		sort(labIds[*iter].begin(), labIds[*iter].end());
	}

}


void JCVB0::showParams() {

	cout << endl << "<<Phe parameters>>" << endl;

	for(map<int, vector<int>>::iterator t = pheIds.begin(); t != pheIds.end(); t++) {

		int typeId = t->first;

		for(vector<int>::iterator w = pheIds[t->first].begin(); w != pheIds[t->first].end(); w++) {

			int pheId = *w;

			cout << "--------------------" << endl;
			printf("typeId %d; pheId %d:\n", typeId, pheId);
			cout << "--------------------" << endl;

			cout << pheParams[make_pair(typeId, pheId)]->phi;

			cout << "--------------------" << endl;
			cout << "beta: " << pheParams[make_pair(typeId, pheId)]->beta << endl;
		}
	}

	cout << endl << "--------------------" << endl;
	cout << "topicCountsPerPheType: " << endl;
	for(map<int, vector<int>>::iterator t = pheIds.begin(); t != pheIds.end(); t++) {
		cout << topicCountsPerPheType[t->first];
	}

	cout << endl << "<<Lab parameters>>" << endl;


	for(map<int, vector<int>>::iterator t = labIds.begin(); t != labIds.end(); t++) {

		int typeId = t->first;

		for(vector<int>::iterator w = labIds[typeId].begin(); w != labIds[typeId].end(); w++) {

			int labId = *w;

			LabParams* labPar = labParams[make_pair(typeId, labId)];

			cout << "--------------------" << endl;
			printf("typeId %d; labId %d:\n", typeId, labId);

			cout << "--------------------" << endl;
			cout << "eta:" << endl;
			cout << labPar->eta;

			cout << "--------------------" << endl;
			cout << "eta normalized:" << endl;
			cout << labPar->eta_normalized;

			cout << "--------------------" << endl;
			cout << "a:" << labPar->a << "; " << "b:" << labPar->b << endl;

			cout << "--------------------" << endl;
			cout << "observedCnt:" << endl;
			cout << labPar->observedCnt;

			cout << "--------------------" << endl;
			cout << "missingCnt:" << endl;
			cout << labPar->missingCnt;


			cout << "--------------------" << endl;
			cout << "zeta:" << endl;
			cout << labPar->zeta;

//			for(vector<Patient>::iterator pat = trainPats->begin(); pat != trainPats->end(); pat++)
//				cout << "pat " << pat->patId << ": pat->stateProbs: " << endl << pat->stateProbs[make_pair(typeId,labId)];
		}
	}

	cout << "--------------------" << endl;
	cout << "alpha: " << endl << alpha << endl;
	cout << "--------------------" << endl;


	cout << endl << "--------------------" << endl << endl;
}







































