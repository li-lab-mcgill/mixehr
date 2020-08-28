#include "MixEHR.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <sys/types.h>
#include <sys/stat.h>

#include <armadillo>

using namespace std;
using namespace arma;

namespace bo = boost::iostreams;

MixEHR::MixEHR(string datafile_train,
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
		string trainPatIDFile,

		string imputeTargetsFilePath,
		string imputePatDataFilePath,
		int impute_knn,
		bool saveIntermediates,
		bool missingAtRandom,
		bool evaluateLabOnly,
		bool testMissingLabOnly,
		bool testObservedLabOnly,
		int targetViewId,
		bool evalTargetViewOnly,
		int inferTestPatThetaMaxiter,
		string output_path,
		int maxthreads)
{
	numOfPats = 0;

	numOfPheTypes = 0;

	numOfLabTypes = 0;

	trainDataFile = datafile_train;

	testDataFile = datafile_test;

	metaFile = metafile;

	labParamsMap = new unordered_map<pair<int,int>, LabParams*>();

	pheParamsMap = new unordered_map<pair<int,int>, PheParams*>();

	numOfTopics = topics;

	numOfIters = maxiter;

	loglikdiff_thres = diff_thres;

	inference = inferMethod;

	testPatsFrac = testSetFrac;

	batchsize = batchSize;

	numOfBurnins=numOfBurnIns;

	kappaStepsize = stepsize;

	logPredLik = zeros<vec>(maxiter);

	logTrainLik = zeros<vec>(maxiter);

	trainTime = zeros<vec>(maxiter);

	logTrainLik_breakdowns = zeros<mat>(maxiter, 5);

	trainedModelPrefix = modelPrefix;

	newDatafile = newDataPath;

	inferNewPatientMetaphe=inferNewPatientParams;


	// infer patient mix for imputation
	inferTrainPatMetaphe_only = inferTrainPatientMix;
	trainPatMetapheFile = trainPatMixFile;
	trainPatIdFile = trainPatIDFile;


	imputeNewPatientData = false;

	if(imputeTargetsFilePath.length() > 0) {

		imputeTargetsFile = imputeTargetsFilePath;

		if(imputePatDataFilePath.length() > 0) {
			imputePatDataFile = imputePatDataFilePath;
		} else {
			throw::runtime_error("imputePatDataFilePath is missing");
		}

		imputeNewPatientData = true;
	}

	inferPatParams_maxiter = inferTestPatThetaMaxiter;

	k_nearest_neighbors = impute_knn;

	outputIntermediates = saveIntermediates;

	mar = missingAtRandom;

	missingLabOnly = testMissingLabOnly;

	observedLabOnly = testObservedLabOnly;

	if(missingLabOnly && observedLabOnly) {

		throw::runtime_error("missingLabOnly and observedLabOnly cannot be true the same time");
	}

	evalLabOnly = evaluateLabOnly;

	evalTargetTypeOnly = evalTargetViewOnly;

	targetTypeId = targetViewId;

	size_t lastindex = trainDataFile.find_last_of(".");

	outPrefix_trainData = trainDataFile.substr(0, lastindex);

	lastindex = testDataFile.find_last_of(".");

	outPrefix_testData = testDataFile.substr(0, lastindex);


	if(mar) {
		outPrefix_trainData = outPrefix_trainData + "_" + inference + "_mar" + "_K" + to_string(numOfTopics);
		outPrefix_testData = outPrefix_testData + "_" + inference + "_mar" + "_K" + to_string(numOfTopics);
	} else {
		outPrefix_trainData = outPrefix_trainData + "_" + inference + "_nmar" + "_K" + to_string(numOfTopics);
		outPrefix_testData = outPrefix_testData + "_" + inference + "_nmar" + "_K" + to_string(numOfTopics);
	}

	if(inferMethod.compare("SJCVB0")==0) {
		outPrefix_trainData = outPrefix_trainData + "_B" + to_string(batchSize);
		outPrefix_testData = outPrefix_testData + "_B" + to_string(batchSize);
	}


	if(newDatafile.length() > 0) {

		ifstream ifile(newDatafile.c_str());

		string trainedModelFile = trainedModelPrefix + "_phi.csv";

		ifstream ifile2(trainedModelFile.c_str());

		if(ifile && ifile2) {

			ifile.close();

			ifile2.close();

		} else {
			cout << "newDatafile: " << newDatafile << endl;
			cout << "trainedModelFile: " << trainedModelFile << endl;
			throw::runtime_error("File does not exists!");
		}
	}

	output_dir = output_path;

	numOfPhes = unordered_map<int,int>();
	numOfLabs = unordered_map<int,int>();

	maxcores = maxthreads;
}


JCVB0* MixEHR::initialize_infer() {

	JCVB0 *infer = new JCVB0();

	infer->initialize(
			numOfPhes, numOfLabs,
			numOfTopics, numOfIters,
			*pheParamsMap, *labParamsMap);

	infer->mar = mar;

	infer->inferTestPatMetaphe_maxiter_finalRun = inferPatParams_maxiter;

	return infer;
}


void MixEHR::exportResults(JCVB0* jcvb0, int iter, bool verbose) {

	// model parameters
	// W x K x V
	string outfile_phi = outPrefix_trainData + "_iter" + to_string(iter) + "_phi.csv";
	string outfile_phi_normalized = outPrefix_trainData + "_iter" + to_string(iter) + "_phi_normalized.csv";

	// L x K x V
	string outfile_eta = outPrefix_trainData + "_iter" + to_string(iter) + "_eta.csv";
	string outfile_eta_normalized = outPrefix_trainData + "_iter" + to_string(iter) + "_eta_normalized.csv";

	// L x K
	string outfile_obscnt = outPrefix_trainData + "_iter" + to_string(iter) + "_obscnt.csv";
	string outfile_miscnt = outPrefix_trainData + "_iter" + to_string(iter) + "_miscnt.csv";
	string outfile_psi = outPrefix_trainData + "_iter" + to_string(iter) + "_psi.csv";

	// hyper-parameters
	string outfile_alpha = outPrefix_trainData + "_iter" + to_string(iter) + "_alpha.csv";
	string outfile_beta = outPrefix_trainData + "_iter" + to_string(iter) + "_beta.csv";
	string outfile_zeta = outPrefix_trainData + "_iter" + to_string(iter) + "_zeta.csv";
	string outfile_psiHyper = outPrefix_trainData + "_iter" + to_string(iter) + "_psiHyper.csv";


	if(verbose) {

		cout << "Save unnormalized model parameters:" << endl;
		cout << "Save phi to " << outfile_phi << endl;
		cout << "Save phi_normalized to " << outfile_phi_normalized << endl;

		cout << "Save eta to " << outfile_eta << endl;
		cout << "Save eta_normalized to " << outfile_eta_normalized << endl;

		cout << "Save obscnt to " << outfile_obscnt << endl;
		cout << "Save miscnt to " << outfile_miscnt << endl;

		cout << "Save hyper-parameters: " << endl;
		cout << "Save alpha to " << outfile_alpha << endl;
		cout << "Save beta to " << outfile_beta << endl;
		cout << "Save zeta to " << outfile_zeta << endl;
		cout << "Save psiHyper to " << outfile_psiHyper << endl;
	}


	// main param
	ofstream outfile_stream_phi;
	ofstream outfile_stream_phi_normalized;
	ofstream outfile_stream_eta;
	ofstream outfile_stream_eta_normalized;

	ofstream outfile_stream_obscnt;
	ofstream outfile_stream_miscnt;
	ofstream outfile_stream_psi;

	// hyper
	ofstream outfile_stream_alpha;
	ofstream outfile_stream_beta;
	ofstream outfile_stream_zeta;
	ofstream outfile_stream_psiHyper;


	// main param
	outfile_stream_phi.open(outfile_phi);
	outfile_stream_phi_normalized.open(outfile_phi_normalized);

	outfile_stream_eta.open(outfile_eta);
	outfile_stream_eta_normalized.open(outfile_eta_normalized);

	outfile_stream_obscnt.open(outfile_obscnt);
	outfile_stream_miscnt.open(outfile_miscnt);
	outfile_stream_psi.open(outfile_psi);

	// hyper param
	outfile_stream_alpha.open(outfile_alpha);
	outfile_stream_beta.open(outfile_beta);
	outfile_stream_zeta.open(outfile_zeta);
	outfile_stream_psiHyper.open(outfile_psiHyper);

	// output matrix ordered by typeId, pheId, stateId from the unordered_map using an ordered map
	for(map<int, vector<int>>::iterator t = jcvb0->pheIds.begin(); t != jcvb0->pheIds.end(); t++) {

		int typeId = t->first;

		for(vector<int>::iterator w = jcvb0->pheIds[t->first].begin(); w != jcvb0->pheIds[t->first].end(); w++) {

			int pheId = *w;

			// caution: hashing missing keys will create an entry
			// export beta
			outfile_stream_beta << typeId << "," << pheId << "," <<
					jcvb0->pheParams[make_pair(typeId, pheId)]->beta << endl;

			outfile_stream_phi << typeId << "," << pheId;
			outfile_stream_phi_normalized << typeId << "," << pheId;

			for(int k=0; k<numOfTopics; k++) {

				outfile_stream_phi << "," << jcvb0->pheParams[make_pair(typeId, pheId)]->phi(k);
				outfile_stream_phi_normalized << "," << jcvb0->pheParams[make_pair(typeId, pheId)]->phi_normalized(k);
			}

			outfile_stream_phi << endl;
			outfile_stream_phi_normalized << endl;
		}
	}

	jcvb0->updatePsi(); // update psi before export

	for(map<int, vector<int>>::iterator t = jcvb0->labIds.begin(); t != jcvb0->labIds.end(); t++) {

		int typeId = t->first;


		for(vector<int>::iterator l = jcvb0->labIds[typeId].begin(); l != jcvb0->labIds[typeId].end(); l++) {

			int labId = *l;

			LabParams* labPar = jcvb0->labParams[make_pair(typeId, labId)];

			for(int v=0; v < labPar->V; v++) {

				outfile_stream_eta << typeId << "," << labId << "," << v;
				outfile_stream_eta_normalized << typeId << "," << labId << "," << v;

				outfile_stream_zeta << typeId << "," << labId << "," << v;

				for(int k=0; k<numOfTopics; k++) {

					outfile_stream_eta << "," << labPar->eta(v,k);

					outfile_stream_eta_normalized << "," << labPar->eta_normalized(v,k);
				}

				outfile_stream_eta << endl;
				outfile_stream_eta_normalized << endl;

				outfile_stream_zeta << "," << labPar->zeta(v) << endl;
			}

			outfile_stream_psiHyper << typeId << "," << labId << "," << labPar->a << "," << labPar->b << "\n";

			outfile_stream_obscnt << typeId << "," << labId;
			outfile_stream_miscnt << typeId << "," << labId;
			outfile_stream_psi << typeId << "," << labId;

			for(int k=0; k<numOfTopics; k++) {

				outfile_stream_obscnt << "," << labPar->observedCnt(k);
				outfile_stream_miscnt << "," << labPar->missingCnt(k);
				outfile_stream_psi << "," << labPar->psi(k);
			}

			outfile_stream_obscnt << endl;
			outfile_stream_miscnt << endl;
			outfile_stream_psi << endl;
		}
	}


	// export hyperparameter alpha
	outfile_stream_alpha << jcvb0->alpha(0);
	for(int k=1; k<numOfTopics; k++) {
		outfile_stream_alpha << "," << jcvb0->alpha(k);
	}
	outfile_stream_alpha << endl;



	// main params
	outfile_stream_phi.close();
	outfile_stream_phi_normalized.close();

	outfile_stream_eta.close();
	outfile_stream_eta_normalized.close();

	outfile_stream_obscnt.close();
	outfile_stream_miscnt.close();
	outfile_stream_psi.close();

	// hyper
	outfile_stream_alpha.close();
	outfile_stream_zeta.close();
	outfile_stream_psiHyper.close();
}


void MixEHR::exportLogLik(int iter) {

	string outfile_logTrainLik = outPrefix_trainData + "_iter" + to_string(iter) + "_logTrainLik.txt";

	ofstream outfile_stream_logTrainLik;

	outfile_stream_logTrainLik.open(outfile_logTrainLik);

	vec loglik_train = logTrainLik.head(iter);

	for(int i=0; i<(int)loglik_train.n_elem; i++) {

		outfile_stream_logTrainLik << loglik_train(i) << endl;
	}

	outfile_stream_logTrainLik.close();

	// test predictive log lik
	if(testDataFile.compare("")!=0) {

		string outfile_logPredLik;

		if(evalLabOnly) {

			if(missingLabOnly) {
				outfile_logPredLik = outPrefix_testData + "_iter" + to_string(iter) + "_evalLabOnly_missingLabOnly_logPredLik.txt";
			} else if(observedLabOnly) {
				outfile_logPredLik = outPrefix_testData + "_iter" + to_string(iter) + "_evalLabOnly_observedLabOnly_logPredLik.txt";
			} else{
				outfile_logPredLik = outPrefix_testData + "_iter" + to_string(iter) + "_evalLabOnly_all_logPredLik.txt";
			}

		} else {
			outfile_logPredLik = outPrefix_testData + "_iter" + to_string(iter) + "_logPredLik.txt";
		}

		ofstream outfile_stream_logPredLik;

		outfile_stream_logPredLik.open(outfile_logPredLik);

		vec loglik_pred = logPredLik.head(iter);

		for(int i=0; i<(int)loglik_pred.n_elem; i++) {

			outfile_stream_logPredLik << loglik_pred(i) << endl;
		}

		outfile_stream_logPredLik.close();
	}
}

void MixEHR::exportTrainTime(int iter) {

	string outfile_trainTime = outPrefix_trainData + "_iter" + to_string(iter) + "_trainTime.txt";

	ofstream outfile_stream_trainTime;

	outfile_stream_trainTime.open(outfile_trainTime);

	vec loglik_train = logTrainLik.head(iter);

	for(int i=0; i<(int)trainTime.n_elem; i++) {

		outfile_stream_trainTime << trainTime(i) << endl;
	}

	outfile_stream_trainTime.close();
}


void MixEHR::exportLogLik_breakdowns() {

	string outfile_logTrainLik = outPrefix_trainData + "_logTrainLikBKDW.csv";

	ofstream outfile_stream_logTrainLik;

	outfile_stream_logTrainLik.open(outfile_logTrainLik);

	for(int i=0; i<(int)logTrainLik_breakdowns.n_rows; i++) {

		outfile_stream_logTrainLik << i;

		for(int j=0; j<(int)logTrainLik_breakdowns.n_cols; j++) {

			outfile_stream_logTrainLik << "," << logTrainLik_breakdowns(i,j);
		}

		outfile_stream_logTrainLik << endl;
	}

	outfile_stream_logTrainLik.close();
}



// output test patient data for evaluation
void MixEHR::exportTestPatData(JCVB0* jcvb0) {

	string outfile_testPat_obsPhe = outPrefix_testData + "_testPat_obsPhe.csv";
	string outfile_testPat_tarPhe = outPrefix_testData + "_testPat_tarPhe.csv";


	ofstream outfile_stream_testPat_obsPhe;
	ofstream outfile_stream_testPat_tarPhe;

	outfile_stream_testPat_obsPhe.open(outfile_testPat_obsPhe);
	outfile_stream_testPat_tarPhe.open(outfile_testPat_tarPhe);

	for(vector<Patient>::iterator pat = jcvb0->testPats->begin(); pat != jcvb0->testPats->end(); pat++) {

		for(unordered_map<pair<int,int>, int>::iterator iter = pat->pheDict.begin(); iter != pat->pheDict.end(); iter++) {

			// obs phe used for inferring theta
			if(pat->isTestPhe[iter->first]) {

				// output format (same as input): patId,typeId,pheId,stateId,freq
				outfile_stream_testPat_tarPhe <<
						pat->patId << "," <<
						iter->first.first << "," <<
						iter->first.second << "," <<
						1 << "," <<
						iter->second << endl;

			} else {


				outfile_stream_testPat_obsPhe <<
						pat->patId << "," <<
						iter->first.first << "," <<
						iter->first.second << "," <<
						1 << "," <<
						iter->second << endl;
			}
		}


		for(unordered_map<pair<int,int>, vector<pair<int,int>>>::iterator iter = pat->labDict.begin(); iter != pat->labDict.end(); iter++) {

			// obs phe used for inferring theta
			if(pat->isTestLab[iter->first]) {

				for(vector<pair<int,int>>::iterator iter2=iter->second.begin(); iter2!=iter->second.end(); iter2++) {

					// output format (same as input): patId,typeId,pheId,stateId,freq
					outfile_stream_testPat_tarPhe <<
							pat->patId << "," <<
							iter->first.first << "," <<
							iter->first.second << "," <<
							iter2->first << "," <<
							iter2->second << endl;
				}

			} else {

				for(vector<pair<int,int>>::iterator iter2=iter->second.begin(); iter2!=iter->second.end(); iter2++) {

					outfile_stream_testPat_obsPhe <<
							pat->patId << "," <<
							iter->first.first << "," <<
							iter->first.second << "," <<
							iter2->first << "," <<
							iter2->second << endl;
				}
			}
		}
	}
}


void MixEHR::inferTrainPatMetaphe() {

	// parse model files
	JCVB0* jcvb0 = parseTrainedModelFiles();

	parseTrainData(jcvb0); // parse train patient data

	// create directory to save the output files
	struct stat buffer;
	if (stat (output_dir.c_str(), &buffer) != 0) {
		const int status = mkdir(output_dir.c_str(), S_IRWXU);
		if (status == -1) {
			cout << "Error creating directory: " << output_dir;
			exit(1);
		}
	}

	// infer train patient mix
	cout << "Infer training patient mix" << endl;

	jcvb0->inferAllPatParams(inferPatParams_maxiter);

	cout << "Training mix inference completed" << endl;

	size_t lastindex = trainPatMetapheFile.find_last_of(".");
	string prefix = trainPatMetapheFile.substr(0, lastindex);

	string outfile_train_patid = output_dir + "/" + prefix + "_patId.csv";

	ofstream outfile_stream_train_patid;
	outfile_stream_train_patid.open(outfile_train_patid);


	string outfile = output_dir + "/" + trainPatMetapheFile;
	ofstream outfile_stream_trainPat_theta;
	outfile_stream_trainPat_theta.open(outfile);

	string outfile2 = output_dir + "/" + prefix + "_normalized.csv";
	ofstream outfile_stream_trainPat_theta_normalized;
	outfile_stream_trainPat_theta_normalized.open(outfile2);

	for(vector<Patient>::iterator pat = jcvb0->trainPats->begin(); pat != jcvb0->trainPats->end(); pat++) {

		outfile_stream_train_patid << pat->patId << endl;

		// output theta
		outfile_stream_trainPat_theta << pat->metaphe(0);
		outfile_stream_trainPat_theta_normalized << pat->metaphe_normalized(0);

		for(int k=1; k<numOfTopics; k++) {

			outfile_stream_trainPat_theta << "," << pat->metaphe(k);
			outfile_stream_trainPat_theta_normalized << "," << pat->metaphe_normalized(k);
		}

		outfile_stream_trainPat_theta << endl;
		outfile_stream_trainPat_theta_normalized << endl;
	}

	outfile_stream_train_patid.close();
	outfile_stream_trainPat_theta.close();
	outfile_stream_trainPat_theta_normalized.close();
}




// infer expectation of patients' theta variables only
void MixEHR::inferNewPatMetaphe(JCVB0* jcvb0) {

	string inputFile;

	if (inferNewPatientMetaphe)
		inputFile = newDatafile;
	else
		inputFile = trainDataFile;

	int j = 0;

	vector<Patient>::iterator pat0 = jcvb0->testPats->begin();

	int D = jcvb0->testPats->size();

#pragma omp parallel for shared(j)
	for(j=0; j<D; j++) {

		vector<Patient>::iterator patj = pat0 + j;

		jcvb0->inferPatParams(patj, inferPatParams_maxiter);

		// free up the memory allocated for the patient gamma hash
		patj->gamma.clear();

		// free up the memory allocated for the patient lambda hash
		patj->lambda.clear();
	}

	size_t lastindex0 = trainedModelPrefix.find_last_of("/");
	string trainedModel =  trainedModelPrefix.substr(lastindex0+1, trainedModelPrefix.length());
	size_t lastindex = inputFile.find_last_of(".");
	string prefix = inputFile.substr(0, lastindex);

	string outfile_testPat_theta = prefix + "_" + trainedModel + "_metaphe.csv";

	cout << "Saving inferred patient metaphe: " << outfile_testPat_theta << endl;

	ofstream outfile_stream_testPat_theta;

	outfile_stream_testPat_theta.open(outfile_testPat_theta);
	
	for(vector<Patient>::iterator pat = jcvb0->testPats->begin(); pat != jcvb0->testPats->end(); pat++) {

		// output theta
		outfile_stream_testPat_theta << pat->metaphe_normalized(0);

		for(int k=1; k<numOfTopics; k++) {

			outfile_stream_testPat_theta << "," << pat->metaphe_normalized(k);
		}
		outfile_stream_testPat_theta << endl;
	}

	outfile_stream_testPat_theta.close();
}

void MixEHR::imputeNewPheData(JCVB0* jcvb0, int nearestNeighborK) {

	int t = 0;
	umat target_phe_true = zeros<umat>(jcvb0->imputeTargetPats->size(), pheImputeTargets.size());
	mat target_phe_pred = zeros<mat>(jcvb0->imputeTargetPats->size(), pheImputeTargets.size());

	for(vector<pair<int,int>>::iterator tar = pheImputeTargets.begin(); tar != pheImputeTargets.end(); tar++) {

		int tar_typeId = tar->first;
		int tar_pheId = tar->second;

		cout << "Predict target code: " << tar_typeId << ", " << tar_pheId << endl;

		int j = 0;
		vector<Patient>::iterator pat0 = jcvb0->imputeTargetPats->begin();

		mat test_pat_mix = zeros<mat>(jcvb0->imputeTargetPats->size(), numOfTopics);

#pragma omp parallel for shared(j)
		for(j=0; j < (int) jcvb0->imputeTargetPats->size(); j++) {

			pair<int,int> tarpheid = make_pair(tar_typeId, tar_pheId);

			vector<Patient>::iterator tar_pat = pat0 + j;

			unordered_map<pair<int,int>, int>::const_iterator hasit = tar_pat->pheDict.find(tarpheid);

			double tarphe_freq = 0;

			if(hasit != tar_pat->pheDict.end()) { // actual positive
				tarphe_freq = tar_pat->pheDict[tarpheid];
				tar_pat->pheDict.erase(hasit);
				target_phe_true(j,t) = 1;
			} // else actual positive

			// infer target patients' mix without the target label
			jcvb0->inferPatParams(tar_pat, inferPatParams_maxiter);

			// save test pat mix for export
			test_pat_mix.row(j) = tar_pat->metaphe_normalized;

			vec dist = zeros<vec>(jcvb0->trainPats->size());

			// find most similar training patients based on their mixes
			int i = 0;
			for(vector<Patient>::iterator pat = jcvb0->trainPats->begin(); pat != jcvb0->trainPats->end(); pat++) {
				dist(i) = accu(square(pat->metaphe_normalized - tar_pat->metaphe_normalized)); // 1 x K
				i++;
			}

			// pick the top K most similar training patients
			uvec top_trainPat_indices = sort_index(dist);
			top_trainPat_indices = top_trainPat_indices.head(nearestNeighborK);

			uvec bot_trainPat_indices = sort_index(dist, "descend");
			bot_trainPat_indices = bot_trainPat_indices.head(nearestNeighborK);


			// DEBUG BEGINS
//			cout << "test pat: " << j << endl;
//			cout << "dist(top_trainPat_indices): " << dist(top_trainPat_indices) << endl;
//			cout << "dist(bot_trainPat_indices): " << dist(bot_trainPat_indices) << endl;
			// DEBUG ENDS


			// predict target code based on the average of the top K most similar training patients' target label
			double patSum = 0;
			double totalSum = 0;
			for(int i=0; i<nearestNeighborK; i++) {
				int top_pat_index = top_trainPat_indices(i);
				vector<Patient>::iterator top_trainPat = jcvb0->trainPats->begin() + top_pat_index;
				if(top_trainPat->pheDict.find(tarpheid) != top_trainPat->pheDict.end()) {
					double weighted_count = top_trainPat->pheDict[tarpheid];
					patSum += weighted_count;
					totalSum += weighted_count;
				} else {
					totalSum++;
				}

			}
			target_phe_pred(j,t) = patSum/totalSum;

			// restore the erased observation
			if(target_phe_true(j,t)==1) { // actual positive
				tar_pat->pheDict[tarpheid] = tarphe_freq;
			}
		}

		// save test patient mix matrix in csv format
		//		string outfile_tar_pat_mix = output_dir + "/target_pat_mix_" + to_string(tar_typeId) + "_" + to_string(tar_pheId) + ".csv";
		//		test_pat_mix.save(outfile_tar_pat_mix, csv_ascii);

		t++;
	}

	// save impute target id
	string outfile_target_pheid = output_dir + "/target_pheid.csv";
	ofstream outfile_stream_target_pheid;
	outfile_stream_target_pheid.open(outfile_target_pheid);
	for(vector<pair<int,int>>::iterator tar = pheImputeTargets.begin(); tar != pheImputeTargets.end(); tar++) {
		int tar_typeId = tar->first;
		int tar_pheId = tar->second;
		outfile_stream_target_pheid << tar_typeId << "," << tar_pheId << endl;
	}
	outfile_stream_target_pheid.close();

	// save prediction and true labels in matrix csv format
	string outfile_target_pred = output_dir + "/target_phe_pred.csv";
	string outfile_target_true = output_dir + "/target_phe_true.csv";

	target_phe_pred.save(outfile_target_pred, csv_ascii);
	target_phe_true.save(outfile_target_true, csv_ascii);
}


void MixEHR::imputeNewLabData(JCVB0* jcvb0, int nearestNeighborK) {

	int t = 0;
	umat target_lab_obs_true = zeros<umat>(jcvb0->imputeTargetPats->size(), labImputeTargets.size());
	mat target_lab_obs_pred = zeros<mat>(jcvb0->imputeTargetPats->size(), labImputeTargets.size());

	umat target_lab_res_true = zeros<umat>(jcvb0->imputeTargetPats->size(), labImputeTargets.size());
	umat target_lab_res_pred = zeros<umat>(jcvb0->imputeTargetPats->size(), labImputeTargets.size());

	for(vector<pair<int,int>>::iterator tar = labImputeTargets.begin(); tar != labImputeTargets.end(); tar++) {

		int tar_typeId = tar->first;
		int tar_labId = tar->second;

		cout << "Predict target code: " << tar_typeId << ", " << tar_labId << endl;

		int j = 0;
		vector<Patient>::iterator pat0 = jcvb0->imputeTargetPats->begin();

		mat test_pat_mix = zeros<mat>(jcvb0->imputeTargetPats->size(), numOfTopics);

#pragma omp parallel for shared(j)
		for(j=0; j < (int) jcvb0->imputeTargetPats->size(); j++) {

			pair<int,int> tarlabid = make_pair(tar_typeId, tar_labId);

			vector<Patient>::iterator tar_pat = pat0 + j;

			unordered_map<pair<int,int>, vector<pair<int,int>>>::const_iterator hasit = tar_pat->labDict.find(tarlabid);

			vector<pair<int,int>> tarlab_res;

			if(hasit != tar_pat->labDict.end()) { // actual positive

				tarlab_res = tar_pat->labDict[hasit->first];
				tar_pat->labDict.erase(hasit);
				tar_pat->obsDict[hasit->first] = false;

				target_lab_obs_true(j,t) = 1;

				vec lab_res_sum = zeros<vec>(jcvb0->labParams[tarlabid]->V);

				for(vector<pair<int,int>>::iterator lab_res_iter = tar_pat->labDict[tarlabid].begin();
						lab_res_iter != tar_pat->labDict[tarlabid].end(); lab_res_iter++) {

					lab_res_sum(lab_res_iter->first) += lab_res_iter->second;
				}

				target_lab_res_true(j,t) = lab_res_sum.index_max();;

			} // else actual positive

			// infer target patients' mix without the target label
			jcvb0->inferPatParams(tar_pat, inferPatParams_maxiter);

			// save test pat mix for export
			test_pat_mix.row(j) = tar_pat->metaphe_normalized;

			vec dist = zeros<vec>(jcvb0->trainPats->size());

			// find most similar training patients based on their mixes
			int i = 0;
			for(vector<Patient>::iterator pat = jcvb0->trainPats->begin(); pat != jcvb0->trainPats->end(); pat++) {
				dist(i) = accu(square(pat->metaphe_normalized - tar_pat->metaphe_normalized)); // 1 x K
				i++;
			}

			// pick the top K most similar training patients
			uvec top_trainPat_indices = sort_index(dist);
			top_trainPat_indices = top_trainPat_indices.head(nearestNeighborK);

			// predict target code based on the average of the top K most similar training patients' target label
			double lab_obs_sum = 0;
			vec lab_res_sum = zeros<vec>(jcvb0->labParams[tarlabid]->V);

			double obs_total_sum = 0;

			for(int i=0; i<nearestNeighborK; i++) {

				int top_pat_index = top_trainPat_indices(i);

				vector<Patient>::iterator top_trainPat = jcvb0->trainPats->begin() + top_pat_index;

				if(top_trainPat->labDict.find(tarlabid) != top_trainPat->labDict.end()) {

					double weighted_obs_count = 0;

					for(vector<pair<int,int>>::iterator lab_res_iter = top_trainPat->labDict[tarlabid].begin();
							lab_res_iter != top_trainPat->labDict[tarlabid].end(); lab_res_iter++) {

						double weighted_res_count = lab_res_iter->second;

						lab_res_sum(lab_res_iter->first) += weighted_res_count;

						weighted_obs_count += weighted_res_count;
					}

					lab_obs_sum += weighted_obs_count;
					obs_total_sum += weighted_obs_count;

				} else {
					obs_total_sum++;
				}
			}

//			target_lab_obs_pred(j,t) = lab_obs_sum/obs_total_sum;
//			target_lab_res_pred(j,t) = lab_res_sum.index_max(); // hard prediction

			if(lab_res_sum.size()==2) { // binary case
				if(accu(lab_res_sum) == 0) {
					target_lab_res_pred(j,t) = 0;
				} else {
					target_lab_res_pred(j,t) = lab_res_sum(1)/accu(lab_res_sum); // soft prediction
				}
			} else {
				target_lab_res_pred(j,t) = lab_res_sum.index_max(); // hard prediction
				cout << "lab_res_sum.size(): " << lab_res_sum.size() << endl;
				throw::runtime_error("should not be here");
			}


			// TEST BEGIN
//			cout << "lab_res_sum: " << lab_res_sum << endl;
//			cout << "accu(lab_res_sum): " << accu(lab_res_sum) << endl;
//			cout << "target_lab_res_pred(j,t): " << target_lab_res_pred(j,t) << endl;
//			throw::runtime_error("testing soft prediction");
			// TEST END


			if(!isfinite(lab_obs_sum)) {
				cout << "lab_obs_sum: " << lab_obs_sum << endl;
				cout << "nearestNeighborK: " << nearestNeighborK << endl;
				cout << "lab_res_sum: " << lab_res_sum.t() << endl;
				throw::runtime_error("lab_obs_sum is not finite");
			}


			// restore the erased observation
			if(target_lab_obs_true(j,t)==1) { // actual positive
				tar_pat->labDict[tarlabid] = tarlab_res;
				tar_pat->obsDict[tarlabid] = true;
			}
		}

		// save test patient mix matrix in csv format
		//		string outfile_tar_pat_mix = output_dir + "/target_pat_mix_" + to_string(tar_typeId) + "_" + to_string(tar_labId) + ".csv";
		//		test_pat_mix.save(outfile_tar_pat_mix, csv_ascii);

		t++;
	}

	// save impute target id
	string outfile_target_labid = output_dir + "/target_labid.csv";
	ofstream outfile_stream_target_labid;
	outfile_stream_target_labid.open(outfile_target_labid);
	for(vector<pair<int,int>>::iterator tar = labImputeTargets.begin(); tar != labImputeTargets.end(); tar++) {
		int tar_typeId = tar->first;
		int tar_labId = tar->second;
		outfile_stream_target_labid << tar_typeId << "," << tar_labId << endl;
	}
	outfile_stream_target_labid.close();

	// save prediction and true labels in matrix csv format
	string outfile_target_lab_obs_pred = output_dir + "/target_lab_obs_pred.csv";
	string outfile_target_lab_obs_true = output_dir + "/target_lab_obs_true.csv";
	target_lab_obs_pred.save(outfile_target_lab_obs_pred, csv_ascii);
	target_lab_obs_true.save(outfile_target_lab_obs_true, csv_ascii);

	string outfile_target_lab_res_pred = output_dir + "/target_lab_res_pred.csv";
	string outfile_target_lab_res_true = output_dir + "/target_lab_res_true.csv";
	target_lab_res_pred.save(outfile_target_lab_res_pred, csv_ascii);
	target_lab_res_true.save(outfile_target_lab_res_true, csv_ascii);
}


void MixEHR::imputeNewPatData(int nearestNeighborK) {

	// parse model files
	JCVB0* jcvb0 = parseTrainedModelFiles();

	// infer train patients' mix
	parseImputeTargetsFile();

	parseTrainData(jcvb0); // parse train patient data

	parseImputePatDataFile(jcvb0);  // parse new patient data for imputation

	// create directory to save the output files
	struct stat buffer;
	if (stat (output_dir.c_str(), &buffer) != 0) {
		const int status = mkdir(output_dir.c_str(), S_IRWXU);
		if (status == -1) {
			cout << "Error creating directory: " << output_dir;
			exit(1);
		}
	}

	// infer train patient mix
//	cout << "Infer training patient mix" << endl;
//	jcvb0->inferAllPatParams(inferPatParams_maxiter);
//	cout << "Training mix inference completed" << endl;

	// insert previously inferred topic mixture to each train pat
	mat train_pat_mix;
	train_pat_mix.load(trainPatMetapheFile, csv_ascii);
	vec train_pat_id;
	train_pat_id.load(trainPatIdFile, csv_ascii);

	for(int j=0; j<(int)train_pat_id.n_rows; j++) {

		vector<Patient>::iterator pat = jcvb0->trainPats->begin() + j;

		if(pat->patId == train_pat_id(j)) {

			pat->metaphe = train_pat_mix.row(j);
			pat->metaphe_normalized = pat->metaphe/accu(pat->metaphe);

		} else {

			cout << "pat->patId" << pat->patId;
			cout << "train_pat_id(j)" << train_pat_id(j);
			throw::runtime_error("Train patients are in same order train pat mix");
		}
	}

	imputeNewPheData(jcvb0, nearestNeighborK);

	imputeNewLabData(jcvb0, nearestNeighborK);

	// save target patient id
	string outfile_target_patid = output_dir + "/target_patid.csv";
	ofstream outfile_stream_target_patid;
	outfile_stream_target_patid.open(outfile_target_patid);
	for(vector<Patient>::iterator patiter = jcvb0->imputeTargetPats->begin();
			patiter != jcvb0->imputeTargetPats->end(); patiter++) {
		outfile_stream_target_patid << patiter->patId << endl;
	}
	outfile_stream_target_patid.close();
}
















































