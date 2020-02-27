#include "MixEHR.h"

using namespace std;
using namespace arma;

MixEHR* parseCmdLine(int argc, char *argv[]) {

	string datafile_train = "";
	string datafile_test = "";
	string metafile = "";

	string inferMethod="JCVB0";

	string newDatafile="";
	string trainedModelPrefix="";

	string imputeTargetsFile="";
	string imputePatDataFile="";

	string trainPatMetapheFile="";
	string trainPatIdFile = "";

	int impute_knn = 25;

	string output_dir=".";

	// default parameters
	int iterations=100;
	double loglikdiff_thres = 1e-7;
	int numOfTopics=10;
	double testSetFrac=0.1;

	double kappaStepsize=0.5;

	int batchsize=100;
	int numOfBurnins=10;

	int targetViewId = 1;
	bool evalTargetViewOnly = false;

	bool inferNewPatientMetaphe = false;

	bool inferTrainPatientMetaphe = false;

	bool outputIntermediates = false;

	bool mar = false;

	bool evalLabOnly = false;

	bool testMissingLabOnly = false;

	bool testObservedLabOnly = false;

	int inferPatParams_maxiter = 10;

	int maxcores = omp_get_max_threads(); // @suppress("Function cannot be resolved")

	string helpMsg = "./mixehr -f examples/toydata.txt -m1 examples/toymeta_phe.txt -i 10 -k 10";

	if (argc < 2) {

		cout << helpMsg << endl;

		exit(1);

	} else {

		for(int i = 1; i < argc; i++){

			string argComp = argv[i];

			if(argComp.compare("-h")== 0 || argComp.compare("--help")== 0) {

				cout << helpMsg << endl;

				exit(0);

			} else if(argComp.compare("-f")== 0 || argComp.compare("--trainDataFile")== 0) {

				datafile_train = string(argv[i+1]);

			} else if(argComp.compare("-t")== 0 || argComp.compare("--testDataFile")== 0) {

				datafile_test = string(argv[i+1]);

			} else if(argComp.compare("-m")== 0 || argComp.compare("--metaFile")== 0) {

				metafile = string(argv[i+1]);

			} else if(argComp.compare("-i") == 0 || argComp.compare("--iter")== 0){

				iterations = atoi(argv[i+1]);

			} else if(argComp.compare("--convergenceThreshold") == 0){

				loglikdiff_thres = atof(argv[i+1]);

			} else if(argComp.compare("-k") == 0 || argComp.compare("--topics")== 0) {

				numOfTopics = atoi(argv[i+1]);

				if(numOfTopics < 2) throw::runtime_error("numOfTopics must be at least 2!");

			} else if(argComp.compare("-b") == 0 || argComp.compare("--batchsize")== 0) {

				batchsize = atoi(argv[i+1]);

			} else if(argComp.compare("-n") == 0 || argComp.compare("--inferenceMethod")== 0) {

				inferMethod = string(argv[i+1]);

			} else if(argComp.compare("-r")== 0 || argComp.compare("--burnin")== 0){

				numOfBurnins = atoi(argv[i+1]);

			} else if(argComp.compare("-s") == 0 || argComp.compare("--stepsize")== 0) {

				kappaStepsize = atof(argv[i+1]);

			} else if(argComp.compare("--mar") == 0) {

				mar = true;

			} else if(argComp.compare("--evalLabOnly")== 0) { // evaluating lab imputation

				evalLabOnly = true;

			} else if(argComp.compare("--targetTypeId")== 0) { // evaluating lab imputation

				targetViewId = atoi(argv[i+1]);
				evalTargetViewOnly = true;

			} else if(argComp.compare("--testMissingLabOnly")== 0) { // evaluating imputing missing lab only

				testMissingLabOnly = true;

			} else if(argComp.compare("--testObservedLabOnly")== 0) { // evaluating imputing missing lab only

				testObservedLabOnly = true;

			} else if(argComp.compare("--newPatsData")== 0) {

				newDatafile = string(argv[i+1]);

			} else if(argComp.compare("--trainedModelPrefix")== 0) {

				trainedModelPrefix =string(argv[i+1]);

			} else if(argComp.compare("--inferNewPatientMetaphe")== 0) {

				inferNewPatientMetaphe = true;

			} else if (argComp.compare("--inferTrainPatientMetaphe")== 0) {

				inferTrainPatientMetaphe = true;

			} else if(argComp.compare("--trainPatMetapheFile")== 0) {

				trainPatMetapheFile = string(argv[i+1]);

			} else if(argComp.compare("--trainPatIdFile")== 0) {

				trainPatIdFile = string(argv[i+1]);

			} else if(argComp.compare("--imputeTargetsFile")== 0) {

				imputeTargetsFile = string(argv[i+1]);

			} else if(argComp.compare("--imputePatDataFile")==0) {

				imputePatDataFile = string(argv[i+1]);

			} else if(argComp.compare("--knn_impute") == 0) {

				impute_knn = atoi(argv[i+1]);

			} else if(argComp.compare("--outputIntermediates")== 0) {

				outputIntermediates = true;

			} else if(argComp.compare("--inferPatParams_maxiter")== 0) {

				inferPatParams_maxiter = atoi(argv[i+1]);

			} else if(argComp.compare("-o")== 0 || argComp.compare("--output_dir")== 0) {

				output_dir = string(argv[i+1]);

			} else if(argComp.compare("--maxcores")== 0) {

				maxcores = atoi(argv[i+1]);

				if(maxcores > omp_get_max_threads()) // @suppress("Function cannot be resolved")
					maxcores = omp_get_max_threads(); // @suppress("Function cannot be resolved")
			}
		}
	}

	cout << "--------------------" << endl;
	cout << "Input arguments: " << endl;
	cout << "trainDataFile: " << datafile_train << endl;
	cout << "testDataFile: " << datafile_test << endl;
	cout << "metaFile: " << metafile << endl;
	cout << "phenoTopics#: " << numOfTopics << endl;
	cout << "iter#: " << iterations << endl;
	cout << "convergenceThreshold: " << loglikdiff_thres << endl;
	cout << "inference method: " << inferMethod << endl;
	cout << "NMAR inference enabled: " << !mar << endl;
	cout << "testMissingLabOnly: " << testMissingLabOnly << endl;
	cout << "testObservedLabOnly: " << testObservedLabOnly << endl;
	cout << "maxcores: " << maxcores << endl;
	cout << "--------------------" << endl;


	return new MixEHR(datafile_train,
			datafile_test,
			metafile,
			numOfTopics,
			iterations,
			loglikdiff_thres,
			inferMethod,
			testSetFrac,
			batchsize,
			numOfBurnins,
			kappaStepsize,
			newDatafile,
			trainedModelPrefix,
			inferNewPatientMetaphe,

			inferTrainPatientMetaphe,
			trainPatMetapheFile,
			trainPatIdFile,

			imputeTargetsFile,
			imputePatDataFile,
			impute_knn,
			outputIntermediates,
			mar, evalLabOnly,
			testMissingLabOnly,
			testObservedLabOnly,
			targetViewId,
			evalTargetViewOnly,
			inferPatParams_maxiter,
			output_dir,
			maxcores);
}

int main(int argc, char *argv[]) {

	arma_rng::set_seed(123);

	MixEHR* mixehr = parseCmdLine(argc, argv);

//	omp_set_num_threads(omp_get_max_threads()); // @suppress("Type cannot be resolved")

	omp_set_num_threads(mixehr->maxcores); // @suppress("Function cannot be resolved")

	// parse meta information of the clinical variables
	mixehr->parseMetaInfo();

	JCVB0 *infer = mixehr->initialize_infer();

	double loglikdiff = 0;
	double logprddiff = 0;

	double tStart = omp_get_wtime(); // @suppress("Function cannot be resolved")

	if(mixehr->inferNewPatientMetaphe) {

		cout << "Use the trained model " << mixehr->trainedModelPrefix << endl;
		cout << "to infer new patient meta-phenotypes from " << mixehr->newDatafile << endl;

		double tStart_parse = omp_get_wtime(); // @suppress("Function cannot be resolved")

		infer = mixehr->parseNewData();

		double tEnd_parse = omp_get_wtime(); // @suppress("Function cannot be resolved")

		printf("Data import time taken: %.2fs\n", (double) tEnd_parse - tStart_parse);

		double tStart_infer = omp_get_wtime(); // @suppress("Function cannot be resolved")

		mixehr->inferNewPatMetaphe(infer);

		double tEnd_infer = omp_get_wtime(); // @suppress("Function cannot be resolved")

		printf("Meta-phenotypes inference time taken: %.2fs\n", (double) tEnd_infer - tStart_infer);


	} else if (mixehr->inferTrainPatMetaphe_only) {

		cout << "Infer training patient mixture using the trained model" << endl;
		cout << "trainedModelPrefix:" << mixehr->trainedModelPrefix << endl;
		cout << "trainDataFile: " << mixehr->trainDataFile << endl;


		mixehr->inferTrainPatMetaphe();


	} else if (mixehr->imputeNewPatientData) {

		cout << "Imputation phase" << endl;
		cout << "imputeTargetsFile: " << mixehr->imputeTargetsFile << endl;
		cout << "imputePatDataFile: " << mixehr->imputePatDataFile << endl;
		cout << "trainPatMetapheFile: " << mixehr->trainPatMetapheFile << endl;
		cout << "trainPatIdFile: " << mixehr->trainPatIdFile << endl;
		cout << "Impute new patient data with " << mixehr->k_nearest_neighbors << " nearest patients" << endl;

		mixehr->imputeNewPatData(mixehr->k_nearest_neighbors);

//		mixehr->inferNewPatMetaphe(infer);
//		infer = mixehr->parseTrainedModelFiles();

	} else { // train

		int myints[] = {mixehr->numOfIters};

		std::vector<int> iter2print(myints, myints + sizeof(myints) / sizeof(int) );

		if(mixehr->outputIntermediates) {

			iter2print.clear();
			int myints2[] = {1,2,5,10,20,50,100,150,200,300,500,1000};
			int moreiters = mixehr->numOfIters/1e3;
			std::vector<int> iter2print2(myints2, myints2 + sizeof(myints2) / sizeof(int) );

			for(std::vector<int>::iterator it = iter2print2.begin(); it != iter2print2.end(); it++) {
				if(*it < mixehr->numOfIters) {
					iter2print.push_back(*it);
				}
			}

			// output every 500 iterations from now on
			if(mixehr->numOfIters > 1000) {
				for(int i = 2; i <= moreiters; i++) {
					iter2print.push_back(i * 500);
				}
			}

			if(mixehr->numOfIters > (moreiters * 1e3)) iter2print.push_back(mixehr->numOfIters);
			cout << "Intermediate results will be saved from the following iterations: " << endl;
			for(std::vector<int>::iterator it = iter2print.begin(); it != iter2print.end(); it++)
				cout << *it << endl;
		}

		double tStart_parse = omp_get_wtime(); // @suppress("Function cannot be resolved")

		if(mixehr->inference.compare("JCVB0")==0) {

			mixehr->initialize_infer();

			mixehr->parseTrainData(infer);

		} else if(mixehr->inference.compare("SJCVB0")==0) {
			// polymorphism
			infer = mixehr->parseTrainDataBatches();
		} else {
			cout << mixehr->inference << endl;
			throw::runtime_error("invalid inference method");
		}
		if(mixehr->testDataFile.compare("")!= 0) {
			mixehr->parseTestData(infer);
		}

		double tEnd_parse = omp_get_wtime(); // @suppress("Function cannot be resolved")

		printf("Data import time taken: %.2fs\n", (double) tEnd_parse - tStart_parse);

		double trainStart = omp_get_wtime(); // @suppress("Function cannot be resolved")

		//		cout << endl << "before training" << endl << endl;
		//		infer->showParams();
		//		cout << endl;

		int iter = 0;

//		cout << "exportResults" << endl;
		if(mixehr->outputIntermediates) mixehr->exportResults(infer, iter, false);

//		cout << "trainLogLik" << endl;
		mixehr->logTrainLik(iter) = 0;

//		cout << "logPredLik" << endl;
		if(infer->testPats->size() > 0) mixehr->logPredLik(iter) = infer->predictLogLik();

		printf("%d: logTrainLik: %.8f; logTrainLik diff: %.8f; logPredLik: %.8f; logPredLik diff: %.8f\n",
				iter+1,
				mixehr->logTrainLik(iter),loglikdiff,
				mixehr->logPredLik(iter), logprddiff);

		for (iter = 1; iter < mixehr->numOfIters; iter++) {

			//			cout << "train" << endl;

			infer->train(true);
			//			cout << endl << "after training at iter " << iter << endl << endl;
			//			infer->showParams();
			//			cout << "logTrainLik" << endl;

			mixehr->logTrainLik(iter) = infer->trainLogLik();

			double trainSoFar = omp_get_wtime(); // @suppress("Function cannot be resolved")

			mixehr->trainTime(iter) = (double) (trainSoFar - trainStart);

			//			mixehr->logTrainLik_breakdowns.row(iter) = infer->trainLogLik_breakdowns();
			//			cout << "logPredLik" << endl;

			if(infer->testPats->size() > 0) mixehr->logPredLik(iter) = infer->predictLogLik();

			loglikdiff = mixehr->logTrainLik(iter) - mixehr->logTrainLik(iter-1);
			logprddiff = mixehr->logPredLik(iter) - mixehr->logPredLik(iter-1);

			printf("%d: logTrainLik: %.8f; logTrainLik diff: %.8f; logPredLik: %.8f; logPredLik diff: %.8f\n",
					iter+1,
					mixehr->logTrainLik(iter),loglikdiff,
					mixehr->logPredLik(iter), logprddiff);

			if(binary_search(iter2print.begin(), iter2print.end(), iter+1) && mixehr->outputIntermediates) {

				mixehr->exportResults(infer, iter, false);
				mixehr->exportLogLik(iter);
				mixehr->exportTrainTime(iter);
			}

			if(abs(loglikdiff) < mixehr->loglikdiff_thres && iter > 100 && !infer->svi) {
				break;
			}
		}

		cout << "Training completed after " << iter << " iterations" << endl;

		double trainEnd = omp_get_wtime(); // @suppress("Function cannot be resolved")

		printf("Training time taken: %.2fs\n", (double) (trainEnd - trainStart));

		if(iter < mixehr->numOfIters) { // converged

			mixehr->logTrainLik = mixehr->logTrainLik.head(iter);

			mixehr->logPredLik = mixehr->logPredLik.head(iter);

			mixehr->trainTime = mixehr->trainTime.head(iter);

			//			mixehr->logTrainLik_breakdowns = mixehr->logTrainLik_breakdowns.head_rows(iter+1);
		}

		if(infer->testPats->size() > 0) {

			cout << "Inferring meta-phenotypes of test patients for the final round" << endl;

			infer->inferTestPatParams_finalRun();
		}

		mixehr->exportResults(infer, iter, false);

		mixehr->exportLogLik(iter);

		mixehr->exportTrainTime(iter);

		//		mixehr->exportLogLik_breakdowns();
		//		mixehr->exportTestPatData(infer);
	}

	double tEnd = omp_get_wtime(); // @suppress("Function cannot be resolved")

	printf("Total time taken: %.2fs\n", (double) (tEnd - tStart));

	return 0;
}




























