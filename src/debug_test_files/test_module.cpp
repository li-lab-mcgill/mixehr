
#include "test_module.h"

using namespace std;
using namespace arma;

Test_Module::Test_Module(double my_beta) {

		beta = my_beta;

//		metaphe = zeros<rowvec>(10);
		metaphe = randu<rowvec>(10);
}

void Test_Module::foo() {

	myDict = unordered_map<pair<int,int>, vector<pair<int, int>>>();

	myDict[make_pair(1,2)].push_back(make_pair(0,3));

	cout << metaphe << endl;

	cout << "done" << endl;
}

void Test_Module::bar() {

	for(unordered_map<pair<int,int>, vector<pair<int, int>>>::iterator iter = myDict.begin(); iter != myDict.end(); iter++) {

//		pair<int,int> pheId = iter->first;
//		cout << pheId << endl;

		for(vector<pair<int,int>>::iterator iter2 = iter->second.begin(); iter2 != iter->second.end(); iter2++) {

			cout << iter2->first << endl;
			cout << iter2->second << endl;
		}
	}
}



