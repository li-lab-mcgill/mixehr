#include <unordered_map>
#include <vector>
#include <list>
#include <random>
#include <chrono>
#include <ctime>
#include <iostream>
#include <string>
#include <assert.h>
#include <string>
#include <algorithm>


#include <armadillo>
using namespace arma;

#include "test_module.h"

int main()
{
	Test_Module* mymod = new Test_Module(0.1);

	mymod->foo();

	mymod->bar();

	return 0;
}
