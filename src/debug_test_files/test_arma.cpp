#include <iostream>

#include <armadillo>
using namespace arma;

using namespace std;

int main()
{
	rowvec metaphe = randu<rowvec>(10);

//	rowvec metaphe = zeros<rowvec>(10);

	cout << metaphe << endl;

	cout << "done" << endl;

	return 0;
}
