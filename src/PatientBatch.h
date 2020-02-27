#ifndef PATIENTBATCH_H_
#define PATIENTBATCH_H_

#include <vector>
#include "Patient.h"

class PatientBatch {
public:
  int M;
  vector<Patient> *patVector;
  PatientBatch();
  PatientBatch(int m, vector<Patient> *patientVector);
  virtual ~PatientBatch();
};

#endif /* PATIENTBATCH_H_ */



