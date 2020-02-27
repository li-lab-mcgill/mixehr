#include "PatientBatch.h"

PatientBatch::PatientBatch(int m, vector<Patient> *patientVector) {
  M = m;
  patVector = patientVector;
}

PatientBatch::PatientBatch() {
  M = 0;
  patVector = new vector<Patient>();
}

PatientBatch::~PatientBatch() {
  patVector->clear();
}

