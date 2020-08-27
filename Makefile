gpp := g++-10 -fopenmp -std=c++17 -I/usr/local/boost-1.68.0/include -I/usr/local/include -L/usr/local/lib -O2 -larmadillo
all: mixehr
mixehr: main.o MixEHR.o JCVB0.o SJCVB0.o Patient.o PatientBatch.o parseDataBatches.o parseData.o updateParams.o loglik.o
	$(gpp) -o mixehr -g main.o MixEHR.o JCVB0.o SJCVB0.o Patient.o PatientBatch.o parseDataBatches.o parseData.o updateParams.o loglik.o # -O2 -larmadillo

main.o: src/main.cpp src/MixEHR.h src/JCVB0.h
	$(gpp) -g -c -Wall src/main.cpp

MixEHR.o: src/MixEHR.cpp src/MixEHR.h
	$(gpp) -g -c -Wall src/MixEHR.cpp

JCVB0.o: src/JCVB0.cpp src/JCVB0.h src/Patient.h src/PheParams.h src/LabParams.h
	$(gpp) -g -c -Wall src/JCVB0.cpp
	
SJCVB0.o: src/SJCVB0.cpp src/SJCVB0.h src/JCVB0.h src/PatientBatch.h
	$(gpp) -g -c -Wall src/SJCVB0.cpp
		
loglik.o: src/loglik.cpp src/JCVB0.h
	$(gpp) -g -c -Wall src/loglik.cpp
	
updateParams.o: src/updateParams.cpp src/JCVB0.h
	$(gpp) -g -c -Wall src/updateParams.cpp

Patient.o: src/Patient.cpp src/Patient.h
	$(gpp) -g -c -Wall src/Patient.cpp
	
PatientBatch.o: src/PatientBatch.cpp src/PatientBatch.h src/Patient.h
	$(gpp) -g -c -Wall src/PatientBatch.cpp
	
parseData.o: src/parseData.cpp src/MixEHR.h src/PheParams.h src/LabParams.h src/JCVB0.h src/Patient.h
	$(gpp) -g -c -Wall src/parseData.cpp

parseDataBatches.o: src/parseDataBatches.cpp src/MixEHR.h
	$(gpp) -g -c -Wall src/parseDataBatches.cpp

clean:
	rm -f *.o

clean2:
	rm -f examples/*CVB0*
