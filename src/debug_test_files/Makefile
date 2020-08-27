gpp := clang++ -fopenmp -std=c++17 -I/usr/local/include -L/usr/local/lib
# gpp := g++-10 -std=c++17 -I/usr/local/Cellar/armadillo/9.900.2/include -L/usr/local/Cellar/armadillo/9.900.2/lib

all: algo
algo: test_algo.o test_module.o
	$(gpp) -o algo -g -Wall test_algo.o test_module.o -O2 -larmadillo

test_algo.o: test_algo.cpp test_module.h
	$(gpp) -g -c -Wall test_algo.cpp

test_module.o: test_module.cpp test_module.h
	$(gpp) -g -c -Wall test_module.cpp

clean:
	rm -f *.o
