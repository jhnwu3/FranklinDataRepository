
# CXX to simplify compilation from makefile
CXX = g++

# target all means all targets currently defined in this file 
all: PSO para PSO_NL PSO_NLt

# target dependencies for main PSO program - also equally messy code inside!
PSO: main.o fileIO.o ODE.o calc.o
	g++ main.o fileIO.o ODE.o calc.o -o PSO -fopenmp
main.o: main.cpp main.hpp fileIO.hpp ODE.hpp calc.hpp  
	g++ -c -O3 -fopenmp main.cpp 
fileIO.o: fileIO.cpp main.hpp fileIO.hpp
	g++ -c fileIO.cpp
ODE.o: ODE.cpp main.hpp ODE.hpp
	g++ -c ODE.cpp
calc.o: calc.cpp main.hpp calc.hpp
	g++ -c calc.cpp

# target for parallel openMP test script
para: para.o
	g++ para.o -o para -fopenmp
para.o: para.cpp
	g++ -c -O3 para.cpp -o para.o -fopenmp

# nonlinear PSO equal weights
PSO_NL: PSO_NL.o 
	g++ PSO_NL.o -o PSO_NL -fopenmp
PSO_NL.o: PSO_NL.cpp
	g++ -c -O3 -fopenmp PSO_NL.cpp

# nonlinear PSO unequal wts
PSO_NLt: PSO_NLt.o 
	g++ PSO_NLt.o -o PSO_NLt -fopenmp
PSO_NLt.o: PSO_NLt.cpp
	g++ -c -O3 -fopenmp PSO_NLt.cpp
	
# this target deletes all files produced from the Makefile
# so that a completely new compile of all items is required
clean:
	rm -rf *.o PSO_S PSO PSO_NL PSO_NLt
