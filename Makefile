
# CXX to simplify compilation from makefile
CXX = g++

# target all means all targets currently defined in this file 
all: PSO para PSO_S

# target dependencies for main PSO program
PSO: main.o fileIO.o ODE.o var.o calc.o
	g++ main.o fileIO.o ODE.o var.o calc.o -o PSO -fopenmp
main.o: main.cpp main.hpp fileIO.hpp ODE.hpp calc.hpp  
	g++ -c main.cpp -fopenmp
fileIO.o: fileIO.cpp main.hpp fileIO.hpp
	g++ -c fileIO.cpp
ODE.o: ODE.cpp main.hpp ODE.hpp
	g++ -c ODE.cpp
var.o: var.cpp main.hpp
	g++ -c var.cpp
calc.o: calc.cpp main.hpp calc.hpp
	g++ -c calc.cpp

# target for parallel openMP test script
para: para.o
	g++ para.o -o para -fopenmp
para.o: para.cpp
	g++ -c para.cpp -o para.o -fopenmp

# this target is for the executable of Dr. Stewarts PSO alg converted to C++ 
# warning! Very messy code inside!
PSO_S: PSO_S.o 
	g++ PSO_S.o -o PSO_S
# this target is the dependency for PSO.o
PSO_S.o: PSO_S.cpp
	g++ -c PSO_S.cpp

# this target deletes all files produced from the Makefile
# so that a completely new compile of all items is required
clean:
	rm -rf *.o PSO_S ODE
