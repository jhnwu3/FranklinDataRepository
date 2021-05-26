
# comments in a Makefile start with sharp 
# target all means all targets currently defined in this file 
all: PSO ODE

# this target is for the executable 
PSO: PSO.o 
	g++ PSO.o -o PSO

# this target is the dependency for PSO.o
PSO.o: PSO.cpp
	g++ -c PSO.cpp
# target dependencies for ODE section
ODE: main.o fileIO.o ODESys.o var.o calc.o
	g++ main.o fileIO.o ODESys.o var.o calc.o -o ODE -fopenmp
main.o: main.cpp ODE.hpp
	g++ -c main.cpp
fileIO.o: fileIO.cpp ODE.hpp
	g++ -c fileIO.cpp
ODESys.o: ODESys.cpp ODE.hpp
	g++ -c ODESys.cpp
var.o: var.cpp ODE.hpp
	g++ -c var.cpp
calc.o: calc.cpp ODE.hpp
	g++ -c calc.cpp
para: para.o
	g++ para.o -o para -fopenmp
para.o: para.cpp
	g++ -c para.cpp -o para.o -fopenmp
# this target deletes all files produced from the Makefile
# so that a completely new compile of all items is required
clean:
	rm -rf *.o PSO ODE
