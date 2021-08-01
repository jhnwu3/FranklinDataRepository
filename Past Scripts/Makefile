all: PSO_S PSO_NL3 PSO_S_ODE_L

# this target is for the executable of Dr. Stewarts PSO alg converted to C++ 
# mat exponentiation
PSO_S: PSO_S.o 
	g++ PSO_S.o -o PSO_S
# this target is the dependency for PSO.o
PSO_S.o: PSO_S.cpp
	g++ -c PSO_S.cpp

# PSO_S_ODE but instead of mat exponentation, do nonlinear ODEs!
PSO_S_ODE_NL: PSO_S_ODE_NL.o
	g++ PSO_S_ODE_NL.o -o PSO_S_ODE_NL -fopenmp
PSO_S_ODE_NL.o: PSO_S_ODE_NL.cpp
	g++ -c -O3 -fopenmp PSO_S_ODE_NL.cpp

# PSO_S_ODE but instead of mat exponentation, do nonlinear ODEs!
PSO_S_ODE_L: PSO_S_ODE_L.o
	g++ PSO_S_ODE_L.o -o PSO_S_ODE_L -fopenmp
PSO_S_ODE_L.o: PSO_S_ODE_L.cpp
	g++ -c -O3 -fopenmp PSO_S_ODE_L.cpp