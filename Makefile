# BY SUBMITTING THIS FILE TO CARMEN, I CERTIFY THAT I HAVE STRICTLY ADHERED
# TO THE TENURES OF THE OHIO STATE UNIVERSITY’S ACADEMIC INTEGRITY POLICY WITH
# RESPECT TO THIS ASSIGNMENT.

# comments in a Makefile start with sharp 
gcc_opt = -ansi -pedantic -g -c -D DEBUG 

# target all means all targets currently defined in this file (lab1.zip and lab1)
all: lab1

# this target is for the lab1 executable 
# the lab1 target gets recreated if lab1.o has changed
PSO: PSO.o 
#this is the linux command we want make to execute
#if lab1.o has changed, note that we only use the -o option here
	g++ PSO.o -o PSO

# this target is the dependency for lab1.o
# the lab1.o target gets recreated if lab1.c has changed
PSO.o: PSO.cpp
	g++ -c PSO.cpp

# this target deletes all files produced from the Makefile
# so that a completely new compile of all items is required
clean:
	rm -rf *.o PSO
