# PSOLinux
PSO - An Implementation of Particle Swarm Optimization, in progress.

- <insert more contextual info and function here >

ODE.cpp - main file 
ODESys.cpp - contains ode systems that are tested
calc.cpp - contains all calculation functions used and functions needed for rng generation
fileIO.cpp - also contains N/A atm
var.cpp - contains all necessary global variables used


On the topic of ODESys:

Unfortunately, there's no easy way to store all the ODE system in one system .cpp file, because each particle that needs to be run in parallel needs to have a global (ever-changing) rate constant vector that needed to be accessed by the ODESys somehow.