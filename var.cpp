#include "ODE.hpp"

/* diff eq. constants definitions */
double ke = 0.0001, kme = 20, kf = 0.01, kmf = 18, kd = 0.03, kmd = 1, 
ka2 = 0.01, ka3 = 0.01, C1T = 20, C2T = 5, C3T = 4;

/* Number of proteins & columns/diff eq solutions for ODEs */
int N = 10000, pCol = 4; 
/* time conditions, t0 = start time, tf = final time, dt = time step*/
double t0 = 0.0, tf = 500.0, dt = 10.0;
