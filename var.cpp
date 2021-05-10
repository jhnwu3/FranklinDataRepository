#include "ODE.hpp"

/* diff eq. constants definitions */
double ke = 0.0001, kme = 20, kf = 0.01, kmf = 18, kd = 0.03, kmd = 1, 
ka2 = 0.01, ka3 = 0.01, C1T = 20, C2T = 5, C3T = 4;

/* Number of proteins & columns/diff eq solutions for ODEs */
int N = 10000, nProt = 3; 
/* time conditions, t0 = start time, tf = final time, dt = time step*/
double t0 = 0.0, tf = 500.0, dt = 10.0;

/* Given avgs and vars about the distributions of vals being given from PSO Stewart */
double mu_x = 1.47, mu_y = 1.74, mu_z = 1.99; // true means for MVN(theta)
double var_x = 0.77, var_y = 0.99, var_z = 1.11; // true variances for MVN(theta);
double rho_xy = 0.10, rho_xz = 0.05, rho_yz = 0.10; // true correlations for MVN
double sigma_x = sqrt(var_x), sigma_y = sqrt(var_y), sigma_z = sqrt(var_z);