#include "ODE.hpp"
/*Global Variables for ease of access by all functions - will eventually slowly remove all of these. */
/* diff eq. constants definitions */
const double ke = 0.0001, kme = 20, kf = 0.01, kmf = 18, kd = 0.03, kmd = 1, 
ka2 = 0.01, ka3 = 0.01, C1T = 20, C2T = 5, C3T = 4;
const double k1 = 0.276782, k2 = 0.8370806, k3 = 0.443217, k4 = 0.04244124, k5 = 0.304645; // Bill's true k
/* time conditions, t0 = start time, tf = final time, dt = time step*/
const double t0 = 0.0, tf = 3.0, dt = 0.2, tn = 3.0;

/* Given avgs and vars about the distributions of vals being given from PSO Stewart */
const double mu_x = 1.47, mu_y = 1.74, mu_z = 1.99; // true means for MVN(theta)
const double var_x = 0.77, var_y = 0.99, var_z = 1.11; // true variances for MVN(theta);
const double rho_xy = 0.10, rho_xz = 0.05, rho_yz = 0.10; // true correlations for MVN
const double sigma_x = sqrt(var_x), sigma_y = sqrt(var_y), sigma_z = sqrt(var_z);
