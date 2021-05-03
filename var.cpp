#include "ODE.h"

/* diff eq. constants definitions */
double ke = 0.0001, kme = 20, kf = 0.01, kmf = 18, kd = 0.03, kmd = 1, 
ka2 = 0.01, ka3 = 0.01, C1T = 20, C2T = 5, C3T = 4;

int N = 10000; // number of items to sample for
/* time conditions */
double x0 = 0.0, xf = 500.0, dxdt = 10.0;
/* Matrix to fill */
MatrixXd pAvg((int) xf / 10, 4);