#include "ODE.hpp"

void nonlinearODE3( const state_type &c , state_type &dcdt , double t )
{
    dcdt[0] =  ((ke*(C1T - c[0]))/(kme + (C1T - c[0]))) + ((kf * (C1T - c[0]) * c[0] * c[1]) / (kmf + (C1T - c[0]))) - ((kd*c[0]*c[2])/(kmd + c[0])); // dc1dt = ke*(C1T-C1).... (in document)
    dcdt[1] =  ka2 *(C2T - c[1]); // dc2/dt = ka2 * (C2T - c2)
    dcdt[2] =  ka3*(C3T - c[2]); // dc3/dt = ka3 * (C3t - c3)
}
/* Try something new */
void linearODE3( const state_type &c , state_type &dcdt , double t )
{
    MatrixXd kr(N_SPECIES, N_SPECIES); 
    kr << 0, k2, k4,
            k3, 0, k1,
            0, k5, 0;
   // double kr[nProteins][nProteins] = {{0, k2, k4}, {k3, 0, k1}, {0, k5, 0}};

    dcdt[0] = (kr(0,0) * c[0] - kr(0,0) * c[0]) +
              (kr(0,1) * c[1] - kr(1,0) * c[0]) + 
              (kr(0,2) * c[2] - kr(2,0) * c[0]);

    dcdt[1] = (kr(1,0) * c[0] - kr(0,1) * c[1]) +
              (kr(1,1) * c[1] - kr(1,1) * c[1]) + 
              (kr(1,2) * c[2] - kr(2,1) * c[1]);

    dcdt[2] = (kr(2,0) * c[0] - kr(0,2) * c[2]) + 
              (kr(2,1) * c[1] - kr(1,2) * c[2]) + 
              (kr(2,2) * c[2] - kr(2,2) * c[2]);
}
void linearODEn_1( const state_type &c , state_type &dcdt , double t )
{ 
    dcdt[0] = 0.001*(c[0]);
    for(int i = 1; i < N_SPECIES; i++){
        dcdt[i] =  0.001*(c[i] - c[i - 1]);
    }
}
/* Test for 6 systems */
void nonlinearODE6( const state_type &c , state_type &dcdt , double t){
    dcdt[0] =  ((ke*(C1T - c[0])) / (kme + (C1T - c[0]))) + 
               ((kf * (C1T - c[0]) * c[0] * c[1]) / (kmf + (C1T - c[0]))) - 
               ((kd*c[0]*c[2]) / (kmd + c[0])) + c[3] * c[4] * c[5]; // dc1dt = ke*(C1T-C1).... (in document)

    dcdt[1] =  ka2 *(C2T - c[1]); // dc2/dt = ka2 * (C2T - c2)
    dcdt[2] =  ka3*(C3T - c[2]) * c[4]; // dc3/dt = ka3 * (C3t - c3)
    dcdt[3] =  ka3*(C3T - c[3]) * c[2]; // dc3/dt = ka3 * (C3t - c3)
    dcdt[4] =  ka3*(C3T - c[4]) * c[0]; // dc3/dt = ka3 * (C3t - c3)
    dcdt[5] =  ka3*(C3T - c[5]) * c[1]; // dc3/dt = ka3 * (C3t - c3)
}