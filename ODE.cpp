#include "main.hpp"
#include "ODE.hpp"

void nonlinearODE3( const State_N &c , State_N &dcdt , double t )
{
    dcdt[0] =  ((ke*(C1T - c[0]))/(kme + (C1T - c[0]))) + ((kf * (C1T - c[0]) * c[0] * c[1]) / (kmf + (C1T - c[0]))) - ((kd*c[0]*c[2])/(kmd + c[0])); // dc1dt = ke*(C1T-C1).... (in document)
    dcdt[1] =  ka2 *(C2T - c[1]); // dc2/dt = ka2 * (C2T - c2)
    dcdt[2] =  ka3*(C3T - c[2]); // dc3/dt = ka3 * (C3t - c3)
}

/* Test ODE function for dynamic number of ODE's, unsure how to get proper linear one.*/
void linearODEn_1( const State_N &c , State_N &dcdt , double t )
{ 
    
    MatrixXd kr(N_SPECIES, N_SPECIES); 
    random_device rand_dev;
    mt19937 generator(rand_dev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);
    /* Form pseudo random k rate constants (note: need to figure out a way for a global rate vector)*/
    for(int i = 0; i < N_SPECIES - 1; i++){
        for(int j = i + 1; j < N_SPECIES; j++){
            kr(i,j) = 0.1 * unifDist(generator);
            kr(j,i) = kr(i,j);
        }
    }

    /*for(int i = 0; i < N_SPECIES; i++){
        for(int j = 0; j < N_SPECIES; j++){
            dcdt[i] += kr(i,j) * c[j] - kr(j,i) * c[i];  
        }
    }*/
    int j = 0;
    for(int i = 0; i < N_SPECIES; i++){
        dcdt[i] = kr(i,j) *c[j] - kr(j,i) * c[i];
        j++; 
    }
}
/* Test for 6 systems */
void nonlinearODE6( const State_N &c , State_N &dcdt , double t){
    dcdt[0] =  ((ke*(C1T - c[0])) / (kme + (C1T - c[0]))) + 
               ((kf * (C1T - c[0]) * c[0] * c[1]) / (kmf + (C1T - c[0]))) - 
               ((kd*c[0]*c[2]) / (kmd + c[0])) +
               ((kf * (C1T - c[0]) * c[0] * c[3]) / (kmf + (C1T - c[0])))- 
               ((kd*c[0]*c[4]) / (kmd + c[0])) +
               ((kf * (C1T - c[0]) * c[0] * c[5]) / (kmf + (C1T - c[0]))); // dc1dt = ke*(C1T-C1).... (in document)

    dcdt[1] =  ka2 *(C2T - c[1]); // dc2/dt = ka2 * (C2T - c2)
    dcdt[2] =  ka3*(C3T - c[2]); // dc3/dt = ka3 * (C3t - c3)
    dcdt[3] =  ka3*(C3T - c[3]); // dc3/dt = ka3 * (C3t - c3)
    dcdt[4] =  ka3*(C3T - c[4]); // dc3/dt = ka3 * (C3t - c3)
    dcdt[5] =  ka3*(C3T - c[5]); // dc3/dt = ka3 * (C3t - c3)
}

