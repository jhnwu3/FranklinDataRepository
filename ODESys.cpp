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
    
    MatrixXd kr(N_SPECIES, N_SPECIES); 
    VectorXd mu(N_SPECIES);
    random_device rand_dev;
    mt19937 generator(rand_dev());
    std::normal_distribution<double> xNorm(mu_x, sigma_x);
    std::normal_distribution<double> yNorm(mu_y, sigma_y);
    std::normal_distribution<double> zNorm(mu_z, sigma_z);
    /* Form pseudo random k rate constants (note: need to figure out a way for a global rate vector)*/
    for(int i = 0; i < N_SPECIES; i++){
        if (i % 3 == 0 ) {
            mu(i) = xNorm(generator);
        }else if (i % 3 == 1) {
            mu(i) = yNorm(generator);
        }else {
            mu(i) = zNorm(generator);
        }
    }
    cout << "lol" << endl;
    for(int i = 0; i < N_SPECIES; i++){
        for(int j = 0; i < N_SPECIES; j++){
            kr(i,j) = exp(mu(i) - mu(j));
        }
    }

    for(int i = 0; i < N_SPECIES; i++){
        for(int j = 0; j < N_SPECIES; j++){
            dcdt[i] += kr(i,j) * c[j] - kr(j,i) * c[i];  
        }
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