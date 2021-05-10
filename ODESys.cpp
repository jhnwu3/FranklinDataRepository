#include "ODE.hpp"
MatrixXd kr;

void tripleNonlinearODE( const state_type &c , state_type &dcdt , double t )
{
    dcdt[0] =  ((ke*(C1T - c[0]))/(kme + (C1T - c[0]))) + ((kf * (C1T - c[0]) * c[0] * c[1]) / (kmf + (C1T - c[0]))) - ((kd*c[0]*c[2])/(kmd + c[0])); // dc1dt = ke*(C1T-C1).... (in document)
    dcdt[1] =  ka2 *(C2T - c[1]); // dc2/dt = ka2 * (C2T - c2)
    dcdt[2] =  ka3*(C3T - c[2]); // dc3/dt = ka3 * (C3t - c3)
}
void tripleLinearODE( const state_type &c , state_type &dcdt , double t )
{
   
    dcdt[0] =  
    dcdt[1] =  
    dcdt[2] =  
}