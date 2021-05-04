#include "ODE.hpp"

int main(int argc, char **argv)
{
    /* Uniform Random Number Generator */
    random_device rand_dev;
    mt19937 generator(rand_dev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);
    
    open_files();
    state_type c0 = {10.0 , 0.0 , 0.0 };
    controlled_stepper_type controlled_stepper;
    /* average randomized sample/initial conditions from unif dist, N=10,000 */
   for(int i = 0; i < N; i++){
       c0 = {10.0*unifDist(generator), unifDist(generator), unifDist(generator)};
       integrate_const(controlled_stepper, tripleNonlinearODE, c0, t0, tf, dt, sample_const);
   }
    close_files();
}

// examples of integrate functions:
// integrate(tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);
// integrate_adaptive(controlled_stepper, tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);