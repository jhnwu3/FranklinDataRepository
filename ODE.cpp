#include "ODE.hpp"

ofstream oFile; 
ofstream oFile1; 
ofstream oFileGNU; 

/* Global Matrix, first moment to fill! */
MatrixXd mAv = MatrixXd::Zero((int) tf / dt, pCol);

/********** File IO **********/

/* open files for writing */
void open_files(){
    oFile.open("ODE_Soln.csv");
    oFile1.open("ODE_Const_Soln.csv"); 
    oFileGNU.open("ODE_Soln");
}

/* write data to specific csv functions */
void write_file( const state_type &c , const double t ){
    oFile << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl; 
    oFileGNU << t << ' ' << c[0] << ' ' << c[1] << ' ' << c[2] << endl; 
}
void write_file_const( const state_type &c , const double t ){ oFile1 << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl; }

/* close files */
void close_files(){
    oFile.close();
    oFile1.close();
    oFileGNU.close();
}

/* Only to be used with integrate_const */
void sample_const( const state_type &c , const double t){
    int row = t/10;
    oFile1 << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl; 
    // Columns filled in matrix: t, c[0], c[1], c[2]
    mAv(row,0) = t;
    mAv(row,1) += c[0] / N;
    mAv(row,2) += c[1] / N;
    mAv(row,3) += c[2] / N;
}
/* Only to be used with integrate/integrate_adaptive - @TODO */
void sample_adapt( const state_type &c , const double t){}


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