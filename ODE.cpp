#include "ODE.hpp"

ofstream oFile; 
ofstream oFile1; 
ofstream oFileMAV; 

/* Global Matrix, first moment to fill */
MatrixXd m1 = MatrixXd::Zero((int) (tf / dt) + 1, nProt);

/* Second moment matrices time steps. */
MatrixXd m2_1 = MatrixXd::Zero(nProt, nProt); // t = 0
MatrixXd m2_2 = MatrixXd::Zero(nProt, nProt); // t = 250
MatrixXd m2_3 = MatrixXd::Zero(nProt, nProt); // t = 500

/********** File IO **********/

/* open files for writing */
void open_files(){
    oFile.open("ODE_Soln.csv");
    oFile1.open("ODE_Const_Soln.csv"); 
    oFileMAV.open("mAv.csv");
}

/* write data to specific csv functions */
void write_file( const state_type &c , const double t ){ oFile << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl; }
void write_file_const( const state_type &c , const double t ){ oFile1 << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl; }

/* close files */
void close_files(){
    oFile.close();
    oFile1.close();
    oFileMAV.close();
}

/* Only to be used with integrate_const */
void sample_const( const state_type &c , const double t){
    int row = t/10;
    // Fill first moment matrix: t, c[0], c[1], c[2]
    m1(row,0) += c[0] / N;
    m1(row,1) += c[1] / N;
    m1(row,2) += c[2] / N;
    
    /* We will have 4 time steps */
    if(t == 0){
        for(int row = 0; row < nProt; row++){
            for(int col = 0; col < nProt; col++){
                m2_1(row,col) += (c[row] * c[col]) / N;    
            }
        }
    }else if (t == 250){
        for(int row = 0; row < nProt; row++){
            for(int col = 0; col < nProt; col++){
                m2_2(row,col) += (c[row] * c[col]) / N;     
            }
        }
    }else if (t == 500){
        for(int row = 0; row < nProt; row++){
            for(int col = 0; col < nProt; col++){
                m2_3(row,col) += (c[row] * c[col]) / N;
            }
        }
    }
}
/* Only to be used with integrate/integrate_adaptive - @TODO */
void sample_adapt( const state_type &c , const double t){}


int main(int argc, char **argv)
{
    /* Random Number Generator */
    random_device rand_dev;
    mt19937 generator(rand_dev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);
    normal_distribution<double> xNorm(mu_x,sigma_x);
    normal_distribution<double> yNorm(mu_y,sigma_y);
    normal_distribution<double> zNorm(mu_z,sigma_z);

    open_files();
    state_type c0 = {10.0 , 0.0 , 0.0 };
    controlled_stepper_type controlled_stepper;
    /* average randomized sample/initial conditions from unif dist, N=10,000 */
   for(int i = 0; i < N; i++){
       c0 = {xNorm(generator), yNorm(generator), zNorm(generator)};
       integrate_const(controlled_stepper, tripleNonlinearODE, c0, t0, tf, dt, sample_const);
   }

    cout << "alive" << endl;

    oFileMAV << m2_1 << endl << endl << m2_2 << endl << endl << m2_3;
    close_files();
}

// examples of integrate functions:
// integrate(tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);
// integrate_adaptive(controlled_stepper, tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);