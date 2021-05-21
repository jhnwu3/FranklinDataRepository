/* This is a test cpp script to test out openMP and show how to use it to solve ODEs, it solves a simple 3 linear ODE system that we eventually use in the actual ODE.cpp, which
may or may not be renamed to main.cpp at some point. 
 */



#include <iostream>
#include <boost/array.hpp>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <chrono>
#include <omp.h>
#define N_SPECIES 3 // using #defines technically right now, but will eventually change it to a variable in main
#define N 10000
#define N_DIM 5
#define N_PARTICLES 5


/* namespaces for ease of use */
using namespace std;
using namespace Eigen;
using namespace boost::numeric::odeint;

/* typedefs for boost ODE-ints*/
typedef boost::numeric::ublas::vector< double > vector_type;
typedef boost::numeric::ublas::matrix< double > matrix_type;
typedef boost::array< double , N_SPECIES > state_type;
typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;

/* time conditions, t0 = start time, tf = final time, dt = time step*/
double t0 = 0.0, tf = 3.0, dt = 0.2, tn = 3.0;
double k1 = 0.276782, k2 = 0.8370806, k3 = 0.443217, k4 = 0.04244124, k5 = 0.304645; // Bill's true k
int i = 0;
int minimum = 100;

/* moment vector */
VectorXd mVecTrue = VectorXd::Zero(N_SPECIES*(N_SPECIES + 3) / 2); // for some t
/* Second moment matrix. */
MatrixXd m2Mat = MatrixXd::Zero(N_SPECIES, N_SPECIES); // secomd moment vector

struct K
{
    array<double, N_DIM> k;
};

class Particle_Linear
{
    struct K T1;

public:
    Particle_Linear(struct K G) : T1(G) {} 

    void operator() (  const state_type &c , state_type &dcdt , double t)
    {
        MatrixXd kr(N_SPECIES, N_SPECIES); 
        kr << 0, T1.k[1], T1.k[3],
            T1.k[2], 0, T1.k[0],
            0, T1.k[4], 0;
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
};
/* Only to be used with integrate/integrate_adaptive - nonlinear */
void sample_adapt( const state_type &c , const double t){
    /* We will have some time we are sampling towards */
    if(t == tf){
        
        for(int row = 0; row < N_SPECIES; row++){
            mVecTrue(row) += c[row]; // store all first moments in the first part of the moment vec
            for(int col = row; col < N_SPECIES; col++){
                m2Mat(row,col) += (c[row] * c[col]);   // store in a 2nd moment matrix
            }
        }
    }
    if( c[0] - c[1] < 1e-10 && c[0] - c[1] > -1e-10){
        cout << "Out of bounds!" << endl;
        return; // break out for loop
    }
}
/* Example Streaming Observer Format */
struct Particle_Observer
{
    VectorXd& momentVector; // note: Unfortunately, VectorXd from Eigen is far more complicated?
    Particle_Observer( VectorXd& vec) : momentVector( vec ){}
    void operator()( const state_type &c , const double t ) 
    {
        if(t == tf){
            for(int row = 0; row < N_SPECIES; row++){
                momentVector(row) += c[row]; 
                for(int col = row; col < N_SPECIES; col++){
                    if( row == col){
                        momentVector(N_SPECIES + row) += c[row] * c[col];
                    }else{
                        momentVector(2*N_SPECIES + (row + col - 1)) += c[row] *c[col];
                    }
                }
      
            }
        }
    }
};





void doStuff(){
    for(int j = 0; j< 20000; j++){

    }
}
/* Test finding min function */
int main (){
    double mu_x = 1.47, mu_y = 1.74, mu_z = 1.99; // true means for MVN(theta)
    /* Random Number Generator */
    random_device rand_dev;
    mt19937 generator(rand_dev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);
     /* ODE solver variables! */
    VectorXd initCon(N_SPECIES); // temp vector to be used for initiation conditions
    state_type c0;
    controlled_stepper_type controlled_stepper;
     /* Variables used for multivariate log normal distribution */
    VectorXd mu(N_SPECIES);
    MatrixXd sigma  = MatrixXd::Zero(N_SPECIES, N_SPECIES);
    /* assign mu vector and sigma matrix values   */
    mu << mu_x, mu_y, mu_z;
    sigma << 0.77, 0.0873098, 0.046225, 
             0.0873098, 0.99, 0.104828, 
             0.046225, 0.104828, 1.11; 

    #pragma omp parallel for
        for(i = 0; i < N; i++){
        
            doStuff();
            #pragma omp critical
                {
                    
                }
        } 


    cout << "min:" << minimum << endl;
    return EXIT_SUCCESS;
}