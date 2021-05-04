#include <iostream>
#include <boost/array.hpp>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <vector>
#include <Eigen/Dense>
#include <random>

/* namespaces for ease of use */
using namespace std;
using namespace boost::numeric::odeint;
using Eigen::MatrixXd;
using Eigen::VectorXd;
/* typedefs for boost ODE-ints*/
typedef boost::numeric::ublas::vector< double > vector_type;
typedef boost::numeric::ublas::matrix< double > matrix_type;
typedef boost::array< double , 3 > state_type;
typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;

/* file io */
ofstream extern oFile; 
ofstream extern oFile1; 
ofstream extern oFileGNU; 
void open_files();
void close_files();
void write_file( const state_type &c , const double t );
void write_file_const( const state_type &c , const double t );

/* Collect data functions */
void sample_const( const state_type &c , const double t);
void sample_adapt( const state_type &c , const double t);
/* ODE Systems Functions */
void tripleNonlinearODE( const state_type &c , state_type &dcdt , double t );

/* model global diff eq. constants */
double extern ke , kme, kf, kmf, kd, kmd, ka2, ka3, C1T, C2T, C3T;
/* vars */
int extern N;
/* time conditions */
double extern t0, tf, dt;

MatrixXd extern pAvg;
/* Uniform Random Number Generator */
random_device extern rand_dev;
mt19937 extern generator(rand_dev());
uniform_real_distribution<double> extern unifDist(0.0, 1.0);