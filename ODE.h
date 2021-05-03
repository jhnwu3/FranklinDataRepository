#include <iostream>
#include <boost/array.hpp>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <vector>
#include <eigen/Dense>
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
ofstream oFile; 
ofstream oFile1; 
ofstream oFileGNU; 
void open_files();
void close_files();
void write_file( const state_type &c , const double t );
void write_file_const( const state_type &c , const double t );

/* Collect data functions */
void sample_const( const state_type &c , const double t, int i );
void sample_adapt( const state_type &c , const double t, int i );
/* ODE Systems Functions */
void tripleNonlinearODE( const state_type &c , state_type &dcdt , double t );

/* model global diff eq. constants */
double extern ke , kme, kf, kmf, kd, kmd, ka2, ka3, C1T, C2T, C3T;
/* vars */
int extern N;
/* time conditions */
double extern x0, xf, dxdt;

MatrixXd extern pAvg( (int) xf/10, 4);
/* Uniform Random Number Generator */
std::random_device rand_dev;
std::mt19937 generator(rand_dev());
uniform_real_distribution<double> unifDist(0.0, 1.0);