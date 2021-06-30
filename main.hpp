#ifndef _MAIN_HPP_
#define _MAIN_HPP_

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
#define N_SPECIES 6
#define N 1000 // # of samples to sample over
#define N_DIM 6 // dim of PSO hypercube
#define N_PARTICLES 20 

/* namespaces for ease of use */
using namespace std;
using namespace Eigen;
using namespace boost::numeric::odeint;

/* typedefs for boost ODE-ints */
typedef boost::array< double , N_SPECIES > State_N;
typedef runge_kutta_cash_karp54< State_N > Error_RK_Stepper_N;
typedef controlled_runge_kutta< Error_RK_Stepper_N > Controlled_RK_Stepper_N;

typedef boost::array< double , 6 > State_6;
typedef runge_kutta_cash_karp54< State_6 > Error_RK_Stepper_6;
typedef controlled_runge_kutta< Error_RK_Stepper_6 > Controlled_RK_Stepper_6;

/* Collect data functions - in main for ease of access - @TODO clean up in other files! */
void sample_const( const State_N &c , const double t);
void sample_adapt( const State_N &c , const double t);
void sample_adapt_linear( const State_N &c , const double t);

#endif

