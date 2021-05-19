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
#define N 10
#define N_DIM 5


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





int main (){

    cout << "Before par. for " << endl;
    int sum = 0;
#pragma omp parallel for
    for(int i = 0; i < N; i++){
        cout << "Printing from thread:" << omp_get_thread_num() << "index:" << i << endl; 
        sum += i;
    }

    cout << endl << endl << "final sum: " << sum;
    return EXIT_SUCCESS;
}