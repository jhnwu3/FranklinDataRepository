#include <iostream>
#include <boost/array.hpp>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <vector>
using namespace std;
using namespace boost::numeric::odeint;

/* typedefs */
typedef boost::numeric::ublas::vector< double > vector_type;
typedef boost::numeric::ublas::matrix< double > matrix_type;
typedef boost::array< double , 3 > state_type;
typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;

/* Constants */
const double ke = 0.0001, kme = 20, kf = 0.01, kmf = 18, kd = 0.03, kmd = 1, 
ka2 = 0.01, ka3 = 0.01, C1T = 20, C2T = 5, C3T = 4;

/* file io */
ofstream oFile; 
void write_file( const state_type &c , const double t )
{
    oFile << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl;
    
    cout << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl;
    
}

void tripleNonlinearODE( const state_type &c , state_type &dcdt , double t )
{
    dcdt[0] =  ((ke*(C1T - c[0]))/(kme + (C1T - c[0]))) + ((kf * (C1T - c[0]) * c[0] * c[1]) / (kmf + (C1T - c[0]))) - ((kd*c[0]*c[2])/(kmd + c[0])); // dc1dt = ke*(C1T-C1).... (in document)
    dcdt[1] =  ka2 *(C2T - c[1]); // dc2/dt = ka2 * (C2T - c2)
    dcdt[2] =  ka3*(C3T - c[2]); // dc3/dt = ka3 * (C3t - c3)
}



int main(int argc, char **argv)
{
    oFile.open("ODE_Soln.csv");
    state_type c0 = {10.0 , 0.0 , 0.0 };
    controlled_stepper_type controlled_stepper;
   // integrate_adaptive(controlled_stepper, tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);
    integrate(tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);
    oFile.close();
}