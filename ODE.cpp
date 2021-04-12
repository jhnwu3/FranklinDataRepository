#include <iostream>
#include <boost/array.hpp>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <vector>
using namespace std;
using namespace boost::numeric::odeint;


 ofstream oFile; 
typedef boost::numeric::ublas::vector< double > vector_type;
typedef boost::numeric::ublas::matrix< double > matrix_type;
//typedef boost::vector <double> state_type;
typedef boost::array< double , 3 > state_type;
typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
struct rhs_van 
{
    void operator()( const state_type &x , vector_type &dxdt , double /* t */ )
    {
        dxdt[0] =  -0.04*x[0] + 10000*x[1] *x[2]; // dx1 /dt = -.04 * x1 + 10^4 * x2*x3
        dxdt[1] =  0.04*x[0] - 10000 * x[1]*x[2]-3*10e7*x[1]*x[1]; // dx2/dt = 0.04*x1 - 10^4 * x2*x3 - 3 *10^7 * x2^2
        dxdt[2] =  3*10e7*x[1]*x[1]; // dx3 / dt = 3* 10^7*x2^2
    }
};


void tripleNonlinearODE( const state_type &x , state_type &dxdt , double t )
{
    dxdt[0] =  x[0] + 4*x[2]; // dx /dt = x1*x2 - x0
    dxdt[1] =  2*x[1]*x[0]; // dy / dt = 2y*x
    dxdt[2] =  3*x[0] + x[1] - 3*x[2]; // dz / dt = 3x + y - 3z
}

void write_file( const state_type &x , const double t )
{
    
    oFile << t << ',' << x[0] << ',' << x[1] << ',' << x[2] << endl;
}

int main(int argc, char **argv)
{
    
    oFile.open("ODE_Soln.csv");
    /*state_type x = { 10.0 , 1.0 , 1.0 }; // initial conditions
    integrate( tripleNonlinearODE, x, 0.0, 10.0, 0.1, write_file);*/
    state_type x0 = { 10.0 , 1.0 , 1.0 };
    controlled_stepper_type controlled_stepper;
    integrate_adaptive(controlled_stepper,tripleNonlinearODE,x0, 1.0, 10.0, 0.01, write_file);
    oFile.close();
}