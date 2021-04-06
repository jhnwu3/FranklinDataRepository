#include <iostream>
#include <boost/array.hpp>
#include <fstream>
#include <boost/numeric/odeint.hpp>

using namespace std;
using namespace boost::numeric::odeint;


 ofstream oFile; 

typedef boost::array< double , 3 > state_type;
typedef runge_kutta_cash_karp54< double > stepper_type;
void tripleNonlinearODE( const state_type &x , state_type &dxdt , double t )
{
    dxdt[0] =   x[0] + 4*x[2]; // dx /dt = x + 4z
    dxdt[1] =  2*x[1]; // dy / dt = 2y
    dxdt[2] =  3*x[0] + x[1] - 3*x[2]; // dz / dt = 3x + y - 3z
}

void write_file( const state_type &x , const double t )
{
    
    oFile << t << ',' << x[0] << ',' << x[1] << ',' << x[2] << endl;
}

int main(int argc, char **argv)
{
    oFile.open("ODE_Soln.csv");
    state_type x = { 10.0 , 1.0 , 1.0 }; // initial conditions
    //integrate_adaptive( make_controlled( 1E-12 , 1E-12 , stepper_type() ) ,
      //                  tripleNonlinearODE , x , 1.0 , 10.0 , 0.1 , write_file );
    oFile.close();
}