#include <iostream>
#include <boost/array.hpp>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <vector>
#include <eigen/Dense>
#include <random>
#include "ODE.hpp"

/* open files for writing */
void open_files(){
    oFile.open("ODE_Soln.csv");
    oFile1.open("ODE_Const_Soln.csv"); 
    oFileGNU.open("ODE_Soln");
}


/* write data to specific csv functions */
void write_file( const state_type &c , const double t )
{
    oFile << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl; 
    oFileGNU << t << ' ' << c[0] << ' ' << c[1] << ' ' << c[2] << endl; 
}
void write_file_const( const state_type &c , const double t )
{
    oFile1 << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl; 
}

/* close files */
void close_files(){
    oFile.close();
    oFile1.close();
    oFileGNU.close();
}