#include "ODE.hpp"

/* Collect data functions for matrix Pav */

/* Only to be used with integrate_const */
void sample_const( const state_type &c , const double t){
    int row = t/10;
    oFile1 << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl; 
    // Columns filled in matrix: t, c[0], c[1], c[2]
    pAvg(row,0) = t;
    pAvg(row,1) += c[0] / N;
    pAvg(row,2) += c[1] / N;
    pAvg(row,3) += c[2] / N;
}
/* Only to be used with integrate/integrate_adaptive */
void sample_adapt( const state_type &c , const double t){

}