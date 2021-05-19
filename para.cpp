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

#define N_PARTICLES 5
using namespace std;
void printHello(){
    cout << "Hello" << endl;
}
int main (){
    int i;
    cout << "Before par. for " << endl;

#pragma omp parallel
{
    cout << "Printing from thread:" << omp_get_thread_num() << "index:" << i << endl; 
}
    
    return EXIT_SUCCESS;
}