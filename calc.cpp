#include "ODE.hpp"
/* Calculation Functions */
double kCost (const VectorXd& kTrueVec, const VectorXd& kEstVec, int n){
    double cost = 0;
    for(int i = 0; i < n; i++){
        cost += (kEstVec(i) - kTrueVec(i)) * (kEstVec(i) - kTrueVec(i));
    }
    return cost;
}

double kCostMat(const VectorXd& kTrueVec, const  VectorXd& kEstVec, const MatrixXd& w, int n){
    double cost = 0;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
           cost += (kEstVec(i) - kTrueVec(i)) * w(i,j) *(kEstVec(j) - kTrueVec(j));
        }
    }
    return cost;
}
