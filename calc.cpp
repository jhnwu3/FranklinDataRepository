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

/* mVec = first moment vector (may or may not have other vector components
   m2 = second moment vector
    cov = value returned after calculating cov matrix
*/
MatrixXd calculate_covariance_matrix(const MatrixXd& m2, const VectorXd& mVec, int nProt){
    MatrixXd cov(nProt, nProt);
     /* calculate covariance matrix */
    for(int row = 0; row < nProt; row++){
        for(int col = 0; col < nProt; col++){
            cov(row, col) = m2(row,col) - mVec(row)*mVec(col);
        }
    }
    return cov;
}