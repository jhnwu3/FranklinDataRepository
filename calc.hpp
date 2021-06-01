#ifndef _CALC_HPP_
#define _CALC_HPP_

#include "main.hpp"
/* Calculation Functions */
double CF1(const VectorXd& trueVec, const VectorXd& estVec, int n);
double CF2(const VectorXd& trueVec, const VectorXd& estVec, const MatrixXd& w, int n);
MatrixXd calculate_covariance_matrix(const MatrixXd& m2, const VectorXd& mVec, int nProt);
MatrixXd create_covariance_matrix(const MatrixXd& sampleSpace, const VectorXd& mu, int nProt);
MatrixXd generate_sample_space(int nProt, int n);
MatrixXd calculate_weight_matrix(const MatrixXd &sample, const VectorXd &mu, int nMom, int n);

#endif