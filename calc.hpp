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



/* Struct for multi-variate normal distribution */
struct normal_random_variable
{
    normal_random_variable(Eigen::MatrixXd const& covar)
        : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
    }
};
#endif