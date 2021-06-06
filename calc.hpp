#ifndef _CALC_HPP_
#define _CALC_HPP_

#include "main.hpp"
/* Calculation Functions */
double calculate_cf1(const VectorXd& trueVec, const VectorXd& estVec, int n);
double calculate_cf2(const VectorXd& trueVec, const VectorXd& estVec, const MatrixXd& w, int n);
MatrixXd calculate_covariance_matrix(const MatrixXd& m2, const VectorXd& mVec, int nProt);
MatrixXd create_covariance_matrix(const MatrixXd& sampleSpace, const VectorXd& mu, int nProt);
MatrixXd generate_sample_space(int nProt, int n);
MatrixXd calculate_weight_matrix(const MatrixXd &sample, const VectorXd &mu, int nMom, int n);

/* Struct for multi-variate normal distribution */
struct Multi_Normal_Random_Variable
{
    Multi_Normal_Random_Variable(Eigen::MatrixXd const& covar)
        : Multi_Normal_Random_Variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    Multi_Normal_Random_Variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
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