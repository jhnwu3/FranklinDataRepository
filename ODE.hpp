#include <iostream>
#include <boost/array.hpp>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <chrono>
#define N_SPECIES 6 // 


/* namespaces for ease of use */
using namespace std;
using namespace Eigen;
using namespace boost::numeric::odeint;

/* typedefs for boost ODE-ints*/
typedef boost::numeric::ublas::vector< double > vector_type;
typedef boost::numeric::ublas::matrix< double > matrix_type;
typedef boost::array< double , N_SPECIES > state_type;
typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;

/* FILE IO */
void open_files();
void close_files();
void write_file( const state_type &c , const double t );
void write_file_const( const state_type &c , const double t );

/* Collect data functions */
void sample_const( const state_type &c , const double t);
void sample_adapt( const state_type &c , const double t);

/* ODE Systems Functions */
void nonlinearODE3( const state_type &c , state_type &dcdt , double t );
void linearODE3( const state_type &c , state_type &dcdt , double t );
void linearODEn_1( const state_type &c , state_type &dcdt , double t );
void nonlinearODE6( const state_type &c , state_type &dcdt , double t);

/* Calculation Functions */
double kCost(const VectorXd& kTrueVec, const VectorXd& kEstVec, int n);
double kCostMat(const VectorXd& kTrueVec, const VectorXd& kEstVec, const MatrixXd& w, int n);
MatrixXd calculate_covariance_matrix(const MatrixXd& m2, const VectorXd& mVec, int nProt);

/* Note: We have global variables in this case for ease of access by ODEINT solvers:*/
/* model global diff eq. constants */
double extern ke, kme, kf, kmf, kd, kmd, ka2, ka3, C1T, C2T, C3T;
/* Bill's K */
double extern k1, k2, k3, k4, k5;
/* vars */
int extern N;
/* global time conditions */
double extern t0, tf, dt, tn;
/* Normal Dist Vars */
double extern mu_x, mu_y, mu_z; // true means for MVN(theta)
double extern var_x, var_y, var_z; // true variances for MVN(theta);
double extern rho_xy, rho_xz, rho_yz; // true correlations for MVN
double extern sigma_x, sigma_y, sigma_z;

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

