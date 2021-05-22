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
#define N_SPECIES 3 // using #defines technically right now, but will eventually change it to a variable in main
#define N 10000
#define N_DIM 5
#define N_PARTICLES 5

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
void sample_adapt_linear( const state_type &c , const double t);

/* ODE Systems Functions */
void nonlinearODE3( const state_type &c , state_type &dcdt , double t );
void linearODE3_true( const state_type &c , state_type &dcdt , double t );
void linearODEn_1( const state_type &c , state_type &dcdt , double t );
void nonlinearODE6( const state_type &c , state_type &dcdt , double t);

/* Calculation Functions */
double CF1(const VectorXd& trueVec, const VectorXd& estVec, int n);
double CF2(const VectorXd& trueVec, const VectorXd& estVec, const MatrixXd& w, int n);
MatrixXd calculate_covariance_matrix(const MatrixXd& m2, const VectorXd& mVec, int nProt);
MatrixXd create_covariance_matrix(const MatrixXd& sampleSpace, const VectorXd& mu, int nProt);
MatrixXd generate_sample_space(int nProt, int n);
/* Note: We have global variables in this case for ease of access by ODEINT solvers:*/
/* model global diff eq. constants */
double extern ke, kme, kf, kmf, kd, kmd, ka2, ka3, C1T, C2T, C3T;
/* Bill's K */
double extern k1, k2, k3, k4, k5;
/* global time conditions */
double extern t0, tf, dt, tn;
/* Normal Dist Vars */
double extern mu_x, mu_y, mu_z; // true means for MVN(theta)
double extern var_x, var_y, var_z; // true variances for MVN(theta);
double extern rho_xy, rho_xz, rho_yz; // true correlations for MVN
double extern sigma_x, sigma_y, sigma_z;

/* Need to figure out how to clean up this code below later */

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

/* The rhs of x' = f(x) defined as a class */
// define structure
struct K
{
    VectorXd k(N_DIM);
};

/* ODE- System to be used for parallel computing for particles */
class Particle_Linear
{
    struct K T1;

public:
    Particle_Linear(struct K G) : T1(G) {}

    void operator() (  const state_type &c , state_type &dcdt , double t)
    {
        MatrixXd kr(N_SPECIES, N_SPECIES); 
        kr << 0, T1.k(1), T1.k(3),
            T1.k(2), 0, T1.k(0),
            0, T1.k(4), 0;
        dcdt[0] = (kr(0,0) * c[0] - kr(0,0) * c[0]) +
              (kr(0,1) * c[1] - kr(1,0) * c[0]) + 
              (kr(0,2) * c[2] - kr(2,0) * c[0]);

        dcdt[1] = (kr(1,0) * c[0] - kr(0,1) * c[1]) +
                (kr(1,1) * c[1] - kr(1,1) * c[1]) + 
                (kr(1,2) * c[2] - kr(2,1) * c[1]);

        dcdt[2] = (kr(2,0) * c[0] - kr(0,2) * c[2]) + 
                (kr(2,1) * c[1] - kr(1,2) * c[2]) + 
                (kr(2,2) * c[2] - kr(2,2) * c[2]);
    }
};

/* Example Streaming Observer Format 
struct streaming_observer
{
    std::ostream& m_out;

    streaming_observer( std::ostream &out ) : m_out( out ) { }

    template< class State >
    void operator()( const State &x , double t ) const
    {
        container_type &q = x.first;
        m_out << t;
        for( size_t i=0 ; i<q.size() ; ++i ) m_out << "\t" << q[i];
        m_out << "\n";
    }
}; */

/* Example Streaming Observer Format */
struct Particle_Observer
{
    VectorXd& momentVector; // note: Unfortunately, VectorXd from Eigen is far more complicated?
    Particle_Observer( VectorXd& vec) : momentVector( vec ){}
    void operator()( const state_type &c , const double t ) 
    {
        if(t == tf){
            for(int row = 0; row < N_SPECIES; row++){
                momentVector(row) += c[row]; 
                for(int col = row; col < N_SPECIES; col++){
                    if( row == col){
                        momentVector(N_SPECIES + row) += c[row] * c[col];
                    }else{
                        momentVector(2*N_SPECIES + (row + col - 1)) += c[row] *c[col];
                    }
                }
            }
        }
    }
}; 
