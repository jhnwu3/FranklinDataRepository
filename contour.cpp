#include <iostream>
#include <fstream>
#include <boost/math/distributions.hpp>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <Eigen/StdVector>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>

#define N_SPECIES 6
#define N_DIM 6 // dim of PSO hypercube

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace boost;
using namespace boost::math;
using namespace boost::numeric::odeint;

/* typedefs for boost ODE-ints */
typedef boost::array< double, N_SPECIES > State_N;
typedef runge_kutta_cash_karp54< State_N > Error_RK_Stepper_N;
typedef controlled_runge_kutta< Error_RK_Stepper_N > Controlled_RK_Stepper_N;

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
        static std::mt19937 gen{ std::random_device{}() }; //std::random_device {} ()
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
    }
};

struct K
{
    VectorXd k;
    K(){
        k = VectorXd::Zero(N_SPECIES);
    }
};

class Nonlinear_ODE6
{
    struct K jay;

public:
    Nonlinear_ODE6(struct K G) : jay(G) {}

    void operator() (const State_N& c, State_N& dcdt, double t)
    {
        dcdt[0] = -(jay.k(0) * c[0] * c[1])  // Syk
            + jay.k(1) * c[2]
            + jay.k(2) * c[2];

        dcdt[1] = -(jay.k(0) * c[0] * c[1]) // Vav
            + jay.k(1) * c[2]
            + jay.k(5) * c[5];

        dcdt[2] = jay.k(0) * c[0] * c[1] // Syk-Vav
            - jay.k(1) * c[2]
            - jay.k(2) * c[2];

        dcdt[3] = jay.k(2) * c[2] //pVav
            - jay.k(3) * c[3] * c[4]
            + jay.k(4) * c[5];

        dcdt[4] = -(jay.k(3) * c[3] * c[4]) // SHP1 
            + jay.k(4) * c[5]
            + jay.k(5) * c[5];

        dcdt[5] = jay.k(3) * c[3] * c[4]  // SHP1-pVav
            - jay.k(4) * c[5]
            - jay.k(5) * c[5];
    }
};

struct Protein_Components {
    int index;
    MatrixXd mat;
    VectorXd mVec;
    double timeToRecord;
    Protein_Components(double tf, int mom, int n) {
        mVec = VectorXd::Zero(mom);
        mat = MatrixXd::Zero(n, N_SPECIES);
        timeToRecord = tf;
    }
};

struct Moments_Mat_Obs
{
    struct Protein_Components& dComp;
    Moments_Mat_Obs(struct Protein_Components& dCom) : dComp(dCom) {}
    void operator()(State_N const& c, const double t) const
    {
        if (t == dComp.timeToRecord) {
            int upperDiag = 2 * N_SPECIES;
            for (int i = 0; i < N_SPECIES; i++) {
                dComp.mVec(i) += c[i];
                //cout << "see what's up:" << dComp.mVec.transpose() << endl;
                dComp.mat(dComp.index, i) = c[i];
                for (int j = i; j < N_SPECIES; j++) {
                    if (i == j && (N_SPECIES + i) < dComp.mVec.size()) { // diagonal elements
                        dComp.mVec(N_SPECIES + i) += c[i] * c[j]; // variances
                    }
                    else if (upperDiag < dComp.mVec.size()){
                        dComp.mVec(upperDiag) += c[i] * c[j]; // covariances
                        upperDiag++;
                    }
                }
            }
        }
    }
};

State_N gen_multi_lognorm_iSub(void) {
    State_N c0;
    VectorXd mu(3);
    mu << 4.78334234137469844730960782,
        5.52142091946216110500584912965,
        4.3815581042632114978686130;
    MatrixXd sigma(3, 3);
    sigma << 0.008298802814695093876186221, 0, 0,
        0, 0.0000799968001706564273219830, 0,
        0, 0, 0.000937060821340228802149700;
    Multi_Normal_Random_Variable gen(mu, sigma);
    VectorXd c0Vec = gen();
    int j = 0;
    for (int i = 0; i < N_SPECIES; i++) {
        if (i == 0 || i == 1 || i == 4) { // Syk, Vav, SHP1
            c0[i] = exp(c0Vec(j));
            j++;
        }
        else {
            c0[i] = 0;
        }
    }

    return c0;
}

State_N gen_multi_norm_iSub(void) {
    State_N c0;
    VectorXd mu(3);
    mu << 80,
        120,
        85;
    MatrixXd sigma(3, 3);
    sigma << 50, 0, 0,
        0, 100, 0,
        0, 0, 50.0;
    Multi_Normal_Random_Variable gen(mu, sigma);
    VectorXd c0Vec = gen();
    int j = 0;
    for (int i = 0; i < N_SPECIES; i++) {
        if (i == 0 || i == 1 || i == 4) { // Syk, Vav, SHP1
            c0[i] = c0Vec(j);
            j++;
        }
        else {
            c0[i] = 0;
        }
    }

    return c0;
}
VectorXd gen_multinorm_iVec(void) {
    VectorXd c0(N_SPECIES);
    VectorXd mu(3);
    mu << 80,
        120,
        85;
    MatrixXd sigma(3, 3);
    sigma << 50, 0, 0,
        0, 100, 0,
        0, 0, 50;
    Multi_Normal_Random_Variable gen(mu, sigma);
    VectorXd c0Vec = gen();
    int j = 0;
    for (int i = 0; i < N_SPECIES; i++) {
        if (i == 0 || i == 1 || i == 4) { // Syk, Vav, SHP1
            c0(i) = c0Vec(j);
            j++;
        }else {
            c0[i] = 0;
        }
    }

    return c0;
}
VectorXd gen_multi_lognorm_vecSub(void) {
    VectorXd initVec(N_SPECIES);
    VectorXd mu(3);
    mu << 4.78334234137469844730960782,
        5.52142091946216110500584912965,
        4.3815581042632114978686130;
    MatrixXd sigma(3, 3);
    sigma << 0.008298802814695093876186221, 0, 0,
        0, 0.0000799968001706564273219830, 0,
        0, 0, 0.000937060821340228802149700;
    Multi_Normal_Random_Variable gen(mu, sigma);
    VectorXd c0Vec = gen();
    int j = 0;
    for (int i = 0; i < N_SPECIES; i++) {
        if (i == 0 || i == 1 || i == 4) { // Syk, Vav, SHP1
            initVec(i) = exp(c0Vec(j));
            j++;
        }
        else {
            initVec(i) = 0;
        }
    }
    return initVec;
}
State_N convertInit(const MatrixXd& sample, int index){
    State_N c0 = {sample(index,0), sample(index,1), 0, 0, sample(index,4), 0};
    return c0;
}
VectorXd comp_vel_vec(const VectorXd& posK, int seed, double epsi, double nan, int hone) {
    
    VectorXd rPoint;
    rPoint = posK;
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    vector<int> rand;
    uniform_real_distribution<double> unifDist(0.0, 1.0);
    // for (int i = 0; i < N_DIM; i++) {
    //     rand.push_back(i);
    // }
    // shuffle(rand.begin(), rand.end(), generator); // shuffle indices as well as possible. 
    // int ncomp = rand.at(0);
    // VectorXd wcomp(ncomp);
    // shuffle(rand.begin(), rand.end(), generator);
    // for (int i = 0; i < ncomp; i++) {
    //     wcomp(i) = rand.at(i);
    // }
    int ncomp = posK.size();
    if(unifDist(generator) < 0.75){
        for (int smart = 0; smart < 2; smart++) {
        // int px = wcomp(smart);
            double pos = rPoint(smart);
            if (pos > 1.0 - nan) {
                cout << "overflow!" << endl;
                // while(pos > 1.0){
                //     pos -= 0.001;
                // }
                pos -= epsi;
            }else if (pos < nan) {
                cout << "underflow!"<< pos << endl;
                // while( pos < 0.001){
                //     pos += 0.001;
                // }
                pos += epsi;
                cout << "pos" << posK.transpose() << endl; 
            }
            double alpha = hone * pos; // Component specific
            double beta = hone - alpha; // pos specific
        // cout << "alpha:" << alpha << "beta:" << beta << endl;
            std::gamma_distribution<double> aDist(alpha, 1); // beta distribution consisting of gamma distributions
            std::gamma_distribution<double> bDist(beta, 1);

            double x = aDist(generator);
            double y = bDist(generator);

            rPoint(smart) = (x / (x + y)); 
        }
    }else{
        for (int smart = 0; smart < ncomp; smart++) {
        // int px = wcomp(smart);
            double pos = rPoint(smart);
            if (pos > 1.0 - nan) {
                cout << "overflow!" << endl;
                // while(pos > 1.0){
                //     pos -= 0.001;
                // }
                pos -= epsi;
            }else if (pos < nan) {
                cout << "underflow!"<< pos << endl;
                // while( pos < 0.001){
                //     pos += 0.001;
                // }
                pos += epsi;
                cout << "pos" << posK.transpose() << endl; 
            }
            double alpha = hone * pos; // Component specific
            double beta = hone - alpha; // pos specific
        // cout << "alpha:" << alpha << "beta:" << beta << endl;
            std::gamma_distribution<double> aDist(alpha, 1); // beta distribution consisting of gamma distributions
            std::gamma_distribution<double> bDist(beta, 1);

            double x = aDist(generator);
            double y = bDist(generator);

            rPoint(smart) = (x / (x + y)); 
        }
    }
    
    return rPoint;
}
MatrixXd calculate_omega_weight_matrix(const MatrixXd &sample, const VectorXd &mu){
    MatrixXd inv = MatrixXd::Zero(mu.size(), mu.size());
    VectorXd X = VectorXd::Zero(mu.size());
    
    for(int s = 0; s < sample.rows(); s++){
        int upperDiag = 2 * N_SPECIES;
        for(int row = 0; row < N_SPECIES; row++){
            X(row) = sample(s, row); 
            for(int col = row; col < N_SPECIES; col++){
                if( row == col){
                    X(N_SPECIES + row) = sample(s, row) * sample(s, col);
                }else{
                    X(upperDiag) = sample(s,row) * sample(s,col);
                    upperDiag++;
                }
            }
        }
        for(int i = 0; i < mu.size(); i++){
            for(int j = 0; j < mu.size(); j++){
                inv(i,j) += (X(i) - mu(i)) * (X(j) - mu(j));
            }
        }
    }
    inv /= sample.rows();
    inv = inv.completeOrthogonalDecomposition().pseudoInverse();
    return inv;
}
double calculate_cf1(const VectorXd& trueVec, const VectorXd& estVec) {
    double cost = 0;
    VectorXd diff(trueVec.size());
    diff = trueVec - estVec;
    cost = diff.transpose() * diff.transpose().transpose();
    return cost;
}
double calculate_cf2(const VectorXd& trueVec, const  VectorXd& estVec, const MatrixXd& w) {
    double cost = 0;
    VectorXd diff(trueVec.size());
    diff = trueVec - estVec;
    cost = diff.transpose() * w * (diff.transpose()).transpose();
    return cost;
}
string removeWhiteSpace(string current)
{
  string myNewString = "";
  string temp = "x";
  for (char c : current)
  {
    if (temp.back() != ' ' && c == ' ')
    {
      myNewString.push_back(' ');
    }
    temp.push_back(c);
    if (c != ' ')
    {
      myNewString.push_back(c);
    }
  }
  return myNewString;
}

string findDouble(string line, int startPos) {
    string doble;
    int i = startPos;
    int wDist = 0;
    while (i < line.length() && !isspace(line.at(i)) && line.at(i) != '\t') {
        i++;
        wDist++;
    }
    doble = line.substr(startPos, wDist);

    return doble;
}
MatrixXd readIntoMatrix(ifstream& in, int rows, int cols) {
    MatrixXd mat(rows, cols);
    // use first row to determine how many columns to read.
    for (int i = 0; i < rows; i++) {
        string line;
        if (in.is_open()) {
            getline(in, line);
            line = removeWhiteSpace(line);
            int wordPos = 0;
            for (int j = 0; j < cols; j++) {
                string subs = findDouble(line, wordPos);
                mat(i, j) = stod(subs);
                wordPos += subs.length() + 1;
            }
        }
        else {
            cout << "Error: File Closed!" << endl;
        }

    }
    return mat;
}
MatrixXd customWtMat(const MatrixXd& Yt, const MatrixXd& Xt, int nMoments, int N, const VectorXd& subCol){
    /* first moment differences */
    MatrixXd fmdiffs = Yt - Xt; 
    /* second moment difference computations - @todo make it variable later */
    MatrixXd smdiffs(N, N_SPECIES);
    for(int i = 0; i < N_SPECIES; i++){
        smdiffs.col(i) = (Yt.col(i).array() * Yt.col(i).array()) - (Xt.col(i).array() * Xt.col(i).array());
    }

   
    int nCross = nMoments - 2 * N_SPECIES;
    if (nCross < 0){
        nCross = 0;
    }
    MatrixXd cpDiff(N, nCross);
    
    /* cross differences */
    if(nCross > 0){
        int upperDiag = 0;
        for(int i = 0; i < N_SPECIES; i++){
            for(int j = i + 1; j < N_SPECIES; j++){
                cpDiff.col(upperDiag) = (Yt.col(i).array() * Yt.col(j).array()) - (Xt.col(i).array() * Xt.col(j).array());
                upperDiag++;
            }
        }
    }
    MatrixXd aDiff(N, nMoments);
    for(int i = 0; i < N; i++){
        for(int moment = 0; moment < nMoments; moment++){
            if(moment < N_SPECIES){
                aDiff(i, moment) = fmdiffs(i, moment);
            }else if (moment >= N_SPECIES && moment < 2 * N_SPECIES){
                aDiff(i, moment) = smdiffs(i, moment - N_SPECIES);
            }else if (moment >= 2 * N_SPECIES){
                aDiff(i, moment) = cpDiff(i, moment - (2 * N_SPECIES));
            }
        }
    }
    double cost = 0;
    VectorXd means = aDiff.colwise().mean();
    VectorXd variances(nMoments);
    for(int i = 0; i < nMoments; i++){
        variances(i) = (aDiff.col(i).array() - aDiff.col(i).array().mean()).square().sum() / ((double) aDiff.col(i).array().size() - 1);
    }
    int rank = subCol.size();
    MatrixXd wt = MatrixXd::Zero(rank, rank);

    for(int i = 0; i < rank; i++){
        wt(i,i) = 1 / variances(subCol(i)); // cleanup code and make it more vectorized later.
    }
    cout << "new wt mat:" << endl << wt << endl;

    return wt;
}

void printToCsv(const MatrixXd& mat, const string& fileName){ // prints matrix to csv
    ofstream plot;
    string csvFile = fileName + ".csv";
	plot.open(csvFile);

    for(int i = 0; i < mat.rows(); i++){
        for(int j = 0; j < mat.cols(); j++){
            if(j == 0){
                plot << mat(i,j);
            }else{
                plot << "," << mat(i,j);
            }
        }
        plot << endl;
    }
    plot.close();
}

int main() {
    auto t1 = std::chrono::high_resolution_clock::now();
    /*---------------------- Setup ------------------------ */
  
    /* Variables (global) */
    double t0 = 0, tf = 15, dt = 1.0; 
    int nTimeSteps = 5;
    VectorXd times = VectorXd::Zero(nTimeSteps);
    times << 0.5, 2, 10, 20, 30; // ultra early, early, medium, late
    int Npars = N_DIM;
    double squeeze = 0.500, sdbeta = 0.10; 
    double boundary = 0.001;
    /* SETUP */
    int useDiag = 0;
    int sf1 = 1;
    int sf2 = 1;
    double epsi = 0.02;
    double nan = 0.005;
    /* PSO params */
    double sfp = 3.0, sfg = 1.0, sfe = 6.0; // initial particle historical weight, global weight social, inertial
    double sfi = sfe, sfc = sfp, sfs = sfg; // below are the variables being used to reiterate weights
    double alpha = 0.2;
    int N = 25000;
    int nParts = 25; // first part PSO
    int nSteps = 50;
    int nParts2 = 10; // second part PSO
    int nSteps2 = 1000;
    int nMoments = (N_SPECIES * (N_SPECIES + 3)) / 2; // var + mean + cov
    nMoments = 2 * N_SPECIES; // mean + var
    int hone = 24;
    //nMoments = 2*N_SPECIES; // mean + var only!
    VectorXd wmatup(4);
    wmatup << 0.15, 0.35, 0.60, 0.9;
    double uniLowBound = 0.0, uniHiBound = 1.0;
    random_device RanDev;
    mt19937 gen(RanDev());
    uniform_real_distribution<double> unifDist(uniLowBound, uniHiBound);
    
    vector<MatrixXd> weights;
    for(int i = 0; i < nTimeSteps; i++){
        weights.push_back(MatrixXd::Identity(nMoments, nMoments));
    }
 
    cout << "Using two part PSO " << "Sample Size:" << N << " with:" << nMoments << " moments." << endl;
    cout << "Using Times:" << times.transpose() << endl;
    cout << "Bounds for Uniform Distribution (" << uniLowBound << "," << uniHiBound << ")"<< endl;
    cout << "Blind PSO --> nParts:" << nParts << " Nsteps:" << nSteps << endl;
    cout << "Targeted PSO --> nParts:" <<  nParts2 << " Nsteps:" << nSteps2 << endl;
    cout << "sdbeta:" << sdbeta << endl;
    // cout << "wt:" << endl << wt << endl;

    MatrixXd GBMAT(0, 0); // iterations of global best vectors
    MatrixXd PBMAT(nParts, Npars + 1); // particle best matrix + 1 for cost component
    MatrixXd POSMAT(nParts, Npars); // Position matrix as it goees through it in parallel

    cout << "Reading in data!" << endl;
    /* Initial Conditions */
    int sizeFile = 25000;
    int startRow = 0;
    MatrixXd X_0_Full(sizeFile, Npars);
    MatrixXd Y_0_Full(sizeFile, Npars);
    MatrixXd X_0(N, Npars);
    MatrixXd Y_0(N, Npars);
    ifstream X0File("noo25-initial-x.txt");
    ifstream Y0File("noo25-initial-y.txt");
    
    X_0_Full = readIntoMatrix(X0File, sizeFile, N_SPECIES);
    Y_0_Full = readIntoMatrix(Y0File, sizeFile, N_SPECIES);
    X0File.close();
    Y0File.close();
    
    X_0 = X_0_Full.block(startRow, 0, N, Npars);
    Y_0 = Y_0_Full.block(startRow, 0, N, Npars);
    cout << "Using starting row of data:" << startRow << " and " << N << " data pts!" << endl;
    cout << "first row X0:" << X_0.row(0) << endl;
    cout << "final row X0:" << X_0.row(N - 1) << endl << endl << endl << endl;
    Controlled_RK_Stepper_N controlledStepper;
    struct K tru;
    tru.k << 0.1, 0.1, 0.95, 0.17, 0.05, 0.18;
    // tru.k << 5.0, 0.1, 1.0, 8.69, 0.05, 0.70;
    // tru.k /= (9.69);
    // tru.k(1) += 0.05;
    // tru.k(4) += 0.05; // make sure not so close to the boundary
    // tru.k << 0.996673, 0.000434062, 0.0740192,  0.795578,  0.00882025, 0.0317506;
    // weights[0] << 0.0217808,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,1.72654e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,7.77961e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,2.46639e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0.000275497,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0.000124104,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,4.23237e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,1.45417e-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,2.78622e-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,7.36829e-11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,2.90089e-09,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,6.08141e-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,1.08416e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,1.53275e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.48166e-08,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000430593,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.0621e-07,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.62e-11,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.32576e-11,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7.26229e-11,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.30053e-11,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.4428e-10,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.35522e-08,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7.85887e-10,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.49694e-05,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.98797e-10,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5.34559e-09;



    // weights[1] << 4.00718e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,1.50966e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0.00012917,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,1.92359e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0.00818417,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,7.72713e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,3.32118e-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,4.47061e-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,1.50355e-09,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,2.70619e-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,6.58839e-08,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,2.71889e-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,1.54933e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,1.03153e-08,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.23214e-11,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00089642,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8.3704e-10,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.62573e-11,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.92663e-12,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.33673e-09,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.17936e-11,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.20387e-11,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5.58145e-07,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.2121e-09,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.06755e-05,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.06908e-11,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.9734e-08;



    // weights[2] << 4.62096e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,4.39497e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0.000300466,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,1.84112e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0.0172175,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,7.60611e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,3.28738e-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,1.99312e-12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,9.03591e-09,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,1.43766e-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,1.79282e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,2.62267e-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,4.48349e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,2.16934e-08,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0276e-11,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000955776,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8.15362e-10,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.13884e-10,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.22443e-12,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.56986e-09,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5.34944e-11,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7.84461e-11,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.07182e-06,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.54594e-09,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.05137e-05,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.02858e-11,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.19902e-08;

    // weights[3] << 6.28437e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,1.16465e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0.000835377,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,1.84304e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0.0197081,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,7.58762e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,3.51523e-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,5.76516e-12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,3.99174e-08,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,1.23267e-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,2.1636e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,2.60702e-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,1.23148e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,2.53417e-08,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.24527e-11,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00110174,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9.86703e-10,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.10732e-09,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7.98952e-12,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.89821e-09,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0856e-10,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.90218e-10,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.2049e-06,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.46546e-09,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.04636e-05,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.75297e-11,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.50426e-08;

    // weights[4] << 6.93804e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,2.35916e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0.00136269,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,1.83155e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0.0202082,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,7.58275e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,3.55859e-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,1.09182e-11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,9.0461e-08,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,1.16211e-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,2.23434e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,2.60314e-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,1.95564e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,2.65126e-08,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.32801e-11,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00114742,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.04614e-09,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.50655e-09,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.84821e-11,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.95995e-09,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.65181e-10,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.00912e-10,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.23018e-06,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5.66463e-09,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.04471e-05,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.66424e-11,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.56316e-08;
    cout << "using truk:" << tru.k.transpose() << endl;
    vector<VectorXd> Yt3Vecs;
    for(int t = 0; t < nTimeSteps; t++){
        Nonlinear_ODE6 trueSys(tru);
        Protein_Components Yt(times(t), nMoments, N);
        Moments_Mat_Obs YtObs(Yt);
        for (int i = 0; i < N; i++) {
            //State_N c0 = gen_multi_norm_iSub(); // Y_0 is simulated using norm dist.
            State_N c0 = convertInit(Y_0, i);
            Yt.index = i;
            integrate_adaptive(controlledStepper, trueSys, c0, t0, times(t), dt, YtObs);
        }
        Yt.mVec /= N;
        Yt3Vecs.push_back(Yt.mVec);
    }
    // struct K seed;
    // seed.k << 0.1659069,	0.6838229,	0.9585955,	0.4651133,	0.4573598,	0.1806655;


    // /* Solve for Cost of specified rates*/
    // double costSeedK = 0;
    // for(int t = 0; t < nTimeSteps; t++){
    //     Protein_Components Xt(times(t), nMoments, N);
    //     Moments_Mat_Obs XtObs(Xt);
    //     Nonlinear_ODE6 sys(seed);
    //     for (int i = 0; i < N; i++) {
    //         //State_N c0 = gen_multi_norm_iSub();
    //         State_N c0 = convertInit(X_0, i);
    //         Xt.index = i;
    //         integrate_adaptive(controlledStepper, sys, c0, t0, times(t), dt, XtObs);
    //     }
    //     Xt.mVec /= N;  
    //     cout << "XtmVec:" << Xt.mVec.transpose() << endl;
    //     costSeedK += calculate_cf2(Yt3Vecs[t], Xt.mVec, weights[t]);
    // }
    // cout << "For K:"<< seed.k.transpose() << "cost:" << costSeedK << endl;

    // seed.k << 0.2498351, 0.7010353,	0.9587309,	0.5199925,	0.4305584,	0.179396;
    // // change to the second moments only
    // for(int i = 0; i < nTimeSteps; i++){
    //     for(int j = 2*N_SPECIES; j < nMoments; j++){
    //         weights[i](j,j) = 0;
    //     }
    // }
    
    // costSeedK = 0;
    // for(int t = 0; t < nTimeSteps; t++){
    //     Protein_Components Xt(times(t), nMoments, N);
    //     Moments_Mat_Obs XtObs(Xt);
    //     Nonlinear_ODE6 sys(seed);
    //     for (int i = 0; i < N; i++) {
    //         //State_N c0 = gen_multi_norm_iSub();
    //         State_N c0 = convertInit(X_0, i);
    //         Xt.index = i;
    //         integrate_adaptive(controlledStepper, sys, c0, t0, times(t), dt, XtObs);
    //     }
    //     Xt.mVec /= N;  
    //     cout << "XtmVec:" << Xt.mVec.transpose() << endl;
    //     costSeedK += calculate_cf2(Yt3Vecs[t], Xt.mVec, weights[t]);
    // }
    // cout << "For K:"<< seed.k.transpose() << "cost:" << costSeedK << endl;
    /* Solve for 50 x 50 contour plot for equal weights */
    int xDim = 50, yDim = 50;
    double scale = (xDim+yDim) / 2;
    double cost = 0;
    MatrixXd eqwts(xDim*yDim, Npars + 1);
    int s = 0;
    for(int x = 0; x < xDim; x++){
        for(int y = 0; y < yDim; y++){
            K rate;
            rate.k = tru.k;
            rate.k(0) = x / scale;
            rate.k(4) = y / scale;
            for(int t = 0; t < nTimeSteps; t++){
                Nonlinear_ODE6 sys(rate);
                Protein_Components Xt(times(t), nMoments, N);
                Moments_Mat_Obs XtObs(Xt);
                for (int i = 0; i < N; i++) {
                    State_N x0 = convertInit(X_0, i);
                    Xt.index = i;
                    integrate_adaptive(controlledStepper, sys, x0, t0, times(t), dt, XtObs);
                }
                Xt.mVec /= N;
                cost += calculate_cf2(Yt3Vecs[t],Xt.mVec, weights[t]);
            }
            for (int i = 0; i < Npars; i++) {
                eqwts(s, i) = rate.k(i);
            }
            eqwts(s, Npars) = cost;
            s++;
            cost = 0;
        }
    }
    printToCsv(eqwts, "eqwts_contour");

    //  weights[0] <<1.09422,0.0010906,0.00278112,-0.0156409,-0.00170461,-0.000388485,0.00102581,8.25739e-10,5.40332e-09,-8.88803e-08,-6.35746e-10,-9.2078e-11,
    // 0.0010906,3.07445e-06,7.84013e-06,-4.40925e-05,-4.8054e-06,-1.09516e-06,2.89181e-06,2.32781e-12,1.52323e-11,-2.50559e-10,-1.7922e-12,-2.59573e-13,
    // 0.00278112,7.84013e-06,0.000265871,-0.00149525,-0.000162959,-3.71387e-05,9.80657e-05,7.89396e-11,5.1655e-10,-8.49684e-09,-6.07765e-11,-8.80254e-12,
    // -0.0156409,-4.40925e-05,-0.00149525,0.0119129,0.00129832,0.00029589,-0.000781306,-6.28925e-10,-4.11545e-09,6.76958e-08,4.84216e-10,7.01313e-11,
    // -0.00170461,-4.8054e-06,-0.000162959,0.00129832,0.000222773,5.07705e-05,-0.000134061,-1.07914e-10,-7.06151e-10,1.16156e-08,8.30845e-11,1.20335e-11,
    // -0.000388485,-1.09516e-06,-3.71387e-05,0.00029589,5.07705e-05,0.00079705,-0.00210463,-1.69416e-09,-1.10859e-08,1.82354e-07,1.30435e-09,1.88915e-10,
    // 0.00102581,2.89181e-06,9.80657e-05,-0.000781306,-0.000134061,-0.00210463,0.02343,1.88604e-08,1.23415e-07,-2.03008e-06,-1.45208e-08,-2.10312e-09,
    // 8.25739e-10,2.32781e-12,7.89396e-11,-6.28925e-10,-1.07914e-10,-1.69416e-09,1.88604e-08,1.36098e-13,8.90571e-13,-1.46492e-11,-1.04783e-13,-1.51762e-14,
    // 5.40332e-09,1.52323e-11,5.1655e-10,-4.11545e-09,-7.06151e-10,-1.10859e-08,1.23415e-07,8.90571e-13,4.0512e-10,-6.6639e-09,-4.76658e-11,-6.90366e-12,
    // -8.88803e-08,-2.50559e-10,-8.49684e-09,6.76958e-08,1.16156e-08,1.82354e-07,-2.03008e-06,-1.46492e-11,-6.6639e-09,3.27852e-07,2.34507e-09,3.39647e-10,
    // -6.35746e-10,-1.7922e-12,-6.07765e-11,4.84216e-10,8.30845e-11,1.30435e-09,-1.45208e-08,-1.04783e-13,-4.76658e-11,2.34507e-09,5.24271e-10,7.59326e-11,
    // -9.2078e-11,-2.59573e-13,-8.80254e-12,7.01313e-11,1.20335e-11,1.88915e-10,-2.10312e-09,-1.51762e-14,-6.90366e-12,3.39647e-10,7.59326e-11,1.28787e-08;


    // weights[1] << 0.10697,0.00286477,0.0560647,-0.0317621,-0.0445589,0.00495941,-2.00699e-06,-6.16071e-11,-2.80581e-09,1.21374e-09,6.88116e-10,-2.06854e-11,
    // 0.00286477,9.38468e-05,0.00183662,-0.00104049,-0.0014597,0.000162465,-6.57469e-08,-2.01818e-12,-9.19151e-11,3.97608e-11,2.25419e-11,-6.77631e-13,
    // 0.0560647,0.00183662,0.0366777,-0.0207789,-0.0291506,0.00324446,-1.31298e-06,-4.03036e-11,-1.83557e-09,7.94033e-10,4.50168e-10,-1.35325e-11,
    // -0.0317621,-0.00104049,-0.0207789,0.0118,0.0165541,-0.00184247,7.45619e-07,2.28877e-11,1.04239e-09,-4.50917e-10,-2.55642e-10,7.68483e-12,
    // -0.0445589,-0.0014597,-0.0291506,0.0165541,0.0235194,-0.00261771,1.05935e-06,3.25179e-11,1.48098e-09,-6.40645e-10,-3.63206e-10,1.09183e-11,
    // 0.00495941,0.000162465,0.00324446,-0.00184247,-0.00261771,0.000413975,-1.67529e-07,-5.14251e-12,-2.34208e-10,1.01314e-10,5.74389e-11,-1.72667e-12,
    // -2.00699e-06,-6.57469e-08,-1.31298e-06,7.45619e-07,1.05935e-06,-1.67529e-07,3.92714e-07,1.20548e-11,5.4902e-10,-2.37496e-10,-1.34646e-10,4.04757e-12,
    // -6.16071e-11,-2.01818e-12,-4.03036e-11,2.28877e-11,3.25179e-11,-5.14251e-12,1.20548e-11,1.92102e-13,8.74903e-12,-3.78467e-12,-2.14567e-12,6.45009e-14,
    // -2.80581e-09,-9.19151e-11,-1.83557e-09,1.04239e-09,1.48098e-09,-2.34208e-10,5.4902e-10,8.74903e-12,1.68792e-09,-7.30165e-10,-4.13958e-10,1.2444e-11,
    // 1.21374e-09,3.97608e-11,7.94033e-10,-4.50917e-10,-6.40645e-10,1.01314e-10,-2.37496e-10,-3.78467e-12,-7.30165e-10,3.83666e-10,2.17515e-10,-6.5387e-12,
    // 6.88116e-10,2.25419e-11,4.50168e-10,-2.55642e-10,-3.63206e-10,5.74389e-11,-1.34646e-10,-2.14567e-12,-4.13958e-10,2.17515e-10,3.13604e-09,-9.42722e-11,
    // -2.06854e-11,-6.77631e-13,-1.35325e-11,7.68483e-12,1.09183e-11,-1.72667e-12,4.04757e-12,6.45009e-14,1.2444e-11,-6.5387e-12,-9.42722e-11,5.99071e-10;



    // weights[2] <<6.07665e-05,7.56214e-06,-1.82441e-05,1.96504e-07,1.68261e-06,-2.31252e-08,-2.39347e-12,-2.39666e-14,4.55913e-14,-1.10389e-16,-9.32386e-16,6.76446e-18,
    // 7.56214e-06,2.62038e-06,-6.32182e-06,6.80914e-08,5.83046e-07,-8.01321e-09,-8.29369e-13,-8.30476e-15,1.5798e-14,-3.82512e-17,-3.23085e-16,2.34398e-18,
    // -1.82441e-05,-6.32182e-06,0.000145739,-1.56973e-06,-1.34411e-05,1.84731e-07,1.91197e-11,1.91452e-13,-3.64196e-13,8.81815e-16,7.44816e-15,-5.40365e-17,
    // 1.96504e-07,6.80914e-08,-1.56973e-06,1.98461e-06,1.69936e-05,-2.33555e-07,-2.4173e-11,-2.42052e-13,4.60452e-13,-1.11488e-15,-9.4167e-15,6.83182e-17,
    // 1.68261e-06,5.83046e-07,-1.34411e-05,1.69936e-05,0.00961031,-0.000132081,-1.36704e-08,-1.36887e-10,2.60397e-10,-6.30491e-13,-5.32538e-12,3.86357e-14,
    // -2.31252e-08,-8.01321e-09,1.84731e-07,-2.33555e-07,-0.000132081,7.89916e-05,8.17565e-09,8.18656e-11,-1.55732e-10,3.77068e-13,3.18486e-12,-2.31062e-14,
    // -2.39347e-12,-8.29369e-13,1.91197e-11,-2.4173e-11,-1.36704e-08,8.17565e-09,3.44482e-10,3.44942e-12,-6.56177e-12,1.58878e-14,1.34195e-13,-9.73582e-16,
    // -2.39666e-14,-8.30476e-15,1.91452e-13,-2.42052e-13,-1.36887e-10,8.18656e-11,3.44942e-12,5.10859e-13,-9.71799e-13,2.35298e-15,1.98742e-14,-1.44188e-16,
    // 4.55913e-14,1.5798e-14,-3.64196e-13,4.60452e-13,2.60397e-10,-1.55732e-10,-6.56177e-12,-9.71799e-13,1.66721e-09,-4.03675e-12,-3.4096e-11,2.47367e-13,
    // -1.10389e-16,-3.82512e-17,8.81815e-16,-1.11488e-15,-6.30491e-13,3.77068e-13,1.58878e-14,2.35298e-15,-4.03675e-12,2.72583e-13,2.30234e-12,-1.67035e-14,
    // -9.32386e-16,-3.23085e-16,7.44816e-15,-9.4167e-15,-5.32538e-12,3.18486e-12,1.34195e-13,1.98742e-14,-3.4096e-11,2.30234e-12,7.00819e-08,-5.08445e-10,
    // 6.76446e-18,2.34398e-18,-5.40365e-17,6.83182e-17,3.86357e-14,-2.31062e-14,-9.73582e-16,-1.44188e-16,2.47367e-13,-1.67035e-14,-5.08445e-10,2.73523e-10;



    // weights[3] <<8.14592e-05,2.0546e-05,-9.3966e-05,-7.05591e-07,-8.61796e-06,7.61065e-08,1.34891e-11,2.62126e-13,-5.36688e-12,-1.11234e-16,-3.04599e-15,1.63608e-17,
    // 2.0546e-05,1.23055e-05,-5.62784e-05,-4.22594e-07,-5.16149e-06,4.55819e-08,8.07894e-12,1.56993e-13,-3.21434e-12,-6.66206e-17,-1.82431e-15,9.79884e-18,
    // -9.3966e-05,-5.62784e-05,0.000582398,4.37322e-06,5.34138e-05,-4.71705e-07,-8.3605e-11,-1.62465e-12,3.32637e-11,6.89425e-16,1.88789e-14,-1.01403e-16,
    // -7.05591e-07,-4.22594e-07,4.37322e-06,1.93245e-06,2.36026e-05,-2.08438e-07,-3.69436e-11,-7.17902e-13,1.46986e-11,3.04645e-16,8.34226e-15,-4.48084e-17,
    // -8.61796e-06,-5.16149e-06,5.34138e-05,2.36026e-05,0.0209139,-0.000184694,-3.27352e-08,-6.36123e-10,1.30243e-08,2.69941e-13,7.39196e-12,-3.97041e-14,
    // 7.61065e-08,4.55819e-08,-4.71705e-07,-2.08438e-07,-0.000184694,7.80818e-05,1.38392e-08,2.68929e-10,-5.50617e-09,-1.14121e-13,-3.12505e-12,1.67854e-14,
    // 1.34891e-11,8.07894e-12,-8.3605e-11,-3.69436e-11,-3.27352e-08,1.38392e-08,3.44154e-10,6.68773e-12,-1.36927e-10,-2.83796e-15,-7.77136e-14,4.1742e-16,
    // 2.62126e-13,1.56993e-13,-1.62465e-12,-7.17902e-13,-6.36123e-10,2.68929e-10,6.68773e-12,2.64192e-12,-5.40918e-11,-1.12111e-15,-3.07e-14,1.64897e-16,
    // -5.36688e-12,-3.21434e-12,3.32637e-11,1.46986e-11,1.30243e-08,-5.50617e-09,-1.36927e-10,-5.40918e-11,1.14225e-08,2.36743e-13,6.48286e-12,-3.48211e-14,
    // -1.11234e-16,-6.66206e-17,6.89425e-16,3.04645e-16,2.69941e-13,-1.14121e-13,-2.83796e-15,-1.12111e-15,2.36743e-13,1.42858e-13,3.91195e-12,-2.10121e-14,
    // -3.04599e-15,-1.82431e-15,1.88789e-14,8.34226e-15,7.39196e-12,-3.12505e-12,-7.77136e-14,-3.07e-14,6.48286e-12,3.91195e-12,1.92329e-07,-1.03305e-09,
    // 1.63608e-17,9.79884e-18,-1.01403e-16,-4.48084e-17,-3.97041e-14,1.67854e-14,4.1742e-16,1.64897e-16,-3.48211e-14,-2.10121e-14,-1.03305e-09,2.66698e-10;



    // weights[4] <<8.09574e-05,2.01789e-05,-8.61975e-05,-3.58452e-07,-4.73831e-06,3.77063e-08,6.53837e-12,1.36791e-13,-3.78602e-12,-6.5875e-17,-2.61153e-15,1.30799e-17,
    // 2.01789e-05,2.37142e-05,-0.000101299,-4.21254e-07,-5.56847e-06,4.43125e-08,7.6839e-12,1.60757e-13,-4.44934e-12,-7.74164e-17,-3.06908e-15,1.53715e-17,
    // -8.61975e-05,-0.000101299,0.00136819,5.68963e-06,7.52101e-05,-5.98503e-07,-1.03782e-10,-2.17125e-12,6.00947e-11,1.04562e-15,4.14523e-14,-2.07614e-16,
    // -3.58452e-07,-4.21254e-07,5.68963e-06,1.91036e-06,2.52526e-05,-2.00954e-07,-3.4846e-11,-7.29022e-13,2.01774e-11,3.51078e-16,1.39181e-14,-6.97087e-17,
    // -4.73831e-06,-5.56847e-06,7.52101e-05,2.52526e-05,0.024241,-0.000192904,-3.34501e-08,-6.99818e-10,1.93691e-08,3.37014e-13,1.33605e-11,-6.69162e-14,
    // 3.77063e-08,4.43125e-08,-5.98503e-07,-2.00954e-07,-0.000192904,7.77761e-05,1.34866e-08,2.82156e-10,-7.80935e-09,-1.35879e-13,-5.38676e-12,2.69796e-14,
    // 6.53837e-12,7.6839e-12,-1.03782e-10,-3.4846e-11,-3.34501e-08,1.34866e-08,3.55822e-10,7.44426e-12,-2.06038e-10,-3.58496e-15,-1.42121e-13,7.11815e-16,
    // 1.36791e-13,1.60757e-13,-2.17125e-12,-7.29022e-13,-6.99818e-10,2.82156e-10,7.44426e-12,7.92824e-12,-2.19433e-10,-3.81803e-15,-1.51361e-13,7.58094e-16,
    // -3.78602e-12,-4.44934e-12,6.00947e-11,2.01774e-11,1.93691e-08,-7.80935e-09,-2.06038e-10,-2.19433e-10,5.41029e-08,9.41365e-13,3.73193e-11,-1.86914e-13,
    // -6.5875e-17,-7.74164e-17,1.04562e-15,3.51078e-16,3.37014e-13,-1.35879e-13,-3.58496e-15,-3.81803e-15,9.41365e-13,1.22545e-13,4.85816e-12,-2.43321e-14,
    // -2.61153e-15,-3.06908e-15,4.14523e-14,1.39181e-14,1.33605e-11,-5.38676e-12,-1.42121e-13,-1.51361e-13,3.73193e-11,4.85816e-12,2.32447e-07,-1.16421e-09,
    // 1.30799e-17,1.53715e-17,-2.07614e-16,-6.97087e-17,-6.69162e-14,2.69796e-14,7.11815e-16,7.58094e-16,-1.86914e-13,-2.43321e-14,-1.16421e-09,2.65506e-10;


    // s = 0;
    // cost = 0;
    // MatrixXd uneqwts = MatrixXd::Zero(xDim*yDim, Npars + 1);
    // for(int x = 0; x < xDim; x++){
    //     for(int y = 0; y < yDim; y++){
    //         K rate;
    //         rate.k = tru.k;
    //         rate.k(1) = x / scale;
    //         rate.k(4) = y / scale;
    //         for(int t = 0; t < nTimeSteps; t++){
    //             Nonlinear_ODE6 sys(rate);
    //             Protein_Components Xt(times(t), nMoments, N);
    //             Moments_Mat_Obs XtObs(Xt);
    //             for (int i = 0; i < N; i++) {
    //                 //State_N c0 = gen_multi_norm_iSub(); // Y_0 is simulated using norm dist.
    //                 State_N x0 = convertInit(X_0, i);
    //                 Xt.index = i;
    //                 integrate_adaptive(controlledStepper, sys, x0, t0, times(t), dt, XtObs);
    //             }
    //             Xt.mVec /= N;
    //             cost += calculate_cf2(Yt3Vecs[t],Xt.mVec, weights[t]);
    //         }
    //         for (int i = 0; i < Npars; i++) {
    //             uneqwts(s, i) = rate.k(i);
    //         }
    //         uneqwts(s, Npars) = cost;
    //         s++;   
    //         cost = 0;
    //     }
    // }
    // printToCsv(uneqwts, "uneqwts_contour");
    

    // MatrixXd zoomedIn = MatrixXd::Zero(xDim*yDim, Npars + 1);
    // VectorXd xCoords = VectorXd::Zero(xDim);
    // VectorXd yCoords = VectorXd::Zero(yDim);
    // xCoords << 0.45600,0.45616,0.45632,0.45648,0.45664,0.45680,0.45696,0.45712,0.45728,0.45744,0.45760,0.45776,0.45792,0.45808,0.45824,0.45840,0.45856,0.45872,0.45888,0.45904,0.45920,0.45936,0.45952,0.45968,0.45984,0.46000,0.46016,0.46032,0.46048,0.46064,0.46080,0.46096,0.46112,0.46128,0.46144,
    // 0.46160,0.46176,0.46192,0.46208,0.46224,0.46240,0.46256,0.46272,0.46288,0.46304,0.46320,0.46336,0.46352,0.46368,0.46384;
    // yCoords << 0.03800,0.03816,0.03832,0.03848,0.03864,0.03880,0.03896,0.03912,0.03928,0.03944,0.03960,0.03976,0.03992,0.04008,0.04024,0.04040,0.04056,0.04072,0.04088,0.04104,0.04120,
    // 0.04136, 0.04152, 0.04168, 0.04184, 0.04200, 0.04216,0.04232,0.04248,0.04264,0.04280,0.04296,0.04312,0.04328,0.04344,0.04360,0.04376,0.04392,0.04408,0.04424,0.04440,0.04456,0.04472,0.04488,0.04504,0.04520,0.04536,0.04552,0.04568,0.04584;
    // s = 0;
    // cost = 0;
    // for(int x = 0; x < xDim; x++){
    //     for(int y = 0; y < yDim; y++){
    //         K rate;
    //         rate.k = tru.k;
    //         rate.k(0) = xCoords(x);
    //         rate.k(1) = yCoords(y);
    //         for(int t = 0; t < nTimeSteps; t++){
    //             Nonlinear_ODE6 trueSys(tru);
    //             Nonlinear_ODE6 sys(rate);
    //             Protein_Components Yt(times(t), nMoments, N);
    //             Protein_Components Xt(times(t), nMoments, N);
    //             Moments_Mat_Obs YtObs(Yt);
    //             Moments_Mat_Obs XtObs(Xt);
    //             for (int i = 0; i < N; i++) {
    //                 //State_N c0 = gen_multi_norm_iSub(); // Y_0 is simulated using norm dist.
    //                 State_N c0 = convertInit(Y_0, i);
    //                 State_N x0 = convertInit(X_0, i);
    //                 Yt.index = i;
    //                 Xt.index = i;
    //                 integrate_adaptive(controlledStepper, trueSys, c0, t0, times(t), dt, YtObs);
    //                 integrate_adaptive(controlledStepper, sys, x0, t0, times(t), dt, XtObs);
    //             }
    //             Yt.mVec /= N;
    //             Xt.mVec /= N;
    //             cost += calculate_cf2(Yt.mVec,Xt.mVec, weights[t]);
    //         }
    //         for (int i = 0; i < Npars; i++) {
    //             zoomedIn(s, i) = rate.k(i);
    //         }
    //         zoomedIn(s, Npars) = cost;
    //         s++;   
    //         cost = 0;
    //     }
    // }
    // printToCsv(zoomedIn, "uneqwts_zoomIn_contour");


    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << "CODE FINISHED RUNNING IN " << duration << " s TIME!" << endl;

    return 0; // just to close the program at the end.
}

