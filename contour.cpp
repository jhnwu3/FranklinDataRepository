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
    times << 2, 10, 20, 30, 40; // ultra early, early, medium, late
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
    int N = 1000;
    int nParts = 25; // first part PSO
    int nSteps = 50;
    int nParts2 = 10; // second part PSO
    int nSteps2 = 1000;
    int nMoments = (N_SPECIES * (N_SPECIES + 3)) / 2; // var + mean + cov
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
    // ifstream weightForSingleTime("time1_wt.txt");
    // weights[0] = readIntoMatrix(weightForSingleTime, nMoments, nMoments);

    // ifstream weight0("time5_wt0.txt");
    // ifstream weight1("time5_wt1.txt");
    // ifstream weight2("time5_wt2.txt");
    // ifstream weight3("time5_wt3.txt");
    // ifstream weight4("time5_wt4.txt");

    // weights[0] = readIntoMatrix(weight0, nMoments, nMoments);
    // weights[1] = readIntoMatrix(weight1, nMoments, nMoments);
    // weights[2] = readIntoMatrix(weight2, nMoments, nMoments);
    // weights[3] = readIntoMatrix(weight3, nMoments, nMoments);
    // weights[4] = readIntoMatrix(weight4, nMoments, nMoments);
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

    struct K tru;
    tru.k << 0.1, 0.1, 0.95, 0.17, 0.05, 0.18;
    // tru.k << 5.0, 0.1, 1.0, 8.69, 0.05, 0.70;
    // tru.k /= (9.69);
    // tru.k(1) += 0.05;
    // tru.k(4) += 0.05; // make sure not so close to the boundary
    // tru.k << 0.996673, 0.000434062, 0.0740192,  0.795578,  0.00882025, 0.0317506;
    cout << "using truk:" << tru.k.transpose() << endl;
    /* Solve for 50 x 50 contour plot for equal weights */
    int xDim = 50, yDim = 50;
    double scale = (xDim+yDim) / 2;
    double cost = 0;
    MatrixXd eqwts(xDim*yDim, Npars + 1);
    Controlled_RK_Stepper_N controlledStepper;
    int s = 0;
    for(int x = 0; x < xDim; x++){
        for(int y = 0; y < yDim; y++){
            K rate;
            rate.k = tru.k;
            rate.k(0) = x / scale;
            rate.k(1) = y / scale;
            for(int t = 0; t < nTimeSteps; t++){
                Nonlinear_ODE6 trueSys(tru);
                Nonlinear_ODE6 sys(rate);
                Protein_Components Yt(times(t), nMoments, N);
                Protein_Components Xt(times(t), nMoments, N);
                Moments_Mat_Obs YtObs(Yt);
                Moments_Mat_Obs XtObs(Xt);
                for (int i = 0; i < N; i++) {
                    //State_N c0 = gen_multi_norm_iSub(); // Y_0 is simulated using norm dist.
                    State_N c0 = convertInit(Y_0, i);
                    State_N x0 = convertInit(X_0, i);
                    Yt.index = i;
                    Xt.index = i;
                    integrate_adaptive(controlledStepper, trueSys, c0, t0, times(t), dt, YtObs);
                    integrate_adaptive(controlledStepper, sys, x0, t0, times(t), dt, XtObs);
                }
                Yt.mVec /= N;
                Xt.mVec /= N;
                cost += calculate_cf2(Yt.mVec,Xt.mVec, weights[t]);
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

    // weights[0] << 0.0137256,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0.00478114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0.0790092,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0.177227,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0.00363366,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0.241506,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,7.50555e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,5.20021e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0.000361337,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0.00076074,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,4.08747e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0.00431104,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,1.04311,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0.000536949,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000745204,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8.22858e-05,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00125977,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000138994,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000601148,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.91117e-05,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000530679,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000669311,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000352234,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0020026,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.47299,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0148698,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000549435;

    // weights[1] << 0.0122161,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0.00564585,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0.155478,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0.0304871,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0.00369779,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0.132155,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,6.44115e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,5.98868e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0.00172804,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,2.79732e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,4.19594e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0.00124461,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,1.73831,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0.00116672,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000110125,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7.87869e-05,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000737975,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000240962,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000119193,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.77826e-05,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000292525,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000460489,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000528447,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00189959,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.20546,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00215693,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000298264;

    // weights[2] << 0.0120074,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0.00617075,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0.193294,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0.0188695,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0.0037058,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0.121902,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,6.29842e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,6.55006e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0.00261131,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,1.2086e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,4.2111e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0.00105337,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,2.01907,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0.00138549,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,7.18318e-05,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7.8666e-05,0,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000707229,0,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000320122,0,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7.28313e-05,0,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.0021e-05,0,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000276095,0,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000494172,0,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000551135,0,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00192019,0,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.04339,0,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0012772,0,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000278127;

    weights[0] << 0.0137708,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0.00479694,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0.0807319,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0.178045,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0.00363427,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0.244524,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,7.49452e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,5.20386e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0.000375805,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0.000760057,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,4.08501e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0.00442285,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;


    weights[1] << 0.0122451,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0.00566424,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0.158385,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0.0306952,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0.00369683,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0.133857,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,6.43374e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,6.00122e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0.00179044,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,2.81343e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,4.19143e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0.00128113,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;


    weights[2] << 0.0120313,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0.00619302,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0.197461,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0.0189385,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0.0037044,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0.123447,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,6.29161e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,6.56966e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0.00274582,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,1.20606e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,4.20638e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0.00108306,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;

    s = 0;
    cost = 0;
    MatrixXd uneqwts = MatrixXd::Zero(xDim*yDim, Npars + 1);
    for(int x = 0; x < xDim; x++){
        for(int y = 0; y < yDim; y++){
            K rate;
            rate.k = tru.k;
            rate.k(0) = x / scale;
            rate.k(1) = y / scale;
            for(int t = 0; t < nTimeSteps; t++){
                Nonlinear_ODE6 trueSys(tru);
                Nonlinear_ODE6 sys(rate);
                Protein_Components Yt(times(t), nMoments, N);
                Protein_Components Xt(times(t), nMoments, N);
                Moments_Mat_Obs YtObs(Yt);
                Moments_Mat_Obs XtObs(Xt);
                for (int i = 0; i < N; i++) {
                    //State_N c0 = gen_multi_norm_iSub(); // Y_0 is simulated using norm dist.
                    State_N c0 = convertInit(Y_0, i);
                    State_N x0 = convertInit(X_0, i);
                    Yt.index = i;
                    Xt.index = i;
                    integrate_adaptive(controlledStepper, trueSys, c0, t0, times(t), dt, YtObs);
                    integrate_adaptive(controlledStepper, sys, x0, t0, times(t), dt, XtObs);
                }
                Yt.mVec /= N;
                Xt.mVec /= N;
                cost += calculate_cf2(Yt.mVec,Xt.mVec, weights[t]);
            }
            for (int i = 0; i < Npars; i++) {
                uneqwts(s, i) = rate.k(i);
            }
            uneqwts(s, Npars) = cost;
            s++;   
            cost = 0;
        }
    }
    printToCsv(uneqwts, "uneqwts_contour");
    
    MatrixXd zoomedIn = MatrixXd::Zero(xDim*yDim, Npars + 1);
    VectorXd xCoords = VectorXd::Zero(xDim);
    VectorXd yCoords = VectorXd::Zero(yDim);
    xCoords << 0.45600,0.45616,0.45632,0.45648,0.45664,0.45680,0.45696,0.45712,0.45728,0.45744,0.45760,0.45776,0.45792,0.45808,0.45824,0.45840,0.45856,0.45872,0.45888,0.45904,0.45920,0.45936,0.45952,0.45968,0.45984,0.46000,0.46016,0.46032,0.46048,0.46064,0.46080,0.46096,0.46112,0.46128,0.46144,
    0.46160,0.46176,0.46192,0.46208,0.46224,0.46240,0.46256,0.46272,0.46288,0.46304,0.46320,0.46336,0.46352,0.46368,0.46384;
    yCoords << 0.03800,0.03816,0.03832,0.03848,0.03864,0.03880,0.03896,0.03912,0.03928,0.03944,0.03960,0.03976,0.03992,0.04008,0.04024,0.04040,0.04056,0.04072,0.04088,0.04104,0.04120,
    0.04136, 0.04152, 0.04168, 0.04184, 0.04200, 0.04216,0.04232,0.04248,0.04264,0.04280,0.04296,0.04312,0.04328,0.04344,0.04360,0.04376,0.04392,0.04408,0.04424,0.04440,0.04456,0.04472,0.04488,0.04504,0.04520,0.04536,0.04552,0.04568,0.04584;
    s = 0;
    cost = 0;
    for(int x = 0; x < xDim; x++){
        for(int y = 0; y < yDim; y++){
            K rate;
            rate.k = tru.k;
            rate.k(0) = xCoords(x);
            rate.k(1) = yCoords(y);
            for(int t = 0; t < nTimeSteps; t++){
                Nonlinear_ODE6 trueSys(tru);
                Nonlinear_ODE6 sys(rate);
                Protein_Components Yt(times(t), nMoments, N);
                Protein_Components Xt(times(t), nMoments, N);
                Moments_Mat_Obs YtObs(Yt);
                Moments_Mat_Obs XtObs(Xt);
                for (int i = 0; i < N; i++) {
                    //State_N c0 = gen_multi_norm_iSub(); // Y_0 is simulated using norm dist.
                    State_N c0 = convertInit(Y_0, i);
                    State_N x0 = convertInit(X_0, i);
                    Yt.index = i;
                    Xt.index = i;
                    integrate_adaptive(controlledStepper, trueSys, c0, t0, times(t), dt, YtObs);
                    integrate_adaptive(controlledStepper, sys, x0, t0, times(t), dt, XtObs);
                }
                Yt.mVec /= N;
                Xt.mVec /= N;
                cost += calculate_cf2(Yt.mVec,Xt.mVec, weights[t]);
            }
            for (int i = 0; i < Npars; i++) {
                zoomedIn(s, i) = rate.k(i);
            }
            zoomedIn(s, Npars) = cost;
            s++;   
            cost = 0;
        }
    }
    printToCsv(zoomedIn, "uneqwts_zoomIn_contour");


    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << "CODE FINISHED RUNNING IN " << duration << " s TIME!" << endl;

    return 0; // just to close the program at the end.
}

