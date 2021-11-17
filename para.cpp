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
                        dComp.mVec(N_SPECIES + i) += c[i] * c[j]; // 2nd moments
                    }
                    else if (upperDiag < dComp.mVec.size()){
                        dComp.mVec(upperDiag) += c[i] * c[j]; // cross moments
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
    State_N c0 = {sample(index,0), sample(index,1), sample(index,2), sample(index,3), sample(index,4), sample(index,5)};
    return c0;
}
VectorXd comp_vel_vec(const VectorXd& posK, int seed, double epsi, double nan, int hone) {
    
    // VectorXd rPoint;
    // rPoint = posK;
    // std::random_device rand_dev;
    // std::mt19937 generator(rand_dev());
    // vector<int> rand;
    // uniform_real_distribution<double> unifDist(0.0, 1.0);
    // // for (int i = 0; i < N_DIM; i++) {
    // //     rand.push_back(i);
    // // }
    // // shuffle(rand.begin(), rand.end(), generator); // shuffle indices as well as possible. 
    // // int ncomp = rand.at(0);
    // // VectorXd wcomp(ncomp);
    // // shuffle(rand.begin(), rand.end(), generator);
    // // for (int i = 0; i < ncomp; i++) {
    // //     wcomp(i) = rand.at(i);
    // // }
    // VectorXd adaptive = VectorXd::Zero(3); // vector of targeted rate constants
    // adaptive << 1,3,4;
    // int ncomp = posK.size();
    // if(unifDist(generator) < 0.67){
    //     for (int smart = 0; smart < adaptive.size(); smart++) {
    //     // int px = wcomp(smart);
    //         int px = adaptive(smart);
    //         double pos = rPoint(px);
    //         if (pos > 1.0 - nan) {
    //             cout << "overflow!" << endl;
    //             // while(pos > 1.0){
    //             //     pos -= 0.001;
    //             // }
    //             pos -= epsi;
    //         }else if (pos < nan) {
    //             cout << "underflow!"<< pos << endl;
    //             // while( pos < 0.001){
    //             //     pos += 0.001;
    //             // }
    //             pos += epsi;
    //             cout << "pos" << posK.transpose() << endl; 
    //         }
    //         double alpha = hone * pos; // Component specific
    //         double beta = hone - alpha; // pos specific
    //     // cout << "alpha:" << alpha << "beta:" << beta << endl;
    //         std::gamma_distribution<double> aDist(alpha, 1); // beta distribution consisting of gamma distributions
    //         std::gamma_distribution<double> bDist(beta, 1);

    //         double x = aDist(generator);
    //         double y = bDist(generator);

    //         rPoint(px) = (x / (x + y)); 
    //     }
    // }else{
    //     for (int smart = 0; smart < ncomp; smart++) {
    //     // int px = wcomp(smart);
    //         double pos = rPoint(smart);
    //         if (pos > 1.0 - nan) {
    //             cout << "overflow!" << endl;
    //             // while(pos > 1.0){
    //             //     pos -= 0.001;
    //             // }
    //             pos -= epsi;
    //         }else if (pos < nan) {
    //             cout << "underflow!"<< pos << endl;
    //             // while( pos < 0.001){
    //             //     pos += 0.001;
    //             // }
    //             pos += epsi;
    //             cout << "pos" << posK.transpose() << endl; 
    //         }
    //         double alpha = hone * pos; // Component specific
    //         double beta = hone - alpha; // pos specific
    //     // cout << "alpha:" << alpha << "beta:" << beta << endl;
    //         std::gamma_distribution<double> aDist(alpha, 1); // beta distribution consisting of gamma distributions
    //         std::gamma_distribution<double> bDist(beta, 1);

    //         double x = aDist(generator);
    //         double y = bDist(generator);

    //         rPoint(smart) = (x / (x + y)); 
    //     }
    // }
    
    // return rPoint;
    VectorXd rPoint;
    rPoint = posK;
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    vector<int> rand;
    uniform_real_distribution<double> unifDist(0.0, 1.0);
    for (int i = 0; i < posK.size(); i++) {
        rand.push_back(i);
    }
    shuffle(rand.begin(), rand.end(), generator); // shuffle indices as well as possible. 
    int ncomp = rand.at(0);
    VectorXd wcomp(ncomp);
    shuffle(rand.begin(), rand.end(), generator);
    for (int i = 0; i < ncomp; i++) {
        wcomp(i) = rand.at(i);
    }
    
    for (int smart = 0; smart < ncomp; smart++) {
        int px = wcomp(smart);
        double pos = rPoint(px);
        if (pos > 1.0 - nan) {
            cout << "overflow!" << endl;
            pos -= epsi;
        }else if (pos < nan) {
            cout << "underflow!"<< pos << endl;
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

        rPoint(px) = (x / (x + y)); 
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
MatrixXd customWtMat(const MatrixXd& Yt, const MatrixXd& Xt, int nMoments, int N){
    /* first moment differences */
    MatrixXd fmdiffs = Yt - Xt; 
    /* second moment difference computations - @todo make it variable later */
    MatrixXd smdiffs(N, N_SPECIES);
    for(int i = 0; i < N_SPECIES; i++){
        smdiffs.col(i) = (Yt.col(i).array() * Yt.col(i).array()) - (Xt.col(i).array() * Xt.col(i).array());
    }

    /* If no cross moments, then have a check for it */
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
    VectorXd covariances(nMoments - 1);
    
    for(int i = 0; i < nMoments - 1; i++){
        int j = i + 1;
        covariances(i) = ( (aDiff.col(i).array() - aDiff.col(i).array().mean()).array() * (aDiff.col(j).array() - aDiff.col(j).array().mean()).array() ).sum() / ((double) aDiff.col(i).array().size() - 1);
    }

    MatrixXd wt = MatrixXd::Zero(nMoments, nMoments);
   
    for(int i = 0; i < nMoments; i++){
        wt(i,i) = variances(i); // cleanup code and make it more vectorized later.
    }
    for(int i = 0; i < nMoments - 1; i++){
        int j = i + 1;
        wt(i,j) = covariances(i);
        wt(j,i) = covariances(i);
    }
    
    cout << "Weights Before Inversion:" << endl << wt << endl;
    wt = wt.llt().solve(MatrixXd::Identity(nMoments, nMoments));
    // for(int i = 0; i < nMoments; i++){
    //     wt(i,i) = 1 / variances(i); // cleanup code and make it more vectorized later.
    // }
    cout << "Weights:" << endl;
    cout << wt << endl;
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
    double alpha = 0.2;
    int nRuns = 1;
    int N = 5000;
    int nParts = 50; // blind PSO  1000:10
    int nSteps = 200;
    int nParts2 = 1; // targeted PSO
    int nSteps2 = 2;
    int nMoments = (N_SPECIES * (N_SPECIES + 3)) / 2; // var + mean + cov
    int hone = 28;
    //nMoments = 2*N_SPECIES; // mean + var only!
    int nRestarts = 2;
    VectorXd wmatup(nRestarts);
    wmatup << 0.30, 0.7;
    double uniLowBound = 0.0, uniHiBound = 1.0;
    random_device RanDev;
    mt19937 gen(RanDev());
    uniform_real_distribution<double> unifDist(uniLowBound, uniHiBound);
    
    vector<MatrixXd> weights;
    bool useOnlySecMom = true;
    if(useOnlySecMom){
        cout << "USING NONMIXED MOMENTS!!" << endl;
        nMoments = 2 * N_SPECIES;
    }
    for(int i = 0; i < nTimeSteps; i++){
        weights.push_back(MatrixXd::Identity(nMoments, nMoments));
    }
    cout << "nRuns:" << nRuns << endl;
    cout << "Using two part PSO " << "Sample Size:" << N << " with:" << nMoments << " moments." << endl;
    cout << "Using Times:" << times.transpose() << endl;
    cout << "Bounds for Uniform Distribution (" << uniLowBound << "," << uniHiBound << ")"<< endl;
    cout << "Blind PSO --> nParts:" << nParts << " Nsteps:" << nSteps << endl;
    cout << "Targeted PSO --> nParts:" <<  nParts2 << " Nsteps:" << nSteps2 << endl;
    cout << "sdbeta:" << sdbeta << endl;
    // cout << "wt:" << endl << wt << endl;

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

    /* Solve for Y_t (mu). */
    cout << "Loading in Truk!" << endl;
    struct K tru;
    tru.k = VectorXd::Zero(Npars);
    tru.k << 0.1, 0.1, 0.95, 0.17, 0.05, 0.18;

    cout << "Calculating Yt!" << endl;
    vector<MatrixXd> Yt3Mats;
    vector<VectorXd> Yt3Vecs;
    vector<VectorXd> Xt3Vecs;
    Controlled_RK_Stepper_N controlledStepper;
    double trukCost = 0;
    for(int t = 0; t < nTimeSteps; t++){
        Nonlinear_ODE6 trueSys(tru);
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
            integrate_adaptive(controlledStepper, trueSys, x0, t0, times(t), dt, XtObs);
        }
        Yt.mVec /= N;
        Xt.mVec /= N;
        trukCost += calculate_cf2(Yt.mVec,Xt.mVec, weights[t]);
        Xt3Vecs.push_back(Xt.mVec);
        Yt3Mats.push_back(Yt.mat);
        Yt3Vecs.push_back(Yt.mVec);
    }
    cout << "truk cost:"<< trukCost << endl;

    MatrixXd GBVECS = MatrixXd::Zero(nRuns, Npars + 1);
    for(int run = 0; run < nRuns; run++){
        // make sure to reset GBMAT, POSMAT, AND PBMAT EVERY RUN!
        double sfi = sfe, sfc = sfp, sfs = sfg; // below are the variables being used to reiterate weights
        MatrixXd GBMAT = MatrixXd::Zero(0,0); // iterations of global best vectors
        MatrixXd PBMAT = MatrixXd::Zero(nParts, Npars + 1); // particle best matrix + 1 for cost component
        MatrixXd POSMAT = MatrixXd::Zero(nParts, Npars); // Position matrix as it goees through it in parallel
        // reset weight matrices back to identity
        weights[0] <<1.09422,0.0010906,0.00278112,-0.0156409,-0.00170461,-0.000388485,0.00102581,8.25739e-10,5.40332e-09,-8.88803e-08,-6.35746e-10,-9.2078e-11,
        0.0010906,3.07445e-06,7.84013e-06,-4.40925e-05,-4.8054e-06,-1.09516e-06,2.89181e-06,2.32781e-12,1.52323e-11,-2.50559e-10,-1.7922e-12,-2.59573e-13,
        0.00278112,7.84013e-06,0.000265871,-0.00149525,-0.000162959,-3.71387e-05,9.80657e-05,7.89396e-11,5.1655e-10,-8.49684e-09,-6.07765e-11,-8.80254e-12,
        -0.0156409,-4.40925e-05,-0.00149525,0.0119129,0.00129832,0.00029589,-0.000781306,-6.28925e-10,-4.11545e-09,6.76958e-08,4.84216e-10,7.01313e-11,
        -0.00170461,-4.8054e-06,-0.000162959,0.00129832,0.000222773,5.07705e-05,-0.000134061,-1.07914e-10,-7.06151e-10,1.16156e-08,8.30845e-11,1.20335e-11,
        -0.000388485,-1.09516e-06,-3.71387e-05,0.00029589,5.07705e-05,0.00079705,-0.00210463,-1.69416e-09,-1.10859e-08,1.82354e-07,1.30435e-09,1.88915e-10,
        0.00102581,2.89181e-06,9.80657e-05,-0.000781306,-0.000134061,-0.00210463,0.02343,1.88604e-08,1.23415e-07,-2.03008e-06,-1.45208e-08,-2.10312e-09,
        8.25739e-10,2.32781e-12,7.89396e-11,-6.28925e-10,-1.07914e-10,-1.69416e-09,1.88604e-08,1.36098e-13,8.90571e-13,-1.46492e-11,-1.04783e-13,-1.51762e-14,
        5.40332e-09,1.52323e-11,5.1655e-10,-4.11545e-09,-7.06151e-10,-1.10859e-08,1.23415e-07,8.90571e-13,4.0512e-10,-6.6639e-09,-4.76658e-11,-6.90366e-12,
        -8.88803e-08,-2.50559e-10,-8.49684e-09,6.76958e-08,1.16156e-08,1.82354e-07,-2.03008e-06,-1.46492e-11,-6.6639e-09,3.27852e-07,2.34507e-09,3.39647e-10,
        -6.35746e-10,-1.7922e-12,-6.07765e-11,4.84216e-10,8.30845e-11,1.30435e-09,-1.45208e-08,-1.04783e-13,-4.76658e-11,2.34507e-09,5.24271e-10,7.59326e-11,
        -9.2078e-11,-2.59573e-13,-8.80254e-12,7.01313e-11,1.20335e-11,1.88915e-10,-2.10312e-09,-1.51762e-14,-6.90366e-12,3.39647e-10,7.59326e-11,1.28787e-08;


        weights[1] << 0.10697,0.00286477,0.0560647,-0.0317621,-0.0445589,0.00495941,-2.00699e-06,-6.16071e-11,-2.80581e-09,1.21374e-09,6.88116e-10,-2.06854e-11,
        0.00286477,9.38468e-05,0.00183662,-0.00104049,-0.0014597,0.000162465,-6.57469e-08,-2.01818e-12,-9.19151e-11,3.97608e-11,2.25419e-11,-6.77631e-13,
        0.0560647,0.00183662,0.0366777,-0.0207789,-0.0291506,0.00324446,-1.31298e-06,-4.03036e-11,-1.83557e-09,7.94033e-10,4.50168e-10,-1.35325e-11,
        -0.0317621,-0.00104049,-0.0207789,0.0118,0.0165541,-0.00184247,7.45619e-07,2.28877e-11,1.04239e-09,-4.50917e-10,-2.55642e-10,7.68483e-12,
        -0.0445589,-0.0014597,-0.0291506,0.0165541,0.0235194,-0.00261771,1.05935e-06,3.25179e-11,1.48098e-09,-6.40645e-10,-3.63206e-10,1.09183e-11,
        0.00495941,0.000162465,0.00324446,-0.00184247,-0.00261771,0.000413975,-1.67529e-07,-5.14251e-12,-2.34208e-10,1.01314e-10,5.74389e-11,-1.72667e-12,
        -2.00699e-06,-6.57469e-08,-1.31298e-06,7.45619e-07,1.05935e-06,-1.67529e-07,3.92714e-07,1.20548e-11,5.4902e-10,-2.37496e-10,-1.34646e-10,4.04757e-12,
        -6.16071e-11,-2.01818e-12,-4.03036e-11,2.28877e-11,3.25179e-11,-5.14251e-12,1.20548e-11,1.92102e-13,8.74903e-12,-3.78467e-12,-2.14567e-12,6.45009e-14,
        -2.80581e-09,-9.19151e-11,-1.83557e-09,1.04239e-09,1.48098e-09,-2.34208e-10,5.4902e-10,8.74903e-12,1.68792e-09,-7.30165e-10,-4.13958e-10,1.2444e-11,
        1.21374e-09,3.97608e-11,7.94033e-10,-4.50917e-10,-6.40645e-10,1.01314e-10,-2.37496e-10,-3.78467e-12,-7.30165e-10,3.83666e-10,2.17515e-10,-6.5387e-12,
        6.88116e-10,2.25419e-11,4.50168e-10,-2.55642e-10,-3.63206e-10,5.74389e-11,-1.34646e-10,-2.14567e-12,-4.13958e-10,2.17515e-10,3.13604e-09,-9.42722e-11,
        -2.06854e-11,-6.77631e-13,-1.35325e-11,7.68483e-12,1.09183e-11,-1.72667e-12,4.04757e-12,6.45009e-14,1.2444e-11,-6.5387e-12,-9.42722e-11,5.99071e-10;



        weights[2] <<6.07665e-05,7.56214e-06,-1.82441e-05,1.96504e-07,1.68261e-06,-2.31252e-08,-2.39347e-12,-2.39666e-14,4.55913e-14,-1.10389e-16,-9.32386e-16,6.76446e-18,
        7.56214e-06,2.62038e-06,-6.32182e-06,6.80914e-08,5.83046e-07,-8.01321e-09,-8.29369e-13,-8.30476e-15,1.5798e-14,-3.82512e-17,-3.23085e-16,2.34398e-18,
        -1.82441e-05,-6.32182e-06,0.000145739,-1.56973e-06,-1.34411e-05,1.84731e-07,1.91197e-11,1.91452e-13,-3.64196e-13,8.81815e-16,7.44816e-15,-5.40365e-17,
        1.96504e-07,6.80914e-08,-1.56973e-06,1.98461e-06,1.69936e-05,-2.33555e-07,-2.4173e-11,-2.42052e-13,4.60452e-13,-1.11488e-15,-9.4167e-15,6.83182e-17,
        1.68261e-06,5.83046e-07,-1.34411e-05,1.69936e-05,0.00961031,-0.000132081,-1.36704e-08,-1.36887e-10,2.60397e-10,-6.30491e-13,-5.32538e-12,3.86357e-14,
        -2.31252e-08,-8.01321e-09,1.84731e-07,-2.33555e-07,-0.000132081,7.89916e-05,8.17565e-09,8.18656e-11,-1.55732e-10,3.77068e-13,3.18486e-12,-2.31062e-14,
        -2.39347e-12,-8.29369e-13,1.91197e-11,-2.4173e-11,-1.36704e-08,8.17565e-09,3.44482e-10,3.44942e-12,-6.56177e-12,1.58878e-14,1.34195e-13,-9.73582e-16,
        -2.39666e-14,-8.30476e-15,1.91452e-13,-2.42052e-13,-1.36887e-10,8.18656e-11,3.44942e-12,5.10859e-13,-9.71799e-13,2.35298e-15,1.98742e-14,-1.44188e-16,
        4.55913e-14,1.5798e-14,-3.64196e-13,4.60452e-13,2.60397e-10,-1.55732e-10,-6.56177e-12,-9.71799e-13,1.66721e-09,-4.03675e-12,-3.4096e-11,2.47367e-13,
        -1.10389e-16,-3.82512e-17,8.81815e-16,-1.11488e-15,-6.30491e-13,3.77068e-13,1.58878e-14,2.35298e-15,-4.03675e-12,2.72583e-13,2.30234e-12,-1.67035e-14,
        -9.32386e-16,-3.23085e-16,7.44816e-15,-9.4167e-15,-5.32538e-12,3.18486e-12,1.34195e-13,1.98742e-14,-3.4096e-11,2.30234e-12,7.00819e-08,-5.08445e-10,
        6.76446e-18,2.34398e-18,-5.40365e-17,6.83182e-17,3.86357e-14,-2.31062e-14,-9.73582e-16,-1.44188e-16,2.47367e-13,-1.67035e-14,-5.08445e-10,2.73523e-10;



        weights[3] <<8.14592e-05,2.0546e-05,-9.3966e-05,-7.05591e-07,-8.61796e-06,7.61065e-08,1.34891e-11,2.62126e-13,-5.36688e-12,-1.11234e-16,-3.04599e-15,1.63608e-17,
        2.0546e-05,1.23055e-05,-5.62784e-05,-4.22594e-07,-5.16149e-06,4.55819e-08,8.07894e-12,1.56993e-13,-3.21434e-12,-6.66206e-17,-1.82431e-15,9.79884e-18,
        -9.3966e-05,-5.62784e-05,0.000582398,4.37322e-06,5.34138e-05,-4.71705e-07,-8.3605e-11,-1.62465e-12,3.32637e-11,6.89425e-16,1.88789e-14,-1.01403e-16,
        -7.05591e-07,-4.22594e-07,4.37322e-06,1.93245e-06,2.36026e-05,-2.08438e-07,-3.69436e-11,-7.17902e-13,1.46986e-11,3.04645e-16,8.34226e-15,-4.48084e-17,
        -8.61796e-06,-5.16149e-06,5.34138e-05,2.36026e-05,0.0209139,-0.000184694,-3.27352e-08,-6.36123e-10,1.30243e-08,2.69941e-13,7.39196e-12,-3.97041e-14,
        7.61065e-08,4.55819e-08,-4.71705e-07,-2.08438e-07,-0.000184694,7.80818e-05,1.38392e-08,2.68929e-10,-5.50617e-09,-1.14121e-13,-3.12505e-12,1.67854e-14,
        1.34891e-11,8.07894e-12,-8.3605e-11,-3.69436e-11,-3.27352e-08,1.38392e-08,3.44154e-10,6.68773e-12,-1.36927e-10,-2.83796e-15,-7.77136e-14,4.1742e-16,
        2.62126e-13,1.56993e-13,-1.62465e-12,-7.17902e-13,-6.36123e-10,2.68929e-10,6.68773e-12,2.64192e-12,-5.40918e-11,-1.12111e-15,-3.07e-14,1.64897e-16,
        -5.36688e-12,-3.21434e-12,3.32637e-11,1.46986e-11,1.30243e-08,-5.50617e-09,-1.36927e-10,-5.40918e-11,1.14225e-08,2.36743e-13,6.48286e-12,-3.48211e-14,
        -1.11234e-16,-6.66206e-17,6.89425e-16,3.04645e-16,2.69941e-13,-1.14121e-13,-2.83796e-15,-1.12111e-15,2.36743e-13,1.42858e-13,3.91195e-12,-2.10121e-14,
        -3.04599e-15,-1.82431e-15,1.88789e-14,8.34226e-15,7.39196e-12,-3.12505e-12,-7.77136e-14,-3.07e-14,6.48286e-12,3.91195e-12,1.92329e-07,-1.03305e-09,
        1.63608e-17,9.79884e-18,-1.01403e-16,-4.48084e-17,-3.97041e-14,1.67854e-14,4.1742e-16,1.64897e-16,-3.48211e-14,-2.10121e-14,-1.03305e-09,2.66698e-10;



        weights[4] <<8.09574e-05,2.01789e-05,-8.61975e-05,-3.58452e-07,-4.73831e-06,3.77063e-08,6.53837e-12,1.36791e-13,-3.78602e-12,-6.5875e-17,-2.61153e-15,1.30799e-17,
        2.01789e-05,2.37142e-05,-0.000101299,-4.21254e-07,-5.56847e-06,4.43125e-08,7.6839e-12,1.60757e-13,-4.44934e-12,-7.74164e-17,-3.06908e-15,1.53715e-17,
        -8.61975e-05,-0.000101299,0.00136819,5.68963e-06,7.52101e-05,-5.98503e-07,-1.03782e-10,-2.17125e-12,6.00947e-11,1.04562e-15,4.14523e-14,-2.07614e-16,
        -3.58452e-07,-4.21254e-07,5.68963e-06,1.91036e-06,2.52526e-05,-2.00954e-07,-3.4846e-11,-7.29022e-13,2.01774e-11,3.51078e-16,1.39181e-14,-6.97087e-17,
        -4.73831e-06,-5.56847e-06,7.52101e-05,2.52526e-05,0.024241,-0.000192904,-3.34501e-08,-6.99818e-10,1.93691e-08,3.37014e-13,1.33605e-11,-6.69162e-14,
        3.77063e-08,4.43125e-08,-5.98503e-07,-2.00954e-07,-0.000192904,7.77761e-05,1.34866e-08,2.82156e-10,-7.80935e-09,-1.35879e-13,-5.38676e-12,2.69796e-14,
        6.53837e-12,7.6839e-12,-1.03782e-10,-3.4846e-11,-3.34501e-08,1.34866e-08,3.55822e-10,7.44426e-12,-2.06038e-10,-3.58496e-15,-1.42121e-13,7.11815e-16,
        1.36791e-13,1.60757e-13,-2.17125e-12,-7.29022e-13,-6.99818e-10,2.82156e-10,7.44426e-12,7.92824e-12,-2.19433e-10,-3.81803e-15,-1.51361e-13,7.58094e-16,
        -3.78602e-12,-4.44934e-12,6.00947e-11,2.01774e-11,1.93691e-08,-7.80935e-09,-2.06038e-10,-2.19433e-10,5.41029e-08,9.41365e-13,3.73193e-11,-1.86914e-13,
        -6.5875e-17,-7.74164e-17,1.04562e-15,3.51078e-16,3.37014e-13,-1.35879e-13,-3.58496e-15,-3.81803e-15,9.41365e-13,1.22545e-13,4.85816e-12,-2.43321e-14,
        -2.61153e-15,-3.06908e-15,4.14523e-14,1.39181e-14,1.33605e-11,-5.38676e-12,-1.42121e-13,-1.51361e-13,3.73193e-11,4.85816e-12,2.32447e-07,-1.16421e-09,
        1.30799e-17,1.53715e-17,-2.07614e-16,-6.97087e-17,-6.69162e-14,2.69796e-14,7.11815e-16,7.58094e-16,-1.86914e-13,-2.43321e-14,-1.16421e-09,2.65506e-10;


        /* Instantiate seedk aka global costs */
        struct K seed;
        seed.k = VectorXd::Zero(Npars); 
        //seed.k = testVec;
        for (int i = 0; i < Npars; i++) { 
            seed.k(i) = unifDist(gen);
        }

        // seed.k = tru.k;
        double costSeedK = 0;
        for(int t = 0; t < nTimeSteps; t++){
            Protein_Components Xt(times(t), nMoments, N);
            Moments_Mat_Obs XtObs(Xt);
            Nonlinear_ODE6 sys(seed);
            for (int i = 0; i < N; i++) {
                //State_N c0 = gen_multi_norm_iSub();
                State_N c0 = convertInit(X_0, i);
                Xt.index = i;
                integrate_adaptive(controlledStepper, sys, c0, t0, times(t), dt, XtObs);
            }
            Xt.mVec /= N;  
            cout << "XtmVec:" << Xt.mVec.transpose() << endl;
            costSeedK += calculate_cf2(Yt3Vecs[t], Xt.mVec, weights[t]);
        }

        cout << "seedk:"<< seed.k.transpose()<< "| cost:" << costSeedK << endl;
        
        double gCost = costSeedK; //initialize costs and GBMAT
        // global values
        VectorXd GBVEC = seed.k;
        
        GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1);
        for (int i = 0; i < Npars; i++) {
            GBMAT(GBMAT.rows() - 1, i) = seed.k(i);
        }
        GBMAT(GBMAT.rows() - 1, Npars) = gCost;
        
        /* Blind PSO begins */
        cout << "PSO begins!" << endl;
        for(int step = 0; step < nSteps; step++){
        #pragma omp parallel for 
            for(int particle = 0; particle < nParts; particle++){
                random_device pRanDev;
                mt19937 pGenerator(pRanDev());
                uniform_real_distribution<double> pUnifDist(uniLowBound, uniHiBound);
                /* instantiate all particle rate constants with unifDist */
                if(step == 0){
                    /* temporarily assign specified k constants */
                    for(int i = 0; i < Npars; i++){
                        POSMAT(particle, i) = pUnifDist(pGenerator);
                    }

                    // POSMAT.row(particle) = seed.k;

                    struct K pos;
                    pos.k = VectorXd::Zero(Npars);
                    for(int i = 0; i < Npars; i++){
                        pos.k(i) = POSMAT(particle, i);
                    }
                    double cost = 0;
                    for(int t = 0; t < nTimeSteps; t++){
                        Nonlinear_ODE6 initSys(pos);
                        Protein_Components XtPSO(times(t), nMoments, N);
                        Moments_Mat_Obs XtObsPSO(XtPSO);
                        for(int i = 0; i < N; i++){
                            //State_N c0 = gen_multi_norm_iSub();
                            State_N c0 = convertInit(X_0, i);
                            XtPSO.index = i;
                            integrate_adaptive(controlledStepper, initSys, c0, t0, times(t), dt, XtObsPSO);
                        }
                        XtPSO.mVec/=N;
                        cost += calculate_cf2(Yt3Vecs[t], XtPSO.mVec, weights[t]);
                    }
                    
                    
                    /* instantiate PBMAT */
                    for(int i = 0; i < Npars; i++){
                        PBMAT(particle, i) = POSMAT(particle, i);
                    }
                    PBMAT(particle, Npars) = cost; // add cost to final column
                }else{ 
                    /* using new rate constants, instantiate particle best values */
                    /* step into PSO */
                    double w1 = sfi * pUnifDist(pGenerator)/ sf2, w2 = sfc * pUnifDist(pGenerator) / sf2, w3 = sfs * pUnifDist(pGenerator)/ sf2;
                    double sumw = w1 + w2 + w3; //w1 = inertial, w2 = pbest, w3 = gbest
                    w1 = w1 / sumw; w2 = w2 / sumw; w3 = w3 / sumw;
                    //w1 = 0.05; w2 = 0.90; w3 = 0.05;
                    struct K pos;
                    pos.k = VectorXd::Zero(Npars);
                    pos.k = POSMAT.row(particle);
                    VectorXd rpoint = comp_vel_vec(pos.k, particle, epsi, nan, hone);
                    VectorXd PBVEC(Npars);
                    for(int i = 0; i < Npars; i++){
                        PBVEC(i) = PBMAT(particle, i);
                    }
                   
                    pos.k = w1 * rpoint + w2 * PBVEC + w3 * GBVEC; // update position of particle
                    
                    if(pUnifDist(pGenerator) < (3.0/4.0)){ // hard coded grid re-search for an adaptive component
                        pos.k(0) = pUnifDist(pGenerator);
                        pos.k(1) = pUnifDist(pGenerator);
                        pos.k(4) = pUnifDist(pGenerator);
                    }
                    POSMAT.row(particle) = pos.k;
                    double cost = 0;
                    for(int t = 0; t < nTimeSteps; t++){
                        /*solve ODEs and recompute cost */
                        Protein_Components XtPSO(times(t), nMoments, N);
                        Moments_Mat_Obs XtObsPSO1(XtPSO);
                        Nonlinear_ODE6 stepSys(pos);
                        for(int i = 0; i < N; i++){
                            State_N c0 = convertInit(X_0, i);
                            XtPSO.index = i;
                            integrate_adaptive(controlledStepper, stepSys, c0, t0, times(t), dt, XtObsPSO1);
                        }
                        XtPSO.mVec/=N;
                        cost += calculate_cf2(Yt3Vecs[t], XtPSO.mVec, weights[t]);
                    }
                
                    /* update gBest and pBest */
                    #pragma omp critical
                {
                    if(cost < PBMAT(particle, Npars)){ // particle best cost
                        for(int i = 0; i < Npars; i++){
                            PBMAT(particle, i) = pos.k(i);
                        }
                        PBMAT(particle, Npars) = cost;
                        if(cost < gCost){
                            gCost = cost;
                            GBVEC = pos.k;
                        }   
                    }
                }
                }
            }
            GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1); // Add to GBMAT after resizing
            for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = GBVEC(i);}
            GBMAT(GBMAT.rows() - 1, Npars) = gCost;
            sfi = sfi - (sfe - sfg) / nSteps;   // reduce the inertial weight after each step 
            sfs = sfs + (sfe - sfg) / nSteps;
            cout << "current:" << GBVEC.transpose()<<" "<< gCost << endl;
        }

        cout << "GBMAT from blind PSO:" << endl << endl;
        cout << GBMAT << endl << endl;
        cout << "truk: " << tru.k.transpose() << endl;
        double dist = calculate_cf1(tru.k, GBVEC);
        cout << "total difference b/w truk and final GBVEC" << dist << endl << endl; // compute difference
        auto tB = std::chrono::high_resolution_clock::now();
        auto bDuration = std::chrono::duration_cast<std::chrono::seconds>(tB - t1).count();
        cout << "blind PSO FINISHED RUNNING IN " << bDuration << " s TIME!" << endl;
        /*** targeted PSO ***/
        // POSMAT.conservativeResize(nParts2, Npars); // resize matrices to fit targetted PSO
        // PBMAT.conservativeResize(nParts2, Npars + 1);

    
        // // weights[1] << 5.85597e-05,7.55787e-06,-2.04086e-05,1.19984e-07,1.75691e-06,-2.06913e-08,-3.00832e-12,-3.29187e-14,1.1358e-13,-2.14347e-16,-1.77549e-14,4.6905e-17,
        // // 7.55787e-06,2.607e-06,-7.03971e-06,4.13872e-08,6.06025e-07,-7.1372e-09,-1.03768e-12,-1.13549e-14,3.91782e-14,-7.39363e-17,-6.12436e-15,1.61793e-17,
        // // -2.04086e-05,-7.03971e-06,0.00014658,-8.61757e-07,-1.26185e-05,1.4861e-07,2.16065e-11,2.3643e-13,-8.15761e-13,1.53949e-15,1.2752e-13,-3.36883e-16,
        // // 1.19984e-07,4.13872e-08,-8.61757e-07,1.9014e-06,2.78419e-05,-3.27896e-07,-4.76731e-11,-5.21666e-13,1.79992e-12,-3.39677e-15,-2.81364e-13,7.43309e-16,
        // // 1.75691e-06,6.06025e-07,-1.26185e-05,2.78419e-05,0.0127413,-0.000150055,-2.18166e-08,-2.3873e-10,8.23695e-10,-1.55446e-12,-1.28761e-10,3.4016e-13,
        // // -2.06913e-08,-7.1372e-09,1.4861e-07,-3.27896e-07,-0.000150055,7.85291e-05,1.14174e-08,1.24936e-10,-4.31069e-10,8.13505e-13,6.7385e-11,-1.78018e-13,
        // // -3.00832e-12,-1.03768e-12,2.16065e-11,-4.76731e-11,-2.18166e-08,1.14174e-08,3.20613e-10,3.50833e-12,-1.21049e-11,2.28441e-14,1.89224e-12,-4.99893e-15,
        // // -3.29187e-14,-1.13549e-14,2.3643e-13,-5.21666e-13,-2.3873e-10,1.24936e-10,3.50833e-12,5.07179e-13,-1.74993e-12,3.30244e-15,2.73551e-13,-7.22667e-16,
        // // 1.1358e-13,3.91782e-14,-8.15761e-13,1.79992e-12,8.23695e-10,-4.31069e-10,-1.21049e-11,-1.74993e-12,1.62506e-09,-3.06678e-12,-2.54031e-10,6.71098e-13,
        // // -2.14347e-16,-7.39363e-17,1.53949e-15,-3.39677e-15,-1.55446e-12,8.13505e-13,2.28441e-14,3.30244e-15,-3.06678e-12,2.56635e-13,2.12579e-11,-5.61591e-14,
        // // -1.77549e-14,-6.12436e-15,1.2752e-13,-2.81364e-13,-1.28761e-10,6.7385e-11,1.89224e-12,2.73551e-13,-2.54031e-10,2.12579e-11,4.72203e-07,-1.24747e-09,
        // // 4.6905e-17,1.61793e-17,-3.36883e-16,7.43309e-16,3.4016e-13,-1.78018e-13,-4.99893e-15,-7.22667e-16,6.71098e-13,-5.61591e-14,-1.24747e-09,2.82603e-10;

        // // weights[2] << 8.08179e-05,2.18336e-05,-9.74166e-05,-5.01008e-07,-1.10249e-05,7.43191e-08,1.55166e-11,3.92075e-13,-7.72855e-12,7.72232e-16,1.86221e-13,-2.32887e-16,
        // // 2.18336e-05,1.2975e-05,-5.78915e-05,-2.97733e-07,-6.55173e-06,4.41654e-08,9.22099e-12,2.32997e-13,-4.59283e-12,4.58912e-16,1.10665e-13,-1.38397e-16,
        // // -9.74166e-05,-5.78915e-05,0.000555521,2.85701e-06,6.28697e-05,-4.23807e-07,-8.84836e-11,-2.23582e-12,4.40723e-11,-4.40368e-15,-1.06193e-12,1.32804e-15,
        // // -5.01008e-07,-2.97733e-07,2.85701e-06,1.82871e-06,4.02414e-05,-2.71269e-07,-5.66363e-11,-1.4311e-12,2.82096e-11,-2.81869e-15,-6.79718e-13,8.50051e-16,
        // // -1.10249e-05,-6.55173e-06,6.28697e-05,4.02414e-05,0.034622,-0.000233388,-4.87274e-08,-1.23125e-09,2.42703e-08,-2.42508e-12,-5.84799e-10,7.31346e-13,
        // // 7.43191e-08,4.41654e-08,-4.23807e-07,-2.71269e-07,-0.000233388,7.76757e-05,1.62174e-08,4.09783e-10,-8.07761e-09,8.0711e-13,1.94632e-10,-2.43405e-13,
        // // 1.55166e-11,9.22099e-12,-8.84836e-11,-5.66363e-11,-4.87274e-08,1.62174e-08,3.24545e-10,8.20066e-12,-1.61651e-10,1.6152e-14,3.89501e-12,-4.87107e-15,
        // // 3.92075e-13,2.32997e-13,-2.23582e-12,-1.4311e-12,-1.23125e-09,4.09783e-10,8.20066e-12,2.97249e-12,-5.85935e-11,5.85463e-15,1.41182e-12,-1.76562e-15,
        // // -7.72855e-12,-4.59283e-12,4.40723e-11,2.82096e-11,2.42703e-08,-8.07761e-09,-1.61651e-10,-5.85935e-11,1.04699e-08,-1.04614e-12,-2.52274e-10,3.15492e-13,
        // // 7.72232e-16,4.58912e-16,-4.40368e-15,-2.81869e-15,-2.42508e-12,8.0711e-13,1.6152e-14,5.85463e-15,-1.04614e-12,1.34104e-13,3.23388e-11,-4.04426e-14,
        // // 1.86221e-13,1.10665e-13,-1.06193e-12,-6.79718e-13,-5.84799e-10,1.94632e-10,3.89501e-12,1.41182e-12,-2.52274e-10,3.23388e-11,2.69139e-06,-3.36583e-09,
        // // -2.32887e-16,-1.38397e-16,1.32804e-15,8.50051e-16,7.31346e-13,-2.43405e-13,-4.87107e-15,-1.76562e-15,3.15492e-13,-4.04426e-14,-3.36583e-09,2.75599e-10;

        // // weights[3] <<7.90803e-05,2.13992e-05,-9.06126e-05,-3.19084e-07,-7.08527e-06,4.09256e-08,8.21219e-12,2.76542e-13,-7.44298e-12,-5.29405e-17,-1.48428e-14,1.54322e-17,
        // // 2.13992e-05,2.34067e-05,-9.91131e-05,-3.49018e-07,-7.74994e-06,4.47648e-08,8.98259e-12,3.02485e-13,-8.14122e-12,-5.79069e-17,-1.62352e-14,1.68799e-17,
        // // -9.06126e-05,-9.91131e-05,0.00134513,4.73674e-06,0.000105179,-6.07532e-07,-1.21908e-10,-4.10521e-12,1.1049e-10,7.85891e-16,2.20338e-13,-2.29088e-16,
        // // -3.19084e-07,-3.49018e-07,4.73674e-06,1.79515e-06,3.98613e-05,-2.30245e-07,-4.62013e-11,-1.55581e-12,4.18738e-11,2.9784e-16,8.35046e-14,-8.68206e-17,
        // // -7.08527e-06,-7.74994e-06,0.000105179,3.98613e-05,0.0425862,-0.000245984,-4.93596e-08,-1.66216e-09,4.47363e-08,3.18201e-13,8.92131e-11,-9.27557e-14,
        // // 4.09256e-08,4.47648e-08,-6.07532e-07,-2.30245e-07,-0.000245984,7.72621e-05,1.55035e-08,5.22075e-10,-1.40514e-08,-9.99448e-14,-2.80213e-11,2.9134e-14,
        // // 8.21219e-12,8.98259e-12,-1.21908e-10,-4.62013e-11,-4.93596e-08,1.55035e-08,3.35503e-10,1.12979e-11,-3.04078e-10,-2.16285e-15,-6.06392e-13,6.30471e-16,
        // // 2.76542e-13,3.02485e-13,-4.10521e-12,-1.55581e-12,-1.66216e-09,5.22075e-10,1.12979e-11,9.37209e-12,-2.52245e-10,-1.79417e-15,-5.03027e-13,5.23002e-16,
        // // -7.44298e-12,-8.14122e-12,1.1049e-10,4.18738e-11,4.47363e-08,-1.40514e-08,-3.04078e-10,-2.52245e-10,5.4283e-08,3.86105e-13,1.08251e-10,-1.1255e-13,
        // // -5.29405e-17,-5.79069e-17,7.85891e-16,2.9784e-16,3.18201e-13,-9.99448e-14,-2.16285e-15,-1.79417e-15,3.86105e-13,1.1099e-13,3.11179e-11,-3.23535e-14,
        // // -1.48428e-14,-1.62352e-14,2.20338e-13,8.35046e-14,8.92131e-11,-2.80213e-11,-6.06392e-13,-5.03027e-13,1.08251e-10,3.11179e-11,3.89104e-06,-4.04555e-09,
        // // 1.54322e-17,1.68799e-17,-2.29088e-16,-8.68206e-17,-9.27557e-14,2.9134e-14,6.30471e-16,5.23002e-16,-1.1255e-13,-3.23535e-14,-4.04555e-09,2.74214e-10;

        // //  weights[4] <<7.49654e-05,1.93747e-05,-6.03444e-05,-1.62145e-07,-3.5346e-06,1.94493e-08,3.69769e-12,1.30161e-13,-3.61294e-12,-8.02191e-17,-2.30487e-14,2.30203e-17,
        // // 1.93747e-05,3.69692e-05,-0.000115144,-3.09392e-07,-6.74442e-06,3.71115e-08,7.05563e-12,2.48363e-13,-6.89391e-12,-1.53067e-16,-4.39796e-14,4.39254e-17,
        // // -6.03444e-05,-0.000115144,0.00189805,5.10005e-06,0.000111176,-6.1175e-07,-1.16306e-10,-4.09404e-12,1.1364e-10,2.52318e-15,7.24964e-13,-7.24071e-16,
        // // -1.62145e-07,-3.09392e-07,5.10005e-06,1.81353e-06,3.95331e-05,-2.17533e-07,-4.13573e-11,-1.4558e-12,4.04093e-11,8.9722e-16,2.57791e-13,-2.57473e-16,
        // // -3.5346e-06,-6.74442e-06,0.000111176,3.95331e-05,0.0444089,-0.000244362,-4.64581e-08,-1.63535e-09,4.53932e-08,1.00788e-12,2.89585e-10,-2.89228e-13,
        // // 1.94493e-08,3.71115e-08,-6.1175e-07,-2.17533e-07,-0.000244362,7.70738e-05,1.46532e-08,5.15804e-10,-1.43174e-08,-3.17893e-13,-9.13375e-11,9.12249e-14,
        // // 3.69769e-12,7.05563e-12,-1.16306e-10,-4.13573e-11,-4.64581e-08,1.46532e-08,3.3234e-10,1.16986e-11,-3.24723e-10,-7.20992e-15,-2.07156e-12,2.06901e-15,
        // // 1.30161e-13,2.48363e-13,-4.09404e-12,-1.4558e-12,-1.63535e-09,5.15804e-10,1.16986e-11,1.8074e-11,-5.01687e-10,-1.11391e-14,-3.2005e-12,3.19656e-15,
        // // -3.61294e-12,-6.89391e-12,1.1364e-10,4.04093e-11,4.53932e-08,-1.43174e-08,-3.24723e-10,-5.01687e-10,1.30793e-07,2.90404e-12,8.34393e-10,-8.33365e-13,
        // // -8.02191e-17,-1.53067e-16,2.52318e-15,8.9722e-16,1.00788e-12,-3.17893e-13,-7.20992e-15,-1.11391e-14,2.90404e-12,1.07916e-13,3.10066e-11,-3.09683e-14,
        // // -2.30487e-14,-4.39796e-14,7.24964e-13,2.57791e-13,2.89585e-10,-9.13375e-11,-2.07156e-12,-3.2005e-12,8.34393e-10,3.10066e-11,4.16991e-06,-4.16477e-09,
        // // 2.30203e-17,4.39254e-17,-7.24071e-16,-2.57473e-16,-2.89228e-13,9.12249e-14,2.06901e-15,3.19656e-15,-8.33365e-13,-3.09683e-14,-4.16477e-09,2.7381e-10;

        // cout << "targeted PSO has started!" << endl; 
        // sfp = 3.0, sfg = 1.0, sfe = 6.0; // initial particle historical weight, global weight social, inertial
        // sfi = sfe, sfc = sfp, sfs = sfg; // below are the variables being used to reiterate weights
        // double nearby = sdbeta;
        // VectorXd chkpts = wmatup * nSteps2;
        // int chkptNo = 0;
        // for(int step = 0; step < nSteps2; step++){
        //     if(step == 0 || step == chkpts(chkptNo)){ /* update wt   matrix || step == chkpts(0) || step == chkpts(1) || step == chkpts(2) || step == chkpts(3) */
        //         cout << "Updating Weight Matrix!" << endl;
        //         cout << "GBVEC AND COST:" << GBMAT.row(GBMAT.rows() - 1) << endl;
        //         nearby = squeeze * nearby;
        //         /* reinstantiate gCost */
        //         struct K gPos;
        //         // GBVEC << 0.648691,	0.099861,	0.0993075,	0.8542755,	0.049949,	0.0705955;
        //         gPos.k = GBVEC;
                
        //         double cost = 0;
        //         for(int t = 0; t < nTimeSteps; t++){
        //             Protein_Components gXt(times(t), nMoments, N);
        //             Moments_Mat_Obs gXtObs(gXt);
        //             Nonlinear_ODE6 gSys(gPos);
        //             for (int i = 0; i < N; i++) {
        //                 //State_N c0 = gen_multi_norm_iSub();
        //                 State_N c0 = convertInit(X_0, i);
        //                 gXt.index = i;
        //                 integrate_adaptive(controlledStepper, gSys, c0, t0, times(t), dt, gXtObs);
        //             }
        //             gXt.mVec /= N;  
        //             weights[t] = customWtMat(Yt3Mats[t], gXt.mat, nMoments, N);
        //             // if(useOnlySecMom){
        //             //     for(int j = 2*N_SPECIES; j < nMoments; j++){
        //             //         weights[t](j,j) = 0;
        //             //     }
        //             // }
        //             cost += calculate_cf2(Yt3Vecs[t], gXt.mVec, weights[t]);
        //         }
        //         gCost = cost;
        //         hone += 4;
        //         GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1);
        //         for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = gPos.k(i);}
        //         GBMAT(GBMAT.rows() - 1, Npars) = gCost;
        //         if(step > 0 && chkptNo < nRestarts - 1){
        //             chkptNo++;
        //         }
        //     }
        // #pragma omp parallel for 
        //     for(int particle = 0; particle < nParts2; particle++){
        //         random_device pRanDev;
        //         mt19937 pGenerator(pRanDev());
        //         uniform_real_distribution<double> pUnifDist(uniLowBound, uniHiBound);
            
        //         if(step == 0 || step == chkpts(chkptNo)){
        //             /* reinitialize particles around global best */
        //             for(int edim = 0; edim < Npars; edim++){
        //                 int wasflipped = 0;
        //                 double tmean = GBVEC(edim);
        //                 if (GBVEC(edim) > 0.5) {
        //                     tmean = 1 - GBVEC(edim);
        //                     wasflipped = 1;
        //                 }
        //                 double myc = (1 - tmean) / tmean;
        //                 double alpha = myc / ((1 + myc) * (1 + myc) * (1 + myc)*nearby*nearby);
        //                 double beta = myc * alpha;

        //                 if(alpha < nan){
        //                     alpha = epsi;
        //                 }
        //                 if(beta < nan){
        //                     beta = epsi;
        //                 }

        //                 std::gamma_distribution<double> aDist(alpha, 1);
        //                 std::gamma_distribution<double> bDist(beta, 1);

        //                 double x = aDist(pGenerator);
        //                 double y = bDist(pGenerator);
        //                 double myg = x / (x + y);

        //                 if(myg >= 1){
        //                     myg = myg - epsi;
        //                 }
        //                 if(myg <= 0){
        //                     myg = myg + epsi;
        //                 }

        //                 if (wasflipped == 1) {
        //                     wasflipped = 0;
        //                     myg = 1 - myg;
        //                 }
        //                 POSMAT(particle, edim) = myg;
        //             }

        //             /* Write new POSMAT into Ks to be passed into system */
        //             struct K pos;
        //             pos.k = VectorXd::Zero(Npars);
        //             for(int i = 0; i < Npars; i++){
        //                 pos.k(i) = POSMAT(particle, i);
        //             }
        //             //VectorXd XtPSO3 = VectorXd::Zero(nMoments);
        //             double cost = 0;
        //             for(int t = 0; t < nTimeSteps; t++){
        //                 Nonlinear_ODE6 initSys(pos);
        //                 Protein_Components XtPSO(times(t), nMoments, N);
        //                 Moments_Mat_Obs XtObsPSO(XtPSO);
        //                 for(int i = 0; i < N; i++){
        //                     State_N c0 = convertInit(X_0, i);
        //                     XtPSO.index = i;
        //                     integrate_adaptive(controlledStepper, initSys, c0, t0, times(t), dt, XtObsPSO);
        //                 }
        //                 XtPSO.mVec/=N;
        //                 cost += calculate_cf2(Yt3Vecs[t], XtPSO.mVec, weights[t]);
        //             }
                    
        //             /* initialize PBMAT */
        //             for(int i = 0; i < Npars; i++){
        //                 PBMAT(particle, i) = POSMAT(particle, i);
        //             }
        //             PBMAT(particle, Npars) = cost; // add cost to final column
        //         }else{ 
        //             /* using new rate constants, initialize particle best values */
        //             /* step into PSO */
        //             double w1 = sfi * pUnifDist(pGenerator)/ sf2, w2 = sfc * pUnifDist(pGenerator) / sf2, w3 = sfs * pUnifDist(pGenerator)/ sf2;
        //             double sumw = w1 + w2 + w3; //w1 = inertial, w2 = pbest, w3 = gbest
        //             w1 = w1 / sumw; w2 = w2 / sumw; w3 = w3 / sumw;
        //             //w1 = 0.05; w2 = 0.90; w3 = 0.05;
        //             struct K pos;
        //             pos.k = VectorXd::Zero(Npars);
        //             pos.k = POSMAT.row(particle);
        //             VectorXd rpoint = comp_vel_vec(pos.k, particle, epsi, nan, hone);
        //             VectorXd PBVEC(Npars);
        //             for(int i = 0; i < Npars; i++){
        //                 PBVEC(i) = PBMAT(particle, i);
        //             }
        //             pos.k = w1 * rpoint + w2 * PBVEC + w3 * GBVEC; // update position of particle
        //             POSMAT.row(particle) = pos.k; // back into POSMAT
                    
        //             double cost = 0;
        //             /* solve ODEs with new system and recompute cost */
        //             for(int t = 0; t < nTimeSteps; t++){
        //                 Protein_Components XtPSO(times(t), nMoments, N);
        //                 Moments_Mat_Obs XtObsPSO1(XtPSO);
        //                 Nonlinear_ODE6 stepSys(pos);
        //                 for(int i = 0; i < N; i++){
        //                     State_N c0 = convertInit(X_0, i);
        //                     XtPSO.index = i;
        //                     integrate_adaptive(controlledStepper, stepSys, c0, t0, times(t), dt, XtObsPSO1);
        //                 }
        //                 XtPSO.mVec/=N;
        //                 cost += calculate_cf2(Yt3Vecs[t], XtPSO.mVec, weights[t]);
        //             }
                    
        //             /* update pBest and gBest */
        //             #pragma omp critical
        //             {
        //             if(cost < PBMAT(particle, Npars)){ // update particle best 
        //                 for(int i = 0; i < Npars; i++){
        //                     PBMAT(particle, i) = pos.k(i);
        //                 }
        //                 PBMAT(particle, Npars) = cost;
        //                 if(cost < gCost){ // update global 
        //                     gCost = cost;
        //                     GBVEC = pos.k;
        //                 }   
        //             }
        //             }
        //         }
        //     }
        //     GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1); // Add to GBMAT after each step.
        //     for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = GBVEC(i);}
        //     GBMAT(GBMAT.rows() - 1, Npars) = gCost;

        //     sfi = sfi - (sfe - sfg) / nSteps2;   // reduce the inertial weight after each step 
        //     sfs = sfs + (sfe - sfg) / nSteps2;

        //     if(step == 0){ // quick plug to see PBMAT
        //         cout << "New PBMAT:" << endl;
        //         cout << PBMAT << endl << endl;
        //     }
        //     // cout << "current:" << GBVEC.transpose()<<" "<< gCost << endl;
        // }
        // cout << "GBMAT after targeted PSO:" << endl << GBMAT << endl;
        if(run == nRuns - 1){
            printToCsv(GBMAT,"GBMATP");
        }
        for(int i = 0; i < Npars; i++){
            GBVECS(run, i) = GBVEC(i);
        }
        GBVECS(run, Npars) = gCost;
    }

    printToCsv(GBVECS,"runs");
    trukCost = 0;
    for(int t = 0; t < nTimeSteps; t++){
        trukCost += calculate_cf2(Yt3Vecs[t], Xt3Vecs[t], weights[t]);
    }

    cout << "truk: " << tru.k.transpose() << " with trukCost with new weights:" << trukCost << endl;
    
    for(int i = 0; i < nTimeSteps; i++){
        string fileName = "3TimeWeights_" + to_string(i);
        printToCsv(weights[i], fileName);
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << "CODE FINISHED RUNNING IN " << duration << " s TIME!" << endl;

    return 0; // just to close the program at the end.
}

