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
    int nParts = 1; // blind PSO  1000:10
    int nSteps = 1;
    int nParts2 = 1; // targeted PSO
    int nSteps2 = 1;
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

        // reset weight matrices back to some value
        weights[0] <<0.911619,0.000972317,0.00251757,-0.0139532,-0.00149306,-0.000371551,0.000303966,2.85853e-10,2.29692e-09,-4.05718e-08,-3.07561e-10,-6.14489e-11,
        0.000972317,3.06122e-06,7.92627e-06,-4.39299e-05,-4.70073e-06,-1.16978e-06,9.57001e-07,8.99973e-13,7.23157e-12,-1.27736e-10,-9.68319e-13,-1.93465e-13,
        0.00251757,7.92627e-06,0.000248823,-0.00137905,-0.000147566,-3.6722e-05,3.00423e-05,2.82521e-11,2.27014e-10,-4.00989e-09,-3.03976e-11,-6.07327e-12,
        -0.0139532,-4.39299e-05,-0.00137905,0.0110171,0.00117889,0.000293368,-0.000240004,-2.25702e-10,-1.81359e-09,3.20345e-08,2.42843e-10,4.85186e-11,
        -0.00149306,-4.70073e-06,-0.000147566,0.00117889,0.000205497,5.11381e-05,-4.18361e-05,-3.93431e-11,-3.16134e-10,5.58407e-09,4.23309e-11,8.45748e-12,
        -0.000371551,-1.16978e-06,-3.6722e-05,0.000293368,5.11381e-05,0.000642934,-0.000525985,-4.94642e-10,-3.9746e-09,7.02058e-08,5.32206e-10,1.06332e-10,
        0.000303966,9.57001e-07,3.00423e-05,-0.000240004,-4.18361e-05,-0.000525985,0.00540209,5.08018e-09,4.08209e-08,-7.21044e-07,-5.46598e-09,-1.09207e-09,
        2.85853e-10,8.99973e-13,2.82521e-11,-2.25702e-10,-3.93431e-11,-4.94642e-10,5.08018e-09,1.29213e-13,1.03827e-12,-1.83396e-11,-1.39026e-13,-2.77766e-14,
        2.29692e-09,7.23157e-12,2.27014e-10,-1.81359e-09,-3.16134e-10,-3.9746e-09,4.08209e-08,1.03827e-12,4.01678e-10,-7.09507e-09,-5.37853e-11,-1.0746e-11,
        -4.05718e-08,-1.27736e-10,-4.00989e-09,3.20345e-08,5.58407e-09,7.02058e-08,-7.21044e-07,-1.83396e-11,-7.09507e-09,3.48473e-07,2.64165e-09,5.27787e-10,
        -3.07561e-10,-9.68319e-13,-3.03976e-11,2.42843e-10,4.23309e-11,5.32206e-10,-5.46598e-09,-1.39026e-13,-5.37853e-11,2.64165e-09,5.72583e-10,1.14399e-10,
        -6.14489e-11,-1.93465e-13,-6.07327e-12,4.85186e-11,8.45748e-12,1.06332e-10,-1.09207e-09,-2.77766e-14,-1.0746e-11,5.27787e-10,1.14399e-10,1.22328e-08;

        weights[1] <<0.0143451,7.44139e-05,0.000648286,-0.000309878,-1.43894e-07,7.43749e-09,2.38814e-15,1.19078e-23,2.03819e-21,-2.75215e-22,-1.00296e-20,3.9832e-23,
        7.44139e-05,3.38794e-06,2.95153e-05,-1.41082e-05,-6.55126e-09,3.38616e-10,1.08728e-16,5.42143e-25,9.27954e-23,-1.25301e-23,-4.56629e-22,1.81348e-24,
        0.000648286,2.95153e-05,0.000568683,-0.000271828,-1.26225e-07,6.52424e-09,2.0949e-15,1.04457e-23,1.78792e-21,-2.41421e-22,-8.79803e-21,3.49411e-23,
        -0.000309878,-1.41082e-05,-0.000271828,0.000153017,7.10549e-08,-3.67263e-09,-1.17926e-15,-5.88008e-24,-1.00646e-21,1.35901e-22,4.95259e-21,-1.9669e-23,
        -1.43894e-07,-6.55126e-09,-1.26225e-07,7.10549e-08,8.99025e-08,-4.64681e-09,-1.49207e-15,-7.43979e-24,-1.27343e-21,1.71949e-22,6.26629e-21,-2.48863e-23,
        7.43749e-09,3.38616e-10,6.52424e-09,-3.67263e-09,-4.64681e-09,1.49467e-08,4.7993e-15,2.39305e-23,4.09604e-21,-5.53084e-22,-2.01558e-20,8.00482e-23,
        2.38814e-15,1.08728e-16,2.0949e-15,-1.17926e-15,-1.49207e-15,4.7993e-15,2.12268e-14,1.05842e-22,1.81163e-20,-2.44623e-21,-8.91469e-20,3.54044e-22,
        1.19078e-23,5.42143e-25,1.04457e-23,-5.88008e-24,-7.43979e-24,2.39305e-23,1.05842e-22,3.31388e-24,5.67217e-22,-7.65907e-23,-2.79117e-21,1.1085e-23,
        2.03819e-21,9.27954e-23,1.78792e-21,-1.00646e-21,-1.27343e-21,4.09604e-21,1.81163e-20,5.67217e-22,9.7746e-20,-1.31985e-20,-4.80989e-19,1.91023e-21,
        -2.75215e-22,-1.25301e-23,-2.41421e-22,1.35901e-22,1.71949e-22,-5.53084e-22,-2.44623e-21,-7.65907e-23,-1.31985e-20,7.25098e-21,2.64245e-19,-1.04944e-21,
        -1.00296e-20,-4.56629e-22,-8.79803e-21,4.95259e-21,6.26629e-21,-2.01558e-20,-8.91469e-20,-2.79117e-21,-4.80989e-19,2.64245e-19,2.16684e-17,-8.60553e-20,
        3.9832e-23,1.81348e-24,3.49411e-23,-1.9669e-23,-2.48863e-23,8.00482e-23,3.54044e-22,1.1085e-23,1.91023e-21,-1.04944e-21,-8.60553e-20,3.50978e-19;

        weights[2] <<5.94141e-05,7.55936e-06,-1.82467e-05,1.83564e-07,3.15587e-06,-3.71249e-08,-4.43854e-12,-4.78211e-14,8.99555e-14,-2.25078e-16,-5.3686e-15,1.32688e-17,
        7.55936e-06,2.66315e-06,-6.42829e-06,6.46694e-08,1.11181e-06,-1.3079e-08,-1.56369e-12,-1.68473e-14,3.16912e-14,-7.92944e-17,-1.89135e-15,4.67456e-18,
        -1.82467e-05,-6.42829e-06,0.000147262,-1.48148e-06,-2.54698e-05,2.9962e-07,3.58217e-11,3.85945e-13,-7.25996e-13,1.81651e-15,4.33278e-14,-1.07087e-16,
        1.83564e-07,6.46694e-08,-1.48148e-06,2.01691e-06,3.4675e-05,-4.07908e-07,-4.87682e-11,-5.25432e-13,9.88382e-13,-2.47303e-15,-5.89872e-14,1.4579e-16,
        3.15587e-06,1.11181e-06,-2.54698e-05,3.4675e-05,0.0179455,-0.000211107,-2.52393e-08,-2.7193e-10,5.11523e-10,-1.27988e-12,-3.0528e-11,7.54515e-14,
        -3.71249e-08,-1.3079e-08,2.9962e-07,-4.07908e-07,-0.000211107,7.81397e-05,9.34215e-09,1.00653e-10,-1.89337e-10,4.73739e-13,1.12997e-11,-2.79278e-14,
        -4.43854e-12,-1.56369e-12,3.58217e-11,-4.87682e-11,-2.52393e-08,9.34215e-09,3.24063e-10,3.49148e-12,-6.56776e-12,1.64332e-14,3.91968e-13,-9.68769e-16,
        -4.78211e-14,-1.68473e-14,3.85945e-13,-5.25432e-13,-2.7193e-10,1.00653e-10,3.49148e-12,5.31803e-13,-1.00037e-12,2.50301e-15,5.97024e-14,-1.47558e-16,
        8.99555e-14,3.16912e-14,-7.25996e-13,9.88382e-13,5.11523e-10,-1.89337e-10,-6.56776e-12,-1.00037e-12,1.68205e-09,-4.20865e-12,-1.00386e-10,2.48108e-13,
        -2.25078e-16,-7.92944e-17,1.81651e-15,-2.47303e-15,-1.27988e-12,4.73739e-13,1.64332e-14,2.50301e-15,-4.20865e-12,2.77613e-13,6.62167e-12,-1.63658e-14,
        -5.3686e-15,-1.89135e-15,4.33278e-14,-5.89872e-14,-3.0528e-11,1.12997e-11,3.91968e-13,5.97024e-14,-1.00386e-10,6.62167e-12,2.52701e-07,-6.24564e-10,
        1.32688e-17,4.67456e-18,-1.07087e-16,1.4579e-16,7.54515e-14,-2.79278e-14,-9.68769e-16,-1.47558e-16,2.48108e-13,-1.63658e-14,-6.24564e-10,2.63387e-10;

        weights[3] <<7.67999e-05,1.94203e-05,-8.63354e-05,-6.6116e-07,-1.55986e-05,9.53457e-08,1.78562e-11,3.51044e-13,-6.96791e-12,-2.12467e-17,-1.36287e-15,1.98103e-18,
        1.94203e-05,1.21296e-05,-5.39238e-05,-4.12951e-07,-9.74262e-06,5.95515e-08,1.11527e-11,2.19257e-13,-4.35205e-12,-1.32703e-17,-8.51226e-16,1.23732e-18,
        -8.63354e-05,-5.39238e-05,0.000565108,4.32762e-06,0.0001021,-6.24084e-07,-1.16878e-10,-2.29775e-12,4.56084e-11,1.3907e-16,8.92063e-15,-1.29668e-17,
        -6.6116e-07,-4.12951e-07,4.32762e-06,1.98264e-06,4.67758e-05,-2.85916e-07,-5.35459e-11,-1.05269e-12,2.08949e-11,6.37129e-17,4.08687e-15,-5.94057e-18,
        -1.55986e-05,-9.74262e-06,0.0001021,4.67758e-05,0.044635,-0.00027283,-5.10953e-08,-1.00451e-09,1.99386e-08,6.0797e-14,3.89982e-12,-5.66869e-15,
        9.53457e-08,5.95515e-08,-6.24084e-07,-2.85916e-07,-0.00027283,7.66481e-05,1.43545e-08,2.82203e-10,-5.60148e-09,-1.70801e-14,-1.0956e-12,1.59254e-15,
        1.78562e-11,1.11527e-11,-1.16878e-10,-5.35459e-11,-5.10953e-08,1.43545e-08,3.2431e-10,6.37577e-12,-1.26553e-10,-3.85888e-16,-2.47528e-14,3.59801e-17,
        3.51044e-13,2.19257e-13,-2.29775e-12,-1.05269e-12,-1.00451e-09,2.82203e-10,6.37577e-12,2.74092e-12,-5.44049e-11,-1.65892e-16,-1.06412e-14,1.54677e-17,
        -6.96791e-12,-4.35205e-12,4.56084e-11,2.08949e-11,1.99386e-08,-5.60148e-09,-1.26553e-10,-5.44049e-11,1.13278e-08,3.45408e-14,2.21562e-12,-3.22057e-15,
        -2.12467e-17,-1.32703e-17,1.3907e-16,6.37129e-17,6.0797e-14,-1.70801e-14,-3.85888e-16,-1.65892e-16,3.45408e-14,1.44608e-13,9.27591e-12,-1.34832e-14,
        -1.36287e-15,-8.51226e-16,8.92063e-15,4.08687e-15,3.89982e-12,-1.0956e-12,-2.47528e-14,-1.06412e-14,2.21562e-12,9.27591e-12,7.23809e-07,-1.05211e-09,
        1.98103e-18,1.23732e-18,-1.29668e-17,-5.94057e-18,-5.66869e-15,1.59254e-15,3.59801e-17,1.54677e-17,-3.22057e-15,-1.34832e-14,-1.05211e-09,2.55676e-10;

        weights[4] <<7.66852e-05,1.86349e-05,-7.62738e-05,-3.65581e-07,-8.61871e-06,4.42324e-08,8.21345e-12,1.65542e-13,-4.357e-12,-2.30369e-16,-1.82569e-14,2.38984e-17,
        1.86349e-05,2.36298e-05,-9.67182e-05,-4.63572e-07,-1.09289e-05,5.60885e-08,1.0415e-11,2.09914e-13,-5.52485e-12,-2.92117e-16,-2.31504e-14,3.03041e-17,
        -7.62738e-05,-9.67182e-05,0.0013321,6.38478e-06,0.000150524,-7.72508e-07,-1.43446e-10,-2.89116e-12,7.60939e-11,4.02333e-15,3.18852e-13,-4.17379e-16,
        -3.65581e-07,-4.63572e-07,6.38478e-06,1.97057e-06,4.6457e-05,-2.38424e-07,-4.42725e-11,-8.92315e-13,2.34853e-11,1.24174e-15,9.84091e-14,-1.28818e-16,
        -8.61871e-06,-1.09289e-05,0.000150524,4.6457e-05,0.0530432,-0.000272225,-5.0549e-08,-1.01882e-09,2.68148e-08,1.41779e-12,1.1236e-10,-1.47081e-13,
        4.42324e-08,5.60885e-08,-7.72508e-07,-2.38424e-07,-0.000272225,7.61801e-05,1.41458e-08,2.85109e-10,-7.50393e-09,-3.96757e-13,-3.14432e-11,4.11594e-14,
        8.21345e-12,1.0415e-11,-1.43446e-10,-4.42725e-11,-5.0549e-08,1.41458e-08,3.35751e-10,6.76708e-12,-1.78106e-10,-9.41706e-15,-7.46308e-13,9.76923e-16,
        1.65542e-13,2.09914e-13,-2.89116e-12,-8.92315e-13,-1.01882e-09,2.85109e-10,6.76708e-12,8.16929e-12,-2.15012e-10,-1.13684e-14,-9.00952e-13,1.17935e-15,
        -4.357e-12,-5.52485e-12,7.60939e-11,2.34853e-11,2.68148e-08,-7.50393e-09,-1.78106e-10,-2.15012e-10,5.32397e-08,2.81496e-12,2.23087e-10,-2.92023e-13,
        -2.30369e-16,-2.92117e-16,4.02333e-15,1.24174e-15,1.41779e-12,-3.96757e-13,-9.41706e-15,-1.13684e-14,2.81496e-12,1.24934e-13,9.90111e-12,-1.29606e-14,
        -1.82569e-14,-2.31504e-14,3.18852e-13,9.84091e-14,1.1236e-10,-3.14432e-11,-7.46308e-13,-9.00952e-13,2.23087e-10,9.90111e-12,8.77e-07,-1.148e-09,
        2.38984e-17,3.03041e-17,-4.17379e-16,-1.28818e-16,-1.47081e-13,4.11594e-14,9.76923e-16,1.17935e-15,-2.92023e-13,-1.29606e-14,-1.148e-09,2.54424e-10;



        /* Instantiate seedk aka global costs */
        struct K seed;
        seed.k = VectorXd::Zero(Npars); 
        //seed.k = testVec;
        for (int i = 0; i < Npars; i++) { 
            seed.k(i) = unifDist(gen);
        }
        seed.k(4) = tru.k(4);
        seed.k << 0.1414,	0.1,	0.9497,	0.165757,	0.05,	0.183848;
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

        cout << "seedk:"<< seed.k.transpose() << "| cost:" << costSeedK << endl;
        
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
                    POSMAT(particle, 4) = 0.05;
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
                    pos.k(4) = 0.05;
                    // let's fix theta 4
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

