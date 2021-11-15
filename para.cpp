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
    VectorXd adaptive = VectorXd::Zero(3); // vector of targeted rate constants
    adaptive << 1,3,4;
    int ncomp = posK.size();
    if(unifDist(generator) < 0.67){
        for (int smart = 0; smart < adaptive.size(); smart++) {
        // int px = wcomp(smart);
            int px = adaptive(smart);
            double pos = rPoint(px);
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

            rPoint(px) = (x / (x + y)); 
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
    // VectorXd rPoint;
    // rPoint = posK;
    // std::random_device rand_dev;
    // std::mt19937 generator(rand_dev());
    // vector<int> rand;
    // uniform_real_distribution<double> unifDist(0.0, 1.0);
    // for (int i = 0; i < posK.size(); i++) {
    //     rand.push_back(i);
    // }
    // shuffle(rand.begin(), rand.end(), generator); // shuffle indices as well as possible. 
    // int ncomp = rand.at(0);
    // VectorXd wcomp(ncomp);
    // shuffle(rand.begin(), rand.end(), generator);
    // for (int i = 0; i < ncomp; i++) {
    //     wcomp(i) = rand.at(i);
    // }
    
    // for (int smart = 0; smart < ncomp; smart++) {
    //     int px = wcomp(smart);
    //     double pos = rPoint(px);
    //     if (pos > 1.0 - nan) {
    //         cout << "overflow!" << endl;
    //         pos -= epsi;
    //     }else if (pos < nan) {
    //         cout << "underflow!"<< pos << endl;
    //         pos += epsi;
    //         cout << "pos" << posK.transpose() << endl; 
    //     }
    //     double alpha = hone * pos; // Component specific
    //     double beta = hone - alpha; // pos specific
    // // cout << "alpha:" << alpha << "beta:" << beta << endl;
    //     std::gamma_distribution<double> aDist(alpha, 1); // beta distribution consisting of gamma distributions
    //     std::gamma_distribution<double> bDist(beta, 1);

    //     double x = aDist(generator);
    //     double y = bDist(generator);

    //     rPoint(px) = (x / (x + y)); 
    // }
    
    // return rPoint;
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
    // VectorXd covariances(nMoments - 1);
    
    // for(int i = 0; i < nMoments - 1; i++){
    //     int j = i + 1;
    //     covariances(i) = ( (aDiff.col(i).array() - aDiff.col(i).array().mean()).array() * (aDiff.col(j).array() - aDiff.col(j).array().mean()).array() ).sum() / ((double) aDiff.col(i).array().size() - 1);
    // }

    MatrixXd wt = MatrixXd::Zero(nMoments, nMoments);
   
    // for(int i = 0; i < nMoments; i++){
    //     wt(i,i) = variances(i); // cleanup code and make it more vectorized later.
    // }
    // for(int i = 0; i < nMoments - 1; i++){
    //     int j = i + 1;
    //     wt(i,j) = covariances(i);
    //     wt(j,i) = covariances(i);
    // }
    
    // cout << "Weights Before Inversion:" << endl << wt << endl;
    // wt = wt.llt().solve(MatrixXd::Identity(nMoments, nMoments));
    for(int i = 0; i < nMoments; i++){
        wt(i,i) = 1 / variances(i); // cleanup code and make it more vectorized later.
    }
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
    double alpha = 0.2;
    int nRuns = 1;
    int N = 1250;
    int nParts = 1; // blind PSO
    int nSteps = 1;
    int nParts2 = 10; // targeted PSO
    int nSteps2 = 10;
    int nMoments = (N_SPECIES * (N_SPECIES + 3)) / 2; // var + mean + cov
    int hone = 24;
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
    // if(useOnlySecMom){
    //     for(int i = 0; i < nTimeSteps; i++){
    //         for(int j = 2*N_SPECIES; j < nMoments; j++){
    //             weights[i](j,j) = 0;
    //         }
    //     }
    // }

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
    // tru.k /= (9.69);
    // tru.k(1) += 0.05;
    // tru.k(4) += 0.05; // make sure not so close to the boundary
    // tru.k <<  0.51599600,  0.06031990, 0.10319900, 0.89680100, 0.05516000, 0.00722394; // Bill k

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
        for(int i = 0; i < nTimeSteps; i++){
            weights[i] = MatrixXd::Identity(nMoments, nMoments);
        }

        /* Instantiate seedk aka global costs */
        struct K seed;
        seed.k = VectorXd::Zero(Npars); 
        //seed.k = testVec;
        for (int i = 0; i < Npars; i++) { 
            seed.k(i) = unifDist(gen);
        }
        seed.k << 0.098598,	0.100714,	0.950519,	0.155835,	0.034803,	0.178066;
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
                        POSMAT(particle, i) = pUnifDist(pGenerator);//tru.k(i) + alpha * (0.5 - unifDist(pGenerator));
                        // if(i > 1){
                        //     POSMAT(particle, i) = tru.k(i);
                        // }
                    }

                    POSMAT.row(particle) = seed.k;
                    // POSMAT.row(particle) = tru.k;
                    // POSMAT.row(particle) << 0.270536,	0.981999,	0.988012,	0.201166,	0.078759,	0.206342;
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
        POSMAT.conservativeResize(nParts2, Npars); // resize matrices to fit targetted PSO
        PBMAT.conservativeResize(nParts2, Npars + 1);

        weights[0] << 0.0849656,0.00030881,0.00259034,-0.00125985,-5.29795e-07,2.36451e-08,-2.71955e-11,-1.35961e-22,-2.73585e-20,5.29269e-21,1.92087e-19,-3.13899e-22,
        0.00030881,3.78127e-06,3.17178e-05,-1.54264e-05,-6.48716e-09,2.89527e-10,-3.32999e-13,-1.6648e-24,-3.34996e-22,6.48073e-23,2.35204e-21,-3.84359e-24,
        0.00259034,3.17178e-05,0.000578508,-0.000281365,-1.18321e-07,5.28074e-09,-6.07365e-12,-3.03646e-23,-6.11007e-21,1.18203e-21,4.28994e-20,-7.0104e-23,
        -0.00125985,-1.54264e-05,-0.000281365,0.000159121,6.6914e-08,-2.98642e-09,3.43484e-12,1.71721e-23,3.45543e-21,-6.68477e-22,-2.4261e-20,3.9646e-23,
        -5.29795e-07,-6.48716e-09,-1.18321e-07,6.6914e-08,7.57474e-08,-3.38066e-09,3.88827e-12,1.9439e-23,3.91159e-21,-7.56723e-22,-2.74637e-20,4.48797e-23,
        2.36451e-08,2.89527e-10,5.28074e-09,-2.98642e-09,-3.38066e-09,1.49473e-08,-1.71917e-11,-8.5948e-23,-1.72947e-20,3.34578e-21,1.21428e-19,-1.98432e-22,
        -2.71955e-11,-3.32999e-13,-6.07365e-12,3.43484e-12,3.88827e-12,-1.71917e-11,5.68458e-10,2.84195e-21,5.71867e-19,-1.10631e-19,-4.01514e-18,6.56133e-21,
        -1.35961e-22,-1.6648e-24,-3.03646e-23,1.71721e-23,1.9439e-23,-8.5948e-23,2.84195e-21,2.44564e-24,4.92121e-22,-9.5204e-23,-3.45523e-21,5.64636e-24,
        -2.73585e-20,-3.34996e-22,-6.11007e-21,3.45543e-21,3.91159e-21,-1.72947e-20,5.71867e-19,4.92121e-22,9.97969e-20,-1.93064e-20,-7.00685e-19,1.14502e-21,
        5.29269e-21,6.48073e-23,1.18203e-21,-6.68477e-22,-7.56723e-22,3.34578e-21,-1.10631e-19,-9.5204e-23,-1.93064e-20,1.07221e-20,3.89138e-19,-6.35908e-22,
        1.92087e-19,2.35204e-21,4.28994e-20,-2.4261e-20,-2.74637e-20,1.21428e-19,-4.01514e-18,-3.45523e-21,-7.00685e-19,3.89138e-19,2.22393e-17,-3.63424e-20,
        -3.13899e-22,-3.84359e-24,-7.0104e-23,3.9646e-23,4.48797e-23,-1.98432e-22,6.56133e-21,5.64636e-24,1.14502e-21,-6.35908e-22,-3.63424e-20,3.61994e-19;

        weights[1] << 5.85597e-05,7.55787e-06,-2.04086e-05,1.19984e-07,1.75691e-06,-2.06913e-08,-3.00832e-12,-3.29187e-14,1.1358e-13,-2.14347e-16,-1.77549e-14,4.6905e-17,
        7.55787e-06,2.607e-06,-7.03971e-06,4.13872e-08,6.06025e-07,-7.1372e-09,-1.03768e-12,-1.13549e-14,3.91782e-14,-7.39363e-17,-6.12436e-15,1.61793e-17,
        -2.04086e-05,-7.03971e-06,0.00014658,-8.61757e-07,-1.26185e-05,1.4861e-07,2.16065e-11,2.3643e-13,-8.15761e-13,1.53949e-15,1.2752e-13,-3.36883e-16,
        1.19984e-07,4.13872e-08,-8.61757e-07,1.9014e-06,2.78419e-05,-3.27896e-07,-4.76731e-11,-5.21666e-13,1.79992e-12,-3.39677e-15,-2.81364e-13,7.43309e-16,
        1.75691e-06,6.06025e-07,-1.26185e-05,2.78419e-05,0.0127413,-0.000150055,-2.18166e-08,-2.3873e-10,8.23695e-10,-1.55446e-12,-1.28761e-10,3.4016e-13,
        -2.06913e-08,-7.1372e-09,1.4861e-07,-3.27896e-07,-0.000150055,7.85291e-05,1.14174e-08,1.24936e-10,-4.31069e-10,8.13505e-13,6.7385e-11,-1.78018e-13,
        -3.00832e-12,-1.03768e-12,2.16065e-11,-4.76731e-11,-2.18166e-08,1.14174e-08,3.20613e-10,3.50833e-12,-1.21049e-11,2.28441e-14,1.89224e-12,-4.99893e-15,
        -3.29187e-14,-1.13549e-14,2.3643e-13,-5.21666e-13,-2.3873e-10,1.24936e-10,3.50833e-12,5.07179e-13,-1.74993e-12,3.30244e-15,2.73551e-13,-7.22667e-16,
        1.1358e-13,3.91782e-14,-8.15761e-13,1.79992e-12,8.23695e-10,-4.31069e-10,-1.21049e-11,-1.74993e-12,1.62506e-09,-3.06678e-12,-2.54031e-10,6.71098e-13,
        -2.14347e-16,-7.39363e-17,1.53949e-15,-3.39677e-15,-1.55446e-12,8.13505e-13,2.28441e-14,3.30244e-15,-3.06678e-12,2.56635e-13,2.12579e-11,-5.61591e-14,
        -1.77549e-14,-6.12436e-15,1.2752e-13,-2.81364e-13,-1.28761e-10,6.7385e-11,1.89224e-12,2.73551e-13,-2.54031e-10,2.12579e-11,4.72203e-07,-1.24747e-09,
        4.6905e-17,1.61793e-17,-3.36883e-16,7.43309e-16,3.4016e-13,-1.78018e-13,-4.99893e-15,-7.22667e-16,6.71098e-13,-5.61591e-14,-1.24747e-09,2.82603e-10;

        weights[2] << 8.08179e-05,2.18336e-05,-9.74166e-05,-5.01008e-07,-1.10249e-05,7.43191e-08,1.55166e-11,3.92075e-13,-7.72855e-12,7.72232e-16,1.86221e-13,-2.32887e-16,
        2.18336e-05,1.2975e-05,-5.78915e-05,-2.97733e-07,-6.55173e-06,4.41654e-08,9.22099e-12,2.32997e-13,-4.59283e-12,4.58912e-16,1.10665e-13,-1.38397e-16,
        -9.74166e-05,-5.78915e-05,0.000555521,2.85701e-06,6.28697e-05,-4.23807e-07,-8.84836e-11,-2.23582e-12,4.40723e-11,-4.40368e-15,-1.06193e-12,1.32804e-15,
        -5.01008e-07,-2.97733e-07,2.85701e-06,1.82871e-06,4.02414e-05,-2.71269e-07,-5.66363e-11,-1.4311e-12,2.82096e-11,-2.81869e-15,-6.79718e-13,8.50051e-16,
        -1.10249e-05,-6.55173e-06,6.28697e-05,4.02414e-05,0.034622,-0.000233388,-4.87274e-08,-1.23125e-09,2.42703e-08,-2.42508e-12,-5.84799e-10,7.31346e-13,
        7.43191e-08,4.41654e-08,-4.23807e-07,-2.71269e-07,-0.000233388,7.76757e-05,1.62174e-08,4.09783e-10,-8.07761e-09,8.0711e-13,1.94632e-10,-2.43405e-13,
        1.55166e-11,9.22099e-12,-8.84836e-11,-5.66363e-11,-4.87274e-08,1.62174e-08,3.24545e-10,8.20066e-12,-1.61651e-10,1.6152e-14,3.89501e-12,-4.87107e-15,
        3.92075e-13,2.32997e-13,-2.23582e-12,-1.4311e-12,-1.23125e-09,4.09783e-10,8.20066e-12,2.97249e-12,-5.85935e-11,5.85463e-15,1.41182e-12,-1.76562e-15,
        -7.72855e-12,-4.59283e-12,4.40723e-11,2.82096e-11,2.42703e-08,-8.07761e-09,-1.61651e-10,-5.85935e-11,1.04699e-08,-1.04614e-12,-2.52274e-10,3.15492e-13,
        7.72232e-16,4.58912e-16,-4.40368e-15,-2.81869e-15,-2.42508e-12,8.0711e-13,1.6152e-14,5.85463e-15,-1.04614e-12,1.34104e-13,3.23388e-11,-4.04426e-14,
        1.86221e-13,1.10665e-13,-1.06193e-12,-6.79718e-13,-5.84799e-10,1.94632e-10,3.89501e-12,1.41182e-12,-2.52274e-10,3.23388e-11,2.69139e-06,-3.36583e-09,
        -2.32887e-16,-1.38397e-16,1.32804e-15,8.50051e-16,7.31346e-13,-2.43405e-13,-4.87107e-15,-1.76562e-15,3.15492e-13,-4.04426e-14,-3.36583e-09,2.75599e-10;

        weights[3] <<7.90803e-05,2.13992e-05,-9.06126e-05,-3.19084e-07,-7.08527e-06,4.09256e-08,8.21219e-12,2.76542e-13,-7.44298e-12,-5.29405e-17,-1.48428e-14,1.54322e-17,
        2.13992e-05,2.34067e-05,-9.91131e-05,-3.49018e-07,-7.74994e-06,4.47648e-08,8.98259e-12,3.02485e-13,-8.14122e-12,-5.79069e-17,-1.62352e-14,1.68799e-17,
        -9.06126e-05,-9.91131e-05,0.00134513,4.73674e-06,0.000105179,-6.07532e-07,-1.21908e-10,-4.10521e-12,1.1049e-10,7.85891e-16,2.20338e-13,-2.29088e-16,
        -3.19084e-07,-3.49018e-07,4.73674e-06,1.79515e-06,3.98613e-05,-2.30245e-07,-4.62013e-11,-1.55581e-12,4.18738e-11,2.9784e-16,8.35046e-14,-8.68206e-17,
        -7.08527e-06,-7.74994e-06,0.000105179,3.98613e-05,0.0425862,-0.000245984,-4.93596e-08,-1.66216e-09,4.47363e-08,3.18201e-13,8.92131e-11,-9.27557e-14,
        4.09256e-08,4.47648e-08,-6.07532e-07,-2.30245e-07,-0.000245984,7.72621e-05,1.55035e-08,5.22075e-10,-1.40514e-08,-9.99448e-14,-2.80213e-11,2.9134e-14,
        8.21219e-12,8.98259e-12,-1.21908e-10,-4.62013e-11,-4.93596e-08,1.55035e-08,3.35503e-10,1.12979e-11,-3.04078e-10,-2.16285e-15,-6.06392e-13,6.30471e-16,
        2.76542e-13,3.02485e-13,-4.10521e-12,-1.55581e-12,-1.66216e-09,5.22075e-10,1.12979e-11,9.37209e-12,-2.52245e-10,-1.79417e-15,-5.03027e-13,5.23002e-16,
        -7.44298e-12,-8.14122e-12,1.1049e-10,4.18738e-11,4.47363e-08,-1.40514e-08,-3.04078e-10,-2.52245e-10,5.4283e-08,3.86105e-13,1.08251e-10,-1.1255e-13,
        -5.29405e-17,-5.79069e-17,7.85891e-16,2.9784e-16,3.18201e-13,-9.99448e-14,-2.16285e-15,-1.79417e-15,3.86105e-13,1.1099e-13,3.11179e-11,-3.23535e-14,
        -1.48428e-14,-1.62352e-14,2.20338e-13,8.35046e-14,8.92131e-11,-2.80213e-11,-6.06392e-13,-5.03027e-13,1.08251e-10,3.11179e-11,3.89104e-06,-4.04555e-09,
        1.54322e-17,1.68799e-17,-2.29088e-16,-8.68206e-17,-9.27557e-14,2.9134e-14,6.30471e-16,5.23002e-16,-1.1255e-13,-3.23535e-14,-4.04555e-09,2.74214e-10;

         weights[4] <<7.49654e-05,1.93747e-05,-6.03444e-05,-1.62145e-07,-3.5346e-06,1.94493e-08,3.69769e-12,1.30161e-13,-3.61294e-12,-8.02191e-17,-2.30487e-14,2.30203e-17,
        1.93747e-05,3.69692e-05,-0.000115144,-3.09392e-07,-6.74442e-06,3.71115e-08,7.05563e-12,2.48363e-13,-6.89391e-12,-1.53067e-16,-4.39796e-14,4.39254e-17,
        -6.03444e-05,-0.000115144,0.00189805,5.10005e-06,0.000111176,-6.1175e-07,-1.16306e-10,-4.09404e-12,1.1364e-10,2.52318e-15,7.24964e-13,-7.24071e-16,
        -1.62145e-07,-3.09392e-07,5.10005e-06,1.81353e-06,3.95331e-05,-2.17533e-07,-4.13573e-11,-1.4558e-12,4.04093e-11,8.9722e-16,2.57791e-13,-2.57473e-16,
        -3.5346e-06,-6.74442e-06,0.000111176,3.95331e-05,0.0444089,-0.000244362,-4.64581e-08,-1.63535e-09,4.53932e-08,1.00788e-12,2.89585e-10,-2.89228e-13,
        1.94493e-08,3.71115e-08,-6.1175e-07,-2.17533e-07,-0.000244362,7.70738e-05,1.46532e-08,5.15804e-10,-1.43174e-08,-3.17893e-13,-9.13375e-11,9.12249e-14,
        3.69769e-12,7.05563e-12,-1.16306e-10,-4.13573e-11,-4.64581e-08,1.46532e-08,3.3234e-10,1.16986e-11,-3.24723e-10,-7.20992e-15,-2.07156e-12,2.06901e-15,
        1.30161e-13,2.48363e-13,-4.09404e-12,-1.4558e-12,-1.63535e-09,5.15804e-10,1.16986e-11,1.8074e-11,-5.01687e-10,-1.11391e-14,-3.2005e-12,3.19656e-15,
        -3.61294e-12,-6.89391e-12,1.1364e-10,4.04093e-11,4.53932e-08,-1.43174e-08,-3.24723e-10,-5.01687e-10,1.30793e-07,2.90404e-12,8.34393e-10,-8.33365e-13,
        -8.02191e-17,-1.53067e-16,2.52318e-15,8.9722e-16,1.00788e-12,-3.17893e-13,-7.20992e-15,-1.11391e-14,2.90404e-12,1.07916e-13,3.10066e-11,-3.09683e-14,
        -2.30487e-14,-4.39796e-14,7.24964e-13,2.57791e-13,2.89585e-10,-9.13375e-11,-2.07156e-12,-3.2005e-12,8.34393e-10,3.10066e-11,4.16991e-06,-4.16477e-09,
        2.30203e-17,4.39254e-17,-7.24071e-16,-2.57473e-16,-2.89228e-13,9.12249e-14,2.06901e-15,3.19656e-15,-8.33365e-13,-3.09683e-14,-4.16477e-09,2.7381e-10;

        cout << "targeted PSO has started!" << endl; 
        sfp = 3.0, sfg = 1.0, sfe = 6.0; // initial particle historical weight, global weight social, inertial
        sfi = sfe, sfc = sfp, sfs = sfg; // below are the variables being used to reiterate weights
        double nearby = sdbeta;
        VectorXd chkpts = wmatup * nSteps2;
        int chkptNo = 0;
        for(int step = 0; step < nSteps2; step++){
            if(step == 0 || step == chkpts(chkptNo)){ /* update wt   matrix || step == chkpts(0) || step == chkpts(1) || step == chkpts(2) || step == chkpts(3) */
                cout << "Updating Weight Matrix!" << endl;
                cout << "GBVEC AND COST:" << GBMAT.row(GBMAT.rows() - 1) << endl;
                nearby = squeeze * nearby;
                /* reinstantiate gCost */
                struct K gPos;
                // GBVEC << 0.648691,	0.099861,	0.0993075,	0.8542755,	0.049949,	0.0705955;
                gPos.k = GBVEC;
                
                double cost = 0;
                for(int t = 0; t < nTimeSteps; t++){
                    Protein_Components gXt(times(t), nMoments, N);
                    Moments_Mat_Obs gXtObs(gXt);
                    Nonlinear_ODE6 gSys(gPos);
                    for (int i = 0; i < N; i++) {
                        //State_N c0 = gen_multi_norm_iSub();
                        State_N c0 = convertInit(X_0, i);
                        gXt.index = i;
                        integrate_adaptive(controlledStepper, gSys, c0, t0, times(t), dt, gXtObs);
                    }
                    gXt.mVec /= N;  
                    weights[t] = customWtMat(Yt3Mats[t], gXt.mat, nMoments, N);
                    // if(useOnlySecMom){
                    //     for(int j = 2*N_SPECIES; j < nMoments; j++){
                    //         weights[t](j,j) = 0;
                    //     }
                    // }
                    cost += calculate_cf2(Yt3Vecs[t], gXt.mVec, weights[t]);
                }
                gCost = cost;
                hone += 4;
                GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1);
                for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = gPos.k(i);}
                GBMAT(GBMAT.rows() - 1, Npars) = gCost;
                if(step > 0 && chkptNo < nRestarts - 1){
                    chkptNo++;
                }
            }
        #pragma omp parallel for 
            for(int particle = 0; particle < nParts2; particle++){
                random_device pRanDev;
                mt19937 pGenerator(pRanDev());
                uniform_real_distribution<double> pUnifDist(uniLowBound, uniHiBound);
            
                if(step == 0 || step == chkpts(chkptNo)){
                    /* reinitialize particles around global best */
                    for(int edim = 0; edim < Npars; edim++){
                        int wasflipped = 0;
                        double tmean = GBVEC(edim);
                        if (GBVEC(edim) > 0.5) {
                            tmean = 1 - GBVEC(edim);
                            wasflipped = 1;
                        }
                        double myc = (1 - tmean) / tmean;
                        double alpha = myc / ((1 + myc) * (1 + myc) * (1 + myc)*nearby*nearby);
                        double beta = myc * alpha;

                        if(alpha < nan){
                            alpha = epsi;
                        }
                        if(beta < nan){
                            beta = epsi;
                        }

                        std::gamma_distribution<double> aDist(alpha, 1);
                        std::gamma_distribution<double> bDist(beta, 1);

                        double x = aDist(pGenerator);
                        double y = bDist(pGenerator);
                        double myg = x / (x + y);

                        if(myg >= 1){
                            myg = myg - epsi;
                        }
                        if(myg <= 0){
                            myg = myg + epsi;
                        }

                        if (wasflipped == 1) {
                            wasflipped = 0;
                            myg = 1 - myg;
                        }
                        POSMAT(particle, edim) = myg;
                    }

                    /* Write new POSMAT into Ks to be passed into system */
                    struct K pos;
                    pos.k = VectorXd::Zero(Npars);
                    for(int i = 0; i < Npars; i++){
                        pos.k(i) = POSMAT(particle, i);
                    }
                    //VectorXd XtPSO3 = VectorXd::Zero(nMoments);
                    double cost = 0;
                    for(int t = 0; t < nTimeSteps; t++){
                        Nonlinear_ODE6 initSys(pos);
                        Protein_Components XtPSO(times(t), nMoments, N);
                        Moments_Mat_Obs XtObsPSO(XtPSO);
                        for(int i = 0; i < N; i++){
                            State_N c0 = convertInit(X_0, i);
                            XtPSO.index = i;
                            integrate_adaptive(controlledStepper, initSys, c0, t0, times(t), dt, XtObsPSO);
                        }
                        XtPSO.mVec/=N;
                        cost += calculate_cf2(Yt3Vecs[t], XtPSO.mVec, weights[t]);
                    }
                    
                    /* initialize PBMAT */
                    for(int i = 0; i < Npars; i++){
                        PBMAT(particle, i) = POSMAT(particle, i);
                    }
                    PBMAT(particle, Npars) = cost; // add cost to final column
                }else{ 
                    /* using new rate constants, initialize particle best values */
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
                    POSMAT.row(particle) = pos.k; // back into POSMAT
                    
                    double cost = 0;
                    /* solve ODEs with new system and recompute cost */
                    for(int t = 0; t < nTimeSteps; t++){
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
                    
                    /* update pBest and gBest */
                    #pragma omp critical
                    {
                    if(cost < PBMAT(particle, Npars)){ // update particle best 
                        for(int i = 0; i < Npars; i++){
                            PBMAT(particle, i) = pos.k(i);
                        }
                        PBMAT(particle, Npars) = cost;
                        if(cost < gCost){ // update global 
                            gCost = cost;
                            GBVEC = pos.k;
                        }   
                    }
                    }
                }
            }
            GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1); // Add to GBMAT after each step.
            for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = GBVEC(i);}
            GBMAT(GBMAT.rows() - 1, Npars) = gCost;

            sfi = sfi - (sfe - sfg) / nSteps2;   // reduce the inertial weight after each step 
            sfs = sfs + (sfe - sfg) / nSteps2;

            if(step == 0){ // quick plug to see PBMAT
                cout << "New PBMAT:" << endl;
                cout << PBMAT << endl << endl;
            }
            // cout << "current:" << GBVEC.transpose()<<" "<< gCost << endl;
        }
        cout << "GBMAT after targeted PSO:" << endl << GBMAT << endl;
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

