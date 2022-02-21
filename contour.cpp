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

#define N_SPECIES 4
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
        k = VectorXd::Zero(N_DIM);
    }
};

class Nonlinear_ODE6
{
    struct K rate;

public:
    Nonlinear_ODE6(struct K G) : rate(G) {}

    void operator() (const State_N& c, State_N& dcdt, double t)
    {
        // dcdt[0] = -(jay.k(0) * c[0] * c[1])  // Syk
        //     + jay.k(1) * c[2]
        //     + jay.k(2) * c[2];

        // dcdt[1] = -(jay.k(0) * c[0] * c[1]) // Vav
        //     + jay.k(1) * c[2]
        //     + jay.k(5) * c[5];

        // dcdt[2] = jay.k(0) * c[0] * c[1] // Syk-Vav
        //     - jay.k(1) * c[2]
        //     - jay.k(2) * c[2];

        // dcdt[3] = jay.k(2) * c[2] //pVav
        //     - jay.k(3) * c[3] * c[4]
        //     + jay.k(4) * c[5];

        // dcdt[4] = -(jay.k(3) * c[3] * c[4]) // SHP1 
        //     + jay.k(4) * c[5]
        //     + jay.k(5) * c[5];

        // dcdt[5] = jay.k(3) * c[3] * c[4]  // SHP1-pVav
        //     - jay.k(4) * c[5]
        //     - jay.k(5) * c[5];
        dcdt[0] = rate.k(0) - rate.k(5) * c[0];
        dcdt[1] = rate.k(1) * c[0] - rate.k(4) * c[1];
        dcdt[2] = rate.k(2) * c[1] - rate.k(4) * c[2];
        dcdt[3] = rate.k(3) * c[2] - rate.k(4) * c[3];
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
    State_N c0 = {sample(index,0), sample(index,1), sample(index,2), sample(index,3)};
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
MatrixXd wolfWtMat(const MatrixXd& Yt, int nMoments, bool useInverse){
    /* first moment differences */
    MatrixXd fmdiffs = MatrixXd::Zero(Yt.rows(), Yt.cols());
    for(int i = 0; i < Yt.cols(); i++){
        fmdiffs.col(i) = Yt.col(i).array() - Yt.col(i).array().mean();
    }
    /* second moment difference computations - @todo make it variable later */
    MatrixXd smdiffs(Yt.rows(), Yt.cols());
    for(int i = 0; i < Yt.cols(); i++){
        smdiffs.col(i) = (Yt.col(i).array() * Yt.col(i).array()) - (Yt.col(i).array().mean() * Yt.col(i).array().mean());
    }
    /* If no cross moments, then have a check for it */
    int nCross = nMoments - (2 * Yt.cols());
    if (nCross < 0){
        nCross = 0;
    }
    MatrixXd cpDiff(Yt.rows(), nCross);

    /* cross differences */
    if(nCross > 0){
        int upperDiag = 0;
        for(int i = 0; i < Yt.cols(); i++){
            for(int j = i + 1; j < Yt.cols(); j++){
                cpDiff.col(upperDiag) = (Yt.col(i).array() * Yt.col(j).array()) - (Yt.col(i).array().mean() * Yt.col(j).array().mean());
                upperDiag++;
            }
        }
    }

    MatrixXd aDiff(Yt.rows(), nMoments);
    for(int i = 0; i < Yt.rows(); i++){
        for(int moment = 0; moment < nMoments; moment++){
            if(moment < Yt.cols()){
                aDiff(i, moment) = fmdiffs(i, moment);
            }else if (moment >= Yt.cols() && moment < 2 * Yt.cols()){
                aDiff(i, moment) = smdiffs(i, moment - Yt.cols());
            }else if (moment >= 2 * Yt.cols()){
                aDiff(i, moment) = cpDiff(i, moment - (2 * Yt.cols()));
            }
        }
    }
    double cost = 0;
    VectorXd variances(nMoments);
    for(int i = 0; i < nMoments; i++){
        variances(i) = (aDiff.col(i).array() - aDiff.col(i).array().mean()).square().sum() / ((double) aDiff.col(i).array().size() - 1);
    }
  
    MatrixXd wt = MatrixXd::Zero(nMoments, nMoments);
    
    if(useInverse){
         // compute covariances for differences.
        for(int i = 0; i < nMoments; i++){
            wt(i,i) = variances(i); // cleanup code and make it more vectorized later.
        }
        for(int i = 0; i < nMoments; i++){
            for(int j = i + 1; j < nMoments; j++){
                wt(i,j) = ((aDiff.col(i).array() - aDiff.col(i).array().mean()).array() * (aDiff.col(j).array() - aDiff.col(j).array().mean()).array() ).sum() / ((double) aDiff.col(i).array().size() - 1); 
                wt(j,i) = wt(i,j); // across diagonal
            }
        }

        wt = wt.completeOrthogonalDecomposition().solve(MatrixXd::Identity(nMoments, nMoments));

    }else{
        for(int i = 0; i < nMoments; i++){
            wt(i,i) = 1 / variances(i); // cleanup code and make it more vectorized later.
        }
    }
    
    return wt;
}
MatrixXd csvToMatrix (const std::string & path){
    std::ifstream indata;
    indata.open(path);
    if(!indata.is_open()){
        throw std::runtime_error("Invalid Sample File Name!");
        exit(EXIT_FAILURE);
    }
    std::string line;
    std::vector<double> values;
    unsigned int rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    MatrixXd mat = MatrixXd::Zero(rows, values.size()/rows);
    int i = 0;
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < mat.cols(); c++){
            mat(r,c) = values[i];
            i++;
        }
    }
 
    return mat;
}
VectorXd momentVector(const MatrixXd &sample, int nMoments){
    VectorXd moments(nMoments);
    VectorXd mu = sample.colwise().mean();
    VectorXd variances(sample.cols());

    int nVar = sample.cols();// check to make sure if we are using variances to compute means, variances, etc. 
    if(nMoments < sample.cols()){
        nVar = 0;
    }

    // Compute sample variances
    for(int c = 0; c < nVar; c++){
        variances(c) = (sample.col(c).array() - sample.col(c).array().mean()).square().sum() / ((double) sample.col(c).array().size() - 1);
    }

    // again only compute covariances, if number of moments allow for it
    int nCross = nMoments - 2*sample.cols();
    VectorXd covariances(0);
    if(nCross > 0){
        int n = 0;
        covariances.conservativeResize(nCross);
        for (int i = 0; i < sample.cols(); i++) {
            for (int j = i + 1; j < sample.cols(); j++) {
                covariances(n) = ((sample.col(i).array() - sample.col(i).array().mean()) * (sample.col(j).array() - sample.col(j).array().mean())).sum() / ( sample.rows() - 1);
                n++;
            }
        }
    }

    // Now after all computations, add to moment vector
    for(int i = 0; i < nMoments; i++){
        if(i < sample.cols()){
            moments(i) = mu(i);
        }else if (i >= sample.cols() && i < 2 * sample.cols()){
            moments(i) = variances(i - sample.cols());
        }else if (i >= 2 * sample.cols()){
            moments(i) = covariances(i - (2 * sample.cols()));
        }
    }
    return moments;
}
int main() {
    auto t1 = std::chrono::high_resolution_clock::now();
    /*---------------------- Setup ------------------------ */
  
    /* Variables (global) */
    double t0 = 60, tf = 15, dt = 1.0; 
    int nTimeSteps = 1;
    VectorXd times = VectorXd::Zero(nTimeSteps);
    // times << 0.5, 2, 10, 20, 30; // ultra early, early, medium, late
    cout << "loaded in time vals" << endl;
    times << 120;
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
    int nParts = 25; // first part PSO
    int nSteps = 50;
    int nParts2 = 10; // second part PSO
    int nSteps2 = 1000;
    int nMoments = (N_SPECIES * (N_SPECIES + 3)) / 2; // var + mean + cov
    // nMoments = 2 * N_SPECIES; // mean + var
    // nMoments = N_SPECIES;
    int hone = 24;
    //nMoments = 2*N_SPECIES; // mean + var only!
    
    double uniLowBound = 0.0, uniHiBound = 1.0;
    random_device RanDev;
    mt19937 gen(RanDev());
    uniform_real_distribution<double> unifDist(uniLowBound, uniHiBound);
    
    vector<MatrixXd> weights;
  
    // cout << "wt:" << endl << wt << endl;
    cout << "Reading in data!" << endl;

    // X_0_Full = readIntoMatrix(X0File, sizeFile, N_SPECIES);
    // Y_0_Full = readIntoMatrix(Y0File, sizeFile, N_SPECIES);
    // X0File.close();
    // Y0File.close();
    struct K tru;
    tru.k << 0.789183,	0.250346,	0.0915363,	0.969999,	0.243538,	0.0985505;
    // tru.k << 0.73121, 0.210256, 0.0901003, 0.840568, 0.20446, 0.0861549;
    MatrixXd X_0 = csvToMatrix("initial/t1m_processed.csv"); //X_0_Full.block(startRow, 0, N, Npars);
    int N = X_0.rows();
    MatrixXd Y_t = csvToMatrix("initial/t2m_processed.csv");
    for(int i = 0; i < nTimeSteps; i++){
        weights.push_back(wolfWtMat(Y_t, nMoments, false));
    }
    cout << "weights:" << endl;
    cout << weights[0] << endl;
    // Y_0 = Y_0_Full.block(startRow, 0, N, Npars);
    // cout << "Using starting row of data:" << startRow << " and " << N << " data pts!" << endl;
    cout << "first row X0:" << X_0.row(0) << endl;
    cout << "final row X0:" << X_0.row(X_0.rows() - 1) << endl << endl << endl << endl;
    Controlled_RK_Stepper_N controlledStepper;
    vector<VectorXd> Yt3Vecs;
    cout << "Here?" << endl;
    Yt3Vecs.push_back(momentVector(Y_t, nMoments));
    cout << "Using two part PSO " << "Sample Size:" << N << " with:" << nMoments << " moments." << endl;
    cout << "Using Times:" << times.transpose() << endl;
    cout << "Bounds for Uniform Distribution (" << uniLowBound << "," << uniHiBound << ")"<< endl;
    cout << "Blind PSO --> nParts:" << nParts << " Nsteps:" << nSteps << endl;
    cout << "Targeted PSO --> nParts:" <<  nParts2 << " Nsteps:" << nSteps2 << endl;
    cout << "sdbeta:" << sdbeta << endl;
    // tru.k << 5.0, 0.1, 1.0, 8.69, 0.05, 0.70;
    // tru.k /= (9.69);
    // tru.k(1) += 0.05;
    // tru.k(4) += 0.05; // make sure not so close to the boundary
    // tru.k << 0.996673, 0.000434062, 0.0740192,  0.795578,  0.00882025, 0.0317506;
    

    // cout << "using truk:" << tru.k.transpose() << endl;
    // vector<VectorXd> Yt3Vecs;
    // for(int t = 0; t < nTimeSteps; t++){
    //     Nonlinear_ODE6 trueSys(tru);
    //     Protein_Components Yt(times(t), nMoments, N);
    //     Moments_Mat_Obs YtObs(Yt);
    //     for (int i = 0; i < Y_0.rows(); i++) {
    //         //State_N c0 = gen_multi_norm_iSub(); // Y_0 is simulated using norm dist.
    //         State_N c0 = convertInit(Y_0, i);
    //         Yt.index = i;
    //         integrate_adaptive(controlledStepper, trueSys, c0, t0, times(t), dt, YtObs);
    //     }
    //     Yt.mVec /= N;
    //     Yt3Vecs.push_back(Yt.mVec);
    // }
    // struct K seed;
    // seed.k << 0.1659069,	0.6838229,	0.9585955,	0.4651133,	0.4573598,	0.1806655;

    /* Solve for 50 x 50 contour plot for equal weights */
    int xAxis = 2, yAxis = 3; // thetas
    int xDim = 50, yDim = 50;
    double scale = (xDim+yDim) / 2;
    double cost = 0;
    // double holdtheta2 = 0.259;
    MatrixXd eqwts(xDim*yDim, Npars + 1);
    int s = 0;
    cout << "contour rates below! " << endl;
    for(int x = 0; x < xDim; x++){
        for(int y = 0; y < yDim; y++){
            K rate;
            rate.k = tru.k;
            // rate.k(1) = holdtheta2;
            rate.k(xAxis) = x / scale;
            rate.k(yAxis) = y / scale;
            for(int t = 0; t < nTimeSteps; t++){
                Nonlinear_ODE6 sys(rate);
                Protein_Components Xt(times(t), nMoments, N);
                Moments_Mat_Obs XtObs(Xt);
                for (int i = 0; i < X_0.rows(); i++) {
                    State_N x0 = convertInit(X_0, i);
                    Xt.index = i;
                    integrate_adaptive(controlledStepper, sys, x0, t0, times(t), dt, XtObs);
                }
                Xt.mVec /= X_0.rows();
                cost += calculate_cf2(Yt3Vecs[t], Xt.mVec, weights[t]);
            }
            for (int i = 0; i < Npars; i++) {
                eqwts(s, i) = rate.k(i);
            }
            eqwts(s, Npars) = cost;
            
            s++;
            cost = 0;
        }
    }
    printToCsv(eqwts, "eqwts");
    cout << "eqwts" << endl << eqwts; 

    /*Sanity Check */
    cout << endl << "Sanity Check:" << endl;
    K rate;
    rate.k = tru.k;
    cost = 0;
    for(int t = 0; t < nTimeSteps; t++){
        Nonlinear_ODE6 sys(rate);
        Protein_Components Xt(times(t), nMoments, N);
        Moments_Mat_Obs XtObs(Xt);
        for (int i = 0; i < X_0.rows(); i++) {
            State_N x0 = convertInit(X_0, i);
            Xt.index = i;
            integrate_adaptive(controlledStepper, sys, x0, t0, times(t), dt, XtObs);
        }
        Xt.mVec /= N;
        cost += calculate_cf2(Yt3Vecs[t], Xt.mVec, weights[t]);
    }
    cout << "true:" << rate.k.transpose() << " " <<cost << endl;
    cost = 0;
    rate.k << 0.8,	0.24,	0.0915363,	0.969999,	0.243538,	0.0985505;
    for(int t = 0; t < nTimeSteps; t++){
        Nonlinear_ODE6 sys(rate);
        Protein_Components Xt(times(t), nMoments, N);
        Moments_Mat_Obs XtObs(Xt);
        for (int i = 0; i < X_0.rows(); i++) {
            State_N x0 = convertInit(X_0, i);
            Xt.index = i;
            integrate_adaptive(controlledStepper, sys, x0, t0, times(t), dt, XtObs);
        }
        Xt.mVec /= N;
        cost += calculate_cf2(Yt3Vecs[t], Xt.mVec, weights[t]);
    }
    cout << "true:" << rate.k.transpose()<< " " << cost << endl;
    cost = 0;
    rate.k << 0.79,	0.25,	0.0915363,	0.969999,	0.243538,	0.0985505;
      for(int t = 0; t < nTimeSteps; t++){
        Nonlinear_ODE6 sys(rate);
        Protein_Components Xt(times(t), nMoments, N);
        Moments_Mat_Obs XtObs(Xt);
        for (int i = 0; i < X_0.rows(); i++) {
            State_N x0 = convertInit(X_0, i);
            Xt.index = i;
            integrate_adaptive(controlledStepper, sys, x0, t0, times(t), dt, XtObs);
        }
        Xt.mVec /= N;
        cost += calculate_cf2(Yt3Vecs[t], Xt.mVec, weights[t]);
    }
    cout << "true:" << rate.k.transpose() << " " << cost << endl;

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << "CODE FINISHED RUNNING IN " << duration << " s TIME!" << endl;

    return 0; // just to close the program at the end.
}

