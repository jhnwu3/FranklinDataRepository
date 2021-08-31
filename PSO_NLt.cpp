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
struct Protein_Moments {
    VectorXd mVec;
    double timeToRecord;
    Protein_Moments(double tf, int mom) {
        mVec = VectorXd::Zero(mom);
        timeToRecord = tf;
    }

};

struct Moments_Vec_Obs
{
    struct Protein_Moments& pMome;
    Moments_Vec_Obs(struct Protein_Moments& pMom) : pMome(pMom) {}
    void operator()(State_N const& c, const double t) const
    {
        if (t == pMome.timeToRecord) {
            int upperDiag = 2 * N_SPECIES;
            for (int i = 0; i < N_SPECIES; i++) {
                pMome.mVec(i) += c[i];
                for (int j = i; j < N_SPECIES; j++) {
                    if (i == j) { // diagonal elements
                        pMome.mVec(N_SPECIES + i) += c[i] * c[j];
                    }
                    else { //upper right diagonal elements
                       // cout << "upperDiag: " << upperDiag << endl; 
                        pMome.mVec(upperDiag) += c[i] * c[j];
                        upperDiag++;
                    }
                }
            }
        }
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
                    if (i == j) { // diagonal elements
                        dComp.mVec(N_SPECIES + i) += c[i] * c[j]; // variances
                    }
                    else {
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
    for (int i = 0; i < N_DIM; i++) {
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

string findDouble(string line, int startPos) {
    string doble;
    int i = startPos;
    int wDist = 0;
    while (i < line.length() && !isspace(line.at(i))) {
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
            }else{
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

    cout << "Chkpt reached!" << endl;
    cout << "new weight matrix:" << endl << wt << endl << endl;
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
    double t0 = 0, tf = 50, dt = 1.0; // time variables
    int nTimeSteps = 5;
    VectorXd times = VectorXd::Zero(nTimeSteps);
    times << 10, 20, 30, 40, tf;
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
    int N = 5000;
    int nParts = 50; // first part PSO
    int nSteps = 50;
    int nParts2 = 10; // second part PSO
    int nSteps2 = 500;
    int nMoments = (N_SPECIES * (N_SPECIES + 3)) / 2; // var + mean + cov
    int hone = 24;
    //nMoments = 2*N_SPECIES; // mean + var only!
    VectorXd wmatup(4);
    wmatup << 0.15, 0.35, 0.60, 0.9;
    double uniLowBound = 0.0, uniHiBound = 1.0;
    random_device RanDev;
    mt19937 gen(RanDev());
    uniform_real_distribution<double> unifDist(uniLowBound, uniHiBound);
    
    MatrixXd wt = MatrixXd::Identity(nMoments, nMoments); // wt matrix - we will just use this for blind PSO for now
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
    cout << "wt:" << endl << wt << endl;

    MatrixXd GBMAT(0, 0); // iterations of global best vectors
    MatrixXd PBMAT(nParts, Npars + 1); // particle best matrix + 1 for cost component
    MatrixXd POSMAT(nParts, Npars); // Position matrix as it goees through it in parallel

    cout << "Reading in data!" << endl;
    /* Initial Conditions */
    MatrixXd X_0(N, Npars);
    MatrixXd Y_0(N, Npars);
    ifstream X0File("X_0.txt");
    ifstream Y0File("Y_0.txt");
    X_0 = readIntoMatrix(X0File, N, N_SPECIES); // Bill initCond
    Y_0 = readIntoMatrix(Y0File, N, N_SPECIES); 

    /* Solve for Y_t (mu). */
    cout << "Loading in Truk!" << endl;
    struct K tru;
    tru.k = VectorXd::Zero(Npars);
    tru.k << 5.0, 0.1, 1.0, 8.69, 0.05, 0.70;
    tru.k /= (9.69);
    tru.k(1) += 0.05;
    tru.k(4) += 0.05; // make sure not so close to the boundary
    // tru.k <<  0.51599600,  0.06031990, 0.10319900, 0.89680100, 0.05516000, 0.00722394; // Bill k

    /* testing here! */
    //VectorXd testVec = VectorXd::Zero(6);
    //testVec << 0.825114,	0.178173,	0.075811,	0.562319,	0.019967,	0.014666;
    //testVec <<  0.764108,	0.153013,	0.081472,	0.635459,	0.02754,	0.028507;
    //testVec = tru.k;

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
        if(t == 0){
            cout << "Yt:" << Yt.mVec.transpose() << endl;
            cout << "Xt:" << Xt.mVec.transpose() << endl;
        }
        trukCost += calculate_cf2(Yt.mVec,Xt.mVec, wt);
        Xt3Vecs.push_back(Xt.mVec);
        Yt3Mats.push_back(Yt.mat);
        Yt3Vecs.push_back(Yt.mVec);
    }
    cout << "truk cost:"<< trukCost << endl;
    /* Instantiate seedk aka global costs */
    struct K seed;
    seed.k = VectorXd::Zero(Npars); 
    //seed.k = testVec;
    for (int i = 0; i < Npars; i++) { 
        seed.k(i) = unifDist(gen);
    }
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
        costSeedK += calculate_cf2(Yt3Vecs[t], Xt.mVec, wt);
        cout << "Xt at seedk:" << Xt.mVec.transpose()  << endl;
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
    // #pragma omp parallel for 
        for(int particle = 0; particle < nParts; particle++){
            random_device pRanDev;
            mt19937 pGenerator(pRanDev());
            uniform_real_distribution<double> pUnifDist(uniLowBound, uniHiBound);
            /* instantiate all particle rate constants with unifDist */
            if(step == 0){
                /* temporarily assign specified k constants */
                for(int i = 0; i < Npars; i++){
                    POSMAT(particle, i) = pUnifDist(pGenerator);//tru.k(i) + alpha * (0.5 - unifDist(pGenerator));
                }
                //POSMAT.row(particle) = testVec;

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
                    cost += calculate_cf2(Yt3Vecs[t], XtPSO.mVec, wt);
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
                    cost += calculate_cf2(Yt3Vecs[t], XtPSO.mVec, wt);
                }
               
                /* update gBest and pBest */
            //     #pragma omp critical
            //    {
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
            //   }
            }
        }
        GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1); // Add to GBMAT after resizing
        for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = GBVEC(i);}
        GBMAT(GBMAT.rows() - 1, Npars) = gCost;
        cout << "New GBMAT from blind pso!" << endl;
        cout << GBMAT << endl << endl << endl;
        sfi = sfi - (sfe - sfg) / nSteps;   // reduce the inertial weight after each step 
        sfs = sfs + (sfe - sfg) / nSteps;
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
    VectorXd subset = VectorXd::Zero(nMoments);
    subset << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 ,23, 24, 25, 26;
    cout << "targeted PSO has started!" << endl; 
    sfp = 3.0, sfg = 1.0, sfe = 6.0; // initial particle historical weight, global weight social, inertial
    sfi = sfe, sfc = sfp, sfs = sfg; // below are the variables being used to reiterate weights
    double nearby = sdbeta;
    VectorXd chkpts = wmatup * nSteps2;
    for(int step = 0; step < nSteps2; step++){
        if(step == 0 || step == chkpts(0) || step == chkpts(1) || step == chkpts(2) || step == chkpts(3)){ /* update wt   matrix || step == chkpts(0) || step == chkpts(1) || step == chkpts(2) || step == chkpts(3) */
            cout << "Updating Weight Matrix!" << endl;
            cout << "GBVEC AND COST:" << GBMAT.row(GBMAT.rows() - 1) << endl;
            nearby = squeeze * nearby;
            /* reinstantiate gCost */
            struct K gPos;
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
                weights[t] = customWtMat(Yt3Mats[t], gXt.mat, nMoments, N, subset);
                cost += calculate_cf2(Yt3Vecs[t], gXt.mVec, weights[t]);
            }
            gCost = cost;
            // /* make sure to set proper subsets each time*/
            // if(step == 0){
            //     subsetCol.resize(tgCol.size());
            //     subsetCol = tgCol;
            //     wt.resize(tgCol.size(), tgCol.size());
            // }else if (step == chkpts(0)){
            //     subsetCol.resize(reCol1.size());
            //     subsetCol = reCol1;
            // }else if (step == chkpts(1)){
            //     subsetCol.resize(reCol2.size());
            //     subsetCol = reCol2;
            // }

            // resizedYt.resize(subsetCol.size()); // make sure yt is right size
            // VectorXd resizedXt = VectorXd::Zero(subsetCol.size());
            // for(int i = 0; i < subsetCol.size(); i++){
            //     resizedXt(i) = gXt.mVec(subsetCol(i));
            //     resizedYt(i) = Yt.mVec(subsetCol(i));
            // }
            // wt.resize(subsetCol.size(), subsetCol.size());
            // wt = customWtMat(Yt.mat, gXt.mat, nMoments, N, subsetCol);
            
            //gCost = calculate_cf2(resizedYt, resizedXt, wt);
            
            //wt = customWtMat(Yt3Mat, gPSOMat, nMoments, N, subset);
            hone += 4;
            //gCost = calculate_cf2(Yt3Vec, gXt3, wt);
            GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1);
            for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = gPos.k(i);}
            GBMAT(GBMAT.rows() - 1, Npars) = gCost;
        }
    // #pragma omp parallel for 
        for(int particle = 0; particle < nParts2; particle++){
            random_device pRanDev;
            mt19937 pGenerator(pRanDev());
            uniform_real_distribution<double> pUnifDist(uniLowBound, uniHiBound);
        
            if(step == 0 || step == chkpts(0) || step == chkpts(1) || step == chkpts(2) || step == chkpts(3)){
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
                
                // VectorXd resizedXt = VectorXd::Zero(subsetCol.size());
                // for(int i = 0; i < subsetCol.size() ;i++){
                //     resizedXt(i) = XtPSO.mVec(subsetCol(i));
                // }
                // double cost = calculate_cf2(Yt3Vec, XtPSO3, wt);
                /* instantiate PBMAT */
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

                //VectorXd XtPSO3 = VectorXd::Zero(nMoments);
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
                    //XtPSO3 += XtPSO.mVec;
                }
                
                // VectorXd resizedXt = VectorXd::Zero(subsetCol.size());
                // for(int i = 0; i < subsetCol.size(); i++){
                //     resizedXt(i) = XtPSO.mVec(subsetCol(i));
                // }
                //double cost = calculate_cf2(Yt3Vec, XtPSO3, wt);

                /* update pBest and gBest */
                // #pragma omp critical
                // {
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
                // }
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
    }
    cout << "GBMAT after targeted PSO:" << endl << GBMAT << endl;
    trukCost = 0;
    for(int t = 0; t < nTimeSteps; t++){
        trukCost += calculate_cf2(Yt3Vecs[t], Xt3Vecs[t], weights[t]);
    }

    cout << "truk: " << tru.k.transpose() << " with trukCost with new weights:" << trukCost << endl;
    dist = calculate_cf1(tru.k, GBVEC);
    cout << "total difference b/w truk and final GBVEC:" << dist << endl; // compute difference
    
    struct K rs1;
    rs1.k = VectorXd::Zero(Npars); 
    rs1.k << 0.79719, 0.138687, 0.104211, 0.97094, 0.064014, 0.075424;
    double costRS1 = 0;
    for(int t = 0; t < nTimeSteps; t++){
        Protein_Components Xt(times(t), nMoments, N);
        Moments_Mat_Obs XtObs(Xt);
        Nonlinear_ODE6 sys(rs1);
        for (int i = 0; i < N; i++) {
            //State_N c0 = gen_multi_norm_iSub();
            State_N c0 = convertInit(X_0, i);
            Xt.index = i;
            integrate_adaptive(controlledStepper, sys, c0, t0, times(t), dt, XtObs);
        }
        Xt.mVec /= N;  
        costRS1 += calculate_cf2(Yt3Vecs[t], Xt.mVec, weights[t]);
    }
    cout << "for Ks:" << rs1.k.transpose() << " has cost:" << costRS1 << endl;
    struct K rs2;
    rs2.k = VectorXd::Zero(Npars); 
    rs2.k << 0.781196, 0.140816, 0.097871, 0.82296, 0.046377, 0.069316;
    double costRS2 = 0;
    for(int t = 0; t < nTimeSteps; t++){
        Protein_Components Xt(times(t), nMoments, N);
        Moments_Mat_Obs XtObs(Xt);
        Nonlinear_ODE6 sys(rs2);
        for (int i = 0; i < N; i++) {
            //State_N c0 = gen_multi_norm_iSub();
            State_N c0 = convertInit(X_0, i);
            Xt.index = i;
            integrate_adaptive(controlledStepper, sys, c0, t0, times(t), dt, XtObs);
        }
        Xt.mVec /= N;  
        costRS2 += calculate_cf2(Yt3Vecs[t], Xt.mVec, weights[t]);
    }
    cout << "for Ks:" << rs2.k.transpose() << " has cost:" << costRS2 << endl;


    struct K rs3;
    rs3.k = VectorXd::Zero(Npars); 
    rs3.k << 0.781196, 0.140816, 0.097871, 0.82296, 0.046377, 0.069316;
    double costRS3 = 0;
    for(int t = 0; t < nTimeSteps; t++){
        Protein_Components Xt(times(t), nMoments, N);
        Moments_Mat_Obs XtObs(Xt);
        Nonlinear_ODE6 sys(rs3);
        for (int i = 0; i < N; i++) {
            //State_N c0 = gen_multi_norm_iSub();
            State_N c0 = convertInit(X_0, i);
            Xt.index = i;
            integrate_adaptive(controlledStepper, sys, c0, t0, times(t), dt, XtObs);
        }
        Xt.mVec /= N;  
        costRS3 += calculate_cf2(Yt3Vecs[t], Xt.mVec, weights[t]);
    }
    cout << "for Ks:" << rs3.k.transpose() << " has cost:" << costRS3 << endl;
    
    ofstream plot;
	plot.open("GBMAT.csv");
	MatrixXd GBMATWithSteps(GBMAT.rows(), GBMAT.cols() + 1);
	VectorXd globalIterations(GBMAT.rows());
	for(int i = 0; i < GBMAT.rows(); i++){
		globalIterations(i) = i;
	}
	GBMATWithSteps << globalIterations, GBMAT;
	for(int i = 0; i < GBMATWithSteps.rows(); i++){
        for(int j = 0; j < GBMATWithSteps.cols(); j++){
            if(j == 0){
                plot << GBMATWithSteps(i,j);
            }else{
                plot << "," << GBMATWithSteps(i,j);
            }
        }
        plot << endl;
    }

	plot.close();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << "CODE FINISHED RUNNING IN " << duration << " s TIME!" << endl;

    return 0; // just to close the program at the end.
}

