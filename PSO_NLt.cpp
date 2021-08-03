// PSO.cpp : Replacing Dr. Stewarts linear 3 ODE's with the nonlinear3 ODE system provided way earlier
//

#include <iostream>
#include <fstream>
#include <boost/math/distributions.hpp>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>

#define N_SPECIES 6
#define N 10 // # of samples to sample over
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

typedef boost::array< double, 3 > State_3;
typedef runge_kutta_cash_karp54< State_3 > Error_RK_Stepper_3;
typedef controlled_runge_kutta< Error_RK_Stepper_3 > Controlled_RK_Stepper_3;

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

/* /* 3-var linear ODE system - need to rename! @TODO */
class Linear_ODE3
{
    struct K bill;

public:
    Linear_ODE3(struct K G) : bill(G) {}

    void operator() (const State_3& c, State_3& dcdt, double t)
    {
        MatrixXd kr(3, 3);
        kr << 0, bill.k(1), bill.k(3),
            bill.k(2), 0, bill.k(0),
            0, bill.k(4), 0;
        dcdt[0] = (kr(0, 0) * c[0] - kr(0, 0) * c[0]) +
            (kr(0, 1) * c[1] - kr(1, 0) * c[0]) +
            (kr(0, 2) * c[2] - kr(2, 0) * c[0]);

        dcdt[1] = (kr(1, 0) * c[0] - kr(0, 1) * c[1]) +
            (kr(1, 1) * c[1] - kr(1, 1) * c[1]) +
            (kr(1, 2) * c[2] - kr(2, 1) * c[1]);

        dcdt[2] = (kr(2, 0) * c[0] - kr(0, 2) * c[2]) +
            (kr(2, 1) * c[1] - kr(1, 2) * c[2]) +
            (kr(2, 2) * c[2] - kr(2, 2) * c[2]);
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
                dComp.mat(dComp.index, i) = c[i];
                for (int j = i; j < N_SPECIES; j++) {
                    if (i == j) { // diagonal elements
                        dComp.mVec(N_SPECIES + i) += c[i] * c[j];
                    }else {
                        dComp.mVec(upperDiag) += c[i] * c[j];
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
    // VectorXd mu(3);
    // mu << 4.78334234137469844730960782,
    //     5.52142091946216110500584912965,
    //     4.3815581042632114978686130;
    // MatrixXd sigma(3, 3);
    // sigma << 0.008298802814695093876186221, 0, 0,
    //     0, 0.0000799968001706564273219830, 0,
    //     0, 0, 0.000937060821340228802149700;
    // VectorXd mu(3);
    // mu << 4.78334234137469844730960782,
    //     5.52142091946216110500584912965,
    //     4.3815581042632114978686130;
    // MatrixXd sigma(3, 3);
    // sigma << 800.298802814695093876186221, 0, 0,
    //     0, 7.99968001706564273219830, 0,
    //     0, 0, 93.7060821340228802149700;

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
VectorXd comp_vel_vec(const VectorXd& posK, int seed) {
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
        if (pos > 1.0) {
            cout << "overflow!" << endl;
            pos += -0.001;
        }else if (pos < 0.001) {
            cout << "underflow!"<< pos << endl;
            cout << "pos" << posK.transpose() << endl;
            pos += 0.001;
        }
        double alpha = 4 * pos;
        double beta = 4 - alpha;
       // cout << "alpha:" << alpha << "beta:" << beta << endl;
        std::gamma_distribution<double> aDist(alpha, 1);
        std::gamma_distribution<double> bDist(beta, 1);

        double x = aDist(generator);
        double y = bDist(generator);

        rPoint(px) = (x / (x + y)); // test if uniform does worse than the beta version.
        // I will run a beta version, uniform distribution, use larger variances, 50, 100, 50, 
        // initialize positions close to truek 
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
    // for(int i = 0; i < n; i++){
    //     cost += (estVec(i) - trueVec(i)) * (estVec(i) - trueVec(i));
    // }
    return cost;
}
double calculate_cf2(const VectorXd& trueVec, const  VectorXd& estVec, const MatrixXd& w) {
    double cost = 0;
    VectorXd diff(trueVec.size());
    /*for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
           cost += (estVec(i) - trueVec(i)) * w(i,j) *(estVec(j) - trueVec(j));
        }
    }*/
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
MatrixXd customWtMat(const MatrixXd& Yt, const MatrixXd& Xt, int nMoments){
    /* first moment differences */
    MatrixXd fmdiffs = Yt - Xt; 
    /* second moment difference computations - @todo make it variable later */
    MatrixXd smdiffs(N,6);
    for(int i = 0; i < N_SPECIES; i++){
        smdiffs.col(i) = (Yt.col(i).array() * Yt.col(i).array()) - (Xt.col(i).array() * Xt.col(i).array());
    }
    int nCross = nMoments - 2 * N_SPECIES;
    MatrixXd cpDiff(N, nCross);
    
    /* cross differences */
    int upperDiag = 0;
    for(int i = 0; i < N_SPECIES; i++){
        for(int j = i + 1; j < N_SPECIES; j++){
            cpDiff.col(upperDiag) = (Yt.col(i).array() * Yt.col(j).array()) - (Xt.col(i).array() * Xt.col(j).array());
            upperDiag++;
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
    MatrixXd wt = MatrixXd::Zero(nMoments, nMoments);

    for(int i = 0; i < nMoments; i++){
        wt(i,i) = 1 / variances(i); // cleanup code and make it more vectorized later.
    }
    cout << "Chkpt reached!" << endl;
    cout << "new weight matrix:" << endl << wt << endl << endl;
    return wt;
}

int main() {
    auto t1 = std::chrono::high_resolution_clock::now();
    /*---------------------- Setup ------------------------ */
    
    /* Variables (global) */
    double t0 = 0, tf = 5.0 * 9.69, dt = 1.0;
    int Npars = 6;
    double squeeze = 0.975, sdbeta = 0.15;
    double boundary = 0.001;
    /* SETUP */
    int useDiag = 0;
    int sf1 = 1;
    int sf2 = 1;

    /* PSO params */
    double sfp = 3.0, sfg = 1.0, sfe = 6.0; // initial particle historical weight, global weight social, inertial
    double sfi = sfe, sfc = sfp, sfs = sfg; // below are the variables being used to reiterate weights
    double alpha = 0.2;
    int nParts = 300; // first part PSO
    int nSteps = 40;
    int nParts2 = 20; // second part PSO
    int nSteps2 = 200;
    int nMoments = (N_SPECIES * (N_SPECIES + 3)) / 2;
    VectorXd wmatup(4);
    wmatup << 0.2, 0.4, 0.6, 0.8;
    double uniLowBound = 0.0, uniHiBound = 1.0;
    random_device RanDev;
    mt19937 gen(RanDev());
    uniform_real_distribution<double> unifDist(uniLowBound, uniHiBound);
    // nMoments = 2*N_SPECIES;
    // nMoments = N_SPECIES;
    cout << "Using two part PSO"<< endl;
    cout << "Bounds for Uniform Distribution" << uniLowBound << "," << uniHiBound << endl;
    cout << "Blind PSO using "<< nMoments << " moments." << endl;
    cout << "Sample Size:" << N << " Nparts:" << nParts << " Nsteps:" << nSteps << endl;
    cout << "Targeted PSO updated nParts:" <<  nParts2 << " Nsteps:" << nSteps2 << endl;
    cout << "using tf:" << tf << endl;
    MatrixXd wt = MatrixXd::Identity(nMoments, nMoments); // wt matrix
    MatrixXd GBMAT(0, 0); // iterations of global best vectors
    MatrixXd PBMAT(nParts, Npars + 1); // particle best matrix + 1 for cost component
    MatrixXd POSMAT(nParts, Npars); // Position matrix as it goees through it in parallel
    VectorXd mvnVec(3);
    mvnVec << 80,
        120,
        85;
    MatrixXd covarMat(3, 3);
    covarMat << 50, 0, 0,
        0, 100, 0,
        0, 0, 50;

    cout << "mu:" << mvnVec.transpose() << endl;
    cout << "covarMat:" << endl << covarMat << endl << endl;

    /* Initial Conditions */
    MatrixXd X_0(N, Npars);
    MatrixXd Y_0(N, Npars);
    ifstream X0File("X_0.txt");
    ifstream Y0File("Y_0.txt");
    X_0 = readIntoMatrix(X0File, N, N_SPECIES); // Bill initCond
    Y_0 = readIntoMatrix(Y0File, N, N_SPECIES); 
    // for(int i = 0; i < N; i++){
    //     X_0.row(i) = gen_multinorm_iVec();
    //     Y_0.row(i) = gen_multinorm_iVec();
    // }

    /* Solve for Y_t (mu). */
    struct K tru;
    tru.k = VectorXd::Zero(Npars);
    tru.k << 5.0, 0.1, 1.0, 8.69, 0.05, 0.70;
    tru.k /= (9.69);
    tru.k(1) += 0.05;
    tru.k(4) += 0.05; // make sure not so close to the boundary
    //tru.k <<  0.51599600,  0.06031990, 0.10319900, 0.89680100, 0.05516000, 0.00722394; // Bill k
    Nonlinear_ODE6 trueSys(tru);
    Protein_Components Yt(tf, nMoments, N);
    Moments_Mat_Obs YtObs(Yt);
    Controlled_RK_Stepper_N controlledStepper;
    for (int i = 0; i < N; i++) {
        //State_N c0 = gen_multi_norm_iSub(); // Y_0 is simulated using norm dist.
        State_N c0 = convertInit(Y_0, i);
        Yt.index = i;
        integrate_adaptive(controlledStepper, trueSys, c0, t0, tf, dt, YtObs);
    }
    Yt.mVec /= N;
    cout << "Yt:" << Yt.mVec.transpose() << endl;
    /* PSO costs */
    double gCost = 20000;
    /* Instantiate seedk aka global costs */
    struct K seed;
    seed.k = VectorXd::Zero(Npars); 
    for (int i = 0; i < Npars; i++) { 
        seed.k(i) = unifDist(gen);
    }
    
    Protein_Components Xt(tf, nMoments, N);
    Moments_Mat_Obs XtObs(Xt);
    Nonlinear_ODE6 sys(seed);
    for (int i = 0; i < N; i++) {
        //State_N c0 = gen_multi_norm_iSub();
        State_N c0 = convertInit(X_0, i);
        Xt.index = i;
        integrate_adaptive(controlledStepper, sys, c0, t0, tf, dt, XtObs);
    }
    Xt.mVec /= N;  

    double costSeedk = calculate_cf2(Yt.mVec, Xt.mVec, wt); 
    cout << "seedk:"<< seed.k.transpose()<< "| cost:" << costSeedk << endl;
    cout << "Xt:" << Xt.mVec.transpose() << endl;
    gCost = costSeedk; //initialize costs and GBMAT
    VectorXd GBVEC = seed.k;
    GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1);
    for (int i = 0; i < Npars; i++) {
        GBMAT(GBMAT.rows() - 1, i) = seed.k(i);
    }
    GBMAT(GBMAT.rows() - 1, Npars) = gCost;
    
    cout << "PSO begins!" << endl;
    
    /* PSO begins */
    for(int step = 0; step < nSteps; step++){
    //#pragma omp parallel for 
        for(int particle = 0; particle < nParts; particle++){
            random_device pRanDev;
            mt19937 pGenerator(pRanDev());
            uniform_real_distribution<double> pUnifDist(uniLowBound, uniHiBound);
            /* instantiate all particle rate constants with unifDist */
            if(step == 0){
                /* temporarily assign specified k constants */
                for(int i = 0; i < Npars; i++){
                    POSMAT(particle, i) = pUnifDist(pGenerator);//tru.k(i) + alpha * (0.5 - unifDist(pGenerator));
                    // if(POSMAT(particle, i) < 0){
                    //     POSMAT(particle, i) = -POSMAT(particle,i);
                    // }
                }

                struct K pos;
                pos.k = VectorXd::Zero(Npars);
                for(int i = 0; i < Npars; i++){
                    pos.k(i) = POSMAT(particle, i);
                }
                Nonlinear_ODE6 initSys(pos);
                Protein_Components XtPSO(tf, nMoments, N);
                Moments_Mat_Obs XtObsPSO(XtPSO);
                for(int i = 0; i < N; i++){
                    //State_N c0 = gen_multi_norm_iSub();
                    State_N c0 = convertInit(X_0, i);
                    XtPSO.index = i;
                    integrate_adaptive(controlledStepper, initSys, c0, t0, tf, dt, XtObsPSO);
                }
                XtPSO.mVec/=N;
                double cost = calculate_cf2(Yt.mVec, XtPSO.mVec, wt); 
                
                /* instantiate PBMAT */
                for(int i = 0; i < Npars; i++){
                    PBMAT(particle, i) = POSMAT(particle, i);
                }
                PBMAT(particle, Npars) = cost; // add cost to final column
            }else{ // PSO after instantiations
                /* using new rate constants, instantiate particle best values */
                /* step into PSO */
                double w1 = sfi * pUnifDist(pGenerator)/ sf2, w2 = sfc * pUnifDist(pGenerator) / sf2, w3 = sfs * pUnifDist(pGenerator)/ sf2;
                double sumw = w1 + w2 + w3; //w1 = inertial, w2 = pbest, w3 = gbest
                w1 = w1 / sumw; w2 = w2 / sumw; w3 = w3 / sumw;
                //w1 = 0.05; w2 = 0.90; w3 = 0.05;
                struct K pos;
                pos.k = VectorXd::Zero(Npars);
                pos.k = POSMAT.row(particle);
                VectorXd rpoint = comp_vel_vec(pos.k, particle);
                VectorXd PBVEC(Npars);
                for(int i = 0; i < Npars; i++){
                    PBVEC(i) = PBMAT(particle, i);
                }
                pos.k = w1 * rpoint + w2 * PBVEC + w3 * GBVEC; // update position of particle
                POSMAT.row(particle) = pos.k;

                /*solve ODEs and recompute cost */
                Protein_Components XtPSO(tf, nMoments, N);
                Moments_Mat_Obs XtObsPSO1(XtPSO);
                Nonlinear_ODE6 stepSys(pos);
       
                for(int i = 0; i < N; i++){
                    State_N c0 = convertInit(X_0, i);
                    XtPSO.index = i;
                    integrate_adaptive(controlledStepper, stepSys, c0, t0, tf, dt, XtObsPSO1);
                }
                XtPSO.mVec/=N;
                
                double cost = calculate_cf2(Yt.mVec, XtPSO.mVec, wt); 
                /* update gBest and pBest */
            //     #pragma omp critical
            //    {
                    // cout << "step:" << step << " from thread:" << omp_get_thread_num() << endl;
                    // cout << "particle:" << particle << endl;
                if(cost < PBMAT(particle, Npars)){ // particle best cost
                    for(int i = 0; i < Npars; i++){
                        PBMAT(particle, i) = pos.k(i);
                    }
                    PBMAT(particle, Npars) = cost;
                    if(cost < gCost){
                        gCost = cost;
                        GBVEC = pos.k;
                        GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1);
                        for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = GBVEC(i);}
                        GBMAT(GBMAT.rows() - 1, Npars) = gCost;
                    }   
                }
              //}
            }
        }
        sfi = sfi - (sfe - sfg) / nSteps;   // reduce the inertial weight after each step 
        sfs = sfs + (sfe - sfg) / nSteps;
    }
    // cout << "POSMAT:" << endl; 
    // cout <<  POSMAT<< endl << endl;
    // cout << "PBMAT:" << endl;
    // cout << PBMAT << endl << endl;
    cout << "GBMAT from first PSO:" << endl << endl;
    cout << GBMAT << endl << endl;
    cout << "truk: " << tru.k.transpose() << endl;
    double dist = calculate_cf1(tru.k, GBVEC);
    cout << "total difference b/w truk and final GBVEC" << dist << endl << endl; // compute difference
    
    /*** targeted PSO ***/
    POSMAT.conservativeResize(nParts2, Npars); // resize matrices to fit targetted PSO
    PBMAT.conservativeResize(nParts2, Npars + 1);
    cout << "targeted PSO has started!" << endl; 
    sfp = 3.0, sfg = 1.0, sfe = 6.0; // initial particle historical weight, global weight social, inertial
    sfi = sfe, sfc = sfp, sfs = sfg; // below are the variables being used to reiterate weights
    double nearby = sdbeta;
    VectorXd chkpts = wmatup * nSteps2;
    for(int step = 0; step < nSteps2; step++){
        if(step == 0 || step == chkpts(0) || step == chkpts(1) || step == chkpts(2) || step == chkpts(3)){ /* update wt matrix */
            cout << "Updating Weight Matrix!" << endl;
            cout << "GBVEC AND COST:" << GBMAT.row(GBMAT.rows() - 1).transpose() << endl;
            cout << "GBVEC:" << GBVEC << "cost:" << gCost << endl;
            /* reinstantiate gCost */
            struct K gPos;
            gPos.k = GBVEC;
            Protein_Components gXt(tf, nMoments, N);
            Moments_Mat_Obs gXtObs(gXt);
            Nonlinear_ODE6 gSys(gPos);
            for (int i = 0; i < N; i++) {
                //State_N c0 = gen_multi_norm_iSub();
                State_N c0 = convertInit(X_0, i);
                gXt.index = i;
                integrate_adaptive(controlledStepper, gSys, c0, t0, tf, dt, gXtObs);
            }
            gXt.mVec /= N;  
            wt = customWtMat(Yt.mat, gXt.mat, nMoments);
            gCost = calculate_cf2(Yt.mVec, gXt.mVec, wt);
            GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1);
            for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = gPos.k(i);}
            GBMAT(GBMAT.rows() - 1, Npars) = gCost;
        }
    //#pragma omp parallel for 
        for(int particle = 0; particle < nParts2; particle++){
            random_device pRanDev;
            mt19937 pGenerator(pRanDev());
            uniform_real_distribution<double> pUnifDist(uniLowBound, uniHiBound);
        
            /* instantiate all particle rate constants with unifDist */
            if(step == 0){
                /* reinstantiate particles closer towards global best */
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

                    std::gamma_distribution<double> aDist(alpha, 1);
                    std::gamma_distribution<double> bDist(beta, 1);

                    double x = aDist(pGenerator);
                    double y = bDist(pGenerator);
                    double myg = x / (x + y);
            
                    if (wasflipped == 1) {
                        wasflipped = 0;
                        myg = 1 - myg;
                    }
                    POSMAT(particle, edim) = myg;
                }

                struct K pos;
                pos.k = VectorXd::Zero(Npars);
                for(int i = 0; i < Npars; i++){
                    pos.k(i) = POSMAT(particle, i);
                }
                Nonlinear_ODE6 initSys(pos);
                Protein_Components XtPSO(tf, nMoments, N);
                Moments_Mat_Obs XtObsPSO(XtPSO);
                for(int i = 0; i < N; i++){
                    //State_N c0 = gen_multi_norm_iSub();
                    State_N c0 = convertInit(X_0, i);
                    XtPSO.index = i;
                    integrate_adaptive(controlledStepper, initSys, c0, t0, tf, dt, XtObsPSO);
                }
                XtPSO.mVec/=N;
                double cost = calculate_cf2(Yt.mVec, XtPSO.mVec, wt);
                /* instantiate PBMAT */
                for(int i = 0; i < Npars; i++){
                    PBMAT(particle, i) = POSMAT(particle, i);
                }
                PBMAT(particle, Npars) = cost; // add cost to final column
            }else{ // PSO after instantiations
                /* using new rate constants, instantiate particle best values */
                /* step into PSO */
                double w1 = sfi * pUnifDist(pGenerator)/ sf2, w2 = sfc * pUnifDist(pGenerator) / sf2, w3 = sfs * pUnifDist(pGenerator)/ sf2;
                double sumw = w1 + w2 + w3; //w1 = inertial, w2 = pbest, w3 = gbest
                w1 = w1 / sumw; w2 = w2 / sumw; w3 = w3 / sumw;
                //w1 = 0.05; w2 = 0.90; w3 = 0.05;
                struct K pos;
                pos.k = VectorXd::Zero(Npars);
                pos.k = POSMAT.row(particle);
                VectorXd rpoint = comp_vel_vec(pos.k, particle);
                VectorXd PBVEC(Npars);
                for(int i = 0; i < Npars; i++){
                    PBVEC(i) = PBMAT(particle, i);
                }
                pos.k = w1 * rpoint + w2 * PBVEC + w3 * GBVEC; // update position of particle
                POSMAT.row(particle) = pos.k;
                /*solve ODEs and recompute cost */
                Protein_Components XtPSO(tf, nMoments, N);
                Moments_Mat_Obs XtObsPSO1(XtPSO);
                Nonlinear_ODE6 stepSys(pos);
    
                for(int i = 0; i < N; i++){
                    //State_N c0 = gen_multi_norm_iSub();
                    State_N c0 = convertInit(X_0, i);
                    XtPSO.index = i;
                    integrate_adaptive(controlledStepper, stepSys, c0, t0, tf, dt, XtObsPSO1);
                }
                XtPSO.mVec/=N;
                double cost = calculate_cf2(Yt.mVec, XtPSO.mVec, wt);

                /* update pBest and gBest*/
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
                        GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1);
                        for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = GBVEC(i);}
                        GBMAT(GBMAT.rows() - 1, Npars) = gCost;
                    }   
                }
               // }
            }
        }
        sfi = sfi - (sfe - sfg) / nSteps;   // reduce the inertial weight after each step 
        sfs = sfs + (sfe - sfg) / nSteps;
    }
    cout << "GBMAT after targeted PSO:" << endl << GBMAT << endl;
    cout << "truk: " << tru.k.transpose() << endl;
    dist = calculate_cf1(tru.k, GBVEC);
    cout << "total difference b/w truk and final GBVEC" << dist << endl; // compute difference

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

