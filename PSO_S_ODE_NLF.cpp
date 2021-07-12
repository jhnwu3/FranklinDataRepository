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
#define N 10000 // # of samples to sample over
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

const double ke = 0.0001, kme = 20, kf = 0.01, kmf = 18, kd = 0.03, kmd = 1,
ka2 = 0.01, ka3 = 0.01, C1T = 20, C2T = 5, C3T = 4;

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
        static std::mt19937 gen{ std::random_device{}() };
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

struct Data_Components {
    int index;
    MatrixXd mat;
    VectorXd mVec;
    double timeToRecord;
    Data_Components(double tf, int mom, int n) {
        mVec = VectorXd::Zero(mom);
        mat = MatrixXd::Zero(n, N_SPECIES);
        timeToRecord = tf;
    }
};
struct Protein_Moments {
    VectorXd mVec;
   // MatrixXd sec;
    double timeToRecord;
    Protein_Moments(double tf, int mom) {
        mVec = VectorXd::Zero(mom);
        //sec = MatrixXd::Zero(N_SPECIES, N_SPECIES);
        timeToRecord = tf;
    }
};

struct Mom_ODE_Observer
{
    struct Protein_Moments& pMome;
    Mom_ODE_Observer(struct Protein_Moments& pMom) : pMome(pMom) {}
    void operator()(State_N const& c, const double t) const
    {
        if (t == pMome.timeToRecord) {
            int upperDiag = 2 * N_SPECIES;
            for (int i = 0; i < N_SPECIES; i++) {
                pMome.mVec(i) += c[i];
                for (int j = i; j < N_SPECIES; j++) {
                    if (i == j) { // diagonal elements
                        pMome.mVec(N_SPECIES + i) += c[i] * c[j];
                    }else { //upper right diagonal elements
                       // cout << "upperDiag: " << upperDiag << endl; 
                        pMome.mVec(upperDiag) += c[i] * c[j];
                        upperDiag++;
                    }
                    // pMome.sec(i, j) += c[i] * c[j];
                    // pMome.sec(j, i) = pMome.sec(i, j);
                }
            }
        }
    }
};
struct Data_ODE_Observer
{
    struct Data_Components& dComp;
    Data_ODE_Observer(struct Data_Components& dCom) : dComp(dCom) {}
    void operator()(State_N const& c, const double t) const
    {
        if (t == dComp.timeToRecord) {
            int upperDiag = 2 * N_SPECIES;
            for (int i = 0; i < dComp.mat.cols(); i++) { dComp.mat(dComp.index, i) = c[i]; }
            for (int i = 0; i < N_SPECIES; i++) {
                dComp.mVec(i) += c[i];
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
struct Data_ODE_Observer3
{
    struct Data_Components& dComp;
    Data_ODE_Observer3(struct Data_Components& dCom) : dComp(dCom) {}
    void operator()(State_3 const& c, const double t) const
    {
        if (t == dComp.timeToRecord) {
            for (int i = 0; i < dComp.mat.cols(); i++) { dComp.mat(dComp.index, i) = c[i]; }
        }
    }
};
struct Data_Components6 {
    int index;
    MatrixXd mat;
    VectorXd sub;
    double timeToRecord;
};
struct Data_ODE_Observer6
{
    struct Data_Components6& dComp;
    Data_ODE_Observer6(struct Data_Components6& dCom) : dComp(dCom) {}
    void operator()(State_N const& c, const double t) const
    {
        if (t == dComp.timeToRecord) {
            int i = 0, j = 0;
            while (i < N_SPECIES && j < dComp.sub.size()) {
                if (i == dComp.sub(j)) {
                    dComp.mat(dComp.index, j) = c[i];
                    j++;
                }
                i++;
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
            c0[i] = c0Vec(j);
            j++;
        }
        else {
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
VectorXd comp_vel_vec(const VectorXd& posK) {
    VectorXd rPoint;
    rPoint = posK;
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    vector<int> rand;
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

        rPoint(px) = (x / (x + y));
    }
    return rPoint;
}
MatrixXd calculate_omega_weight_matrix(const MatrixXd &sample, const VectorXd &mu, int n){
    MatrixXd inv = MatrixXd::Zero(mu.size(), mu.size());
    VectorXd X = VectorXd::Zero(mu.size());
    for(int s = 0; s < n; s++){
        for(int row = 0; row < N_SPECIES; row++){
            X(row) = sample(s, row); 
            for(int col = row; col < N_SPECIES; col++){
                if( row == col){
                    X(N_SPECIES + row) = sample(s, row) * sample(s, col);
                }else{
                    X(2*N_SPECIES + (row + col - 1)) = sample(s,row) * sample(s,col);
                }
            }
        }
        for(int i = 0; i < mu.size(); i++){
            for(int j = 0; j < mu.size(); j++){
                inv(i,j) += (X(i) - mu(i)) * (X(j) - mu(j));
            }
        }
    }
    inv /= n;
    return inv.inverse();
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
// double unifDistRng(){
//     thread_local random_device pRanDev;
//     thread_local mt19937 engine(pRanDev());
//     uniform_real_distribution<double> dist(0.0, 1.0);
//     return dist(engine);
// }
// VectorXd randUnifVector(int n){
//     thread_local VectorXd unif(n);
//     for(int i = 0; i < n; i++){
//         unif(i) = unifDistRng();
//     }
//     return unif;
// }

int main() {

    auto t1 = std::chrono::high_resolution_clock::now();
    random_device RanDev;
    mt19937 gen(RanDev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);
    /*---------------------- Setup ------------------------ */
    int bsi = 1, Nterms = 9, useEqual = 0, Niter = 1, Biter = 1, psoIter = 2;
    
    /* Variables (global) */
    double t0 = 0, tf = 5.0 * 9.69, dt = 1.0;
    int Nprots = 3, Npars = 6;
    double squeeze = 0.975, sdbeta = 0.15;

    /* SETUP */
    int useDiag = 0;
    int sf1 = 1;
    int sf2 = 1;

    int Nparts = 300;
    int Nsteps = 40;
    
    cout << "sample size:" << N << " Nparts:" << Nparts << " Nsteps:" << Nsteps << endl;
    /* moments */
    int nMoments = (N_SPECIES * (N_SPECIES + 3)) / 2;
    MatrixXd Y_t = MatrixXd::Zero(N, N_SPECIES); // Values we are comparing towards - oMoments is derived from this.
    VectorXd pMoments(nMoments);
    MatrixXd X_t = MatrixXd::Zero(N, N_SPECIES);
    cout << "339" << endl;
    /* PSO weights */
    // double sfp = 3.0, sfg = 1.0, sfe = 6.0; // initial particle historical weight, global weight social, inertial
    // double sfi = sfe, sfc = sfp, sfs = sfg; // below are the variables being used to reiterate weights

    double boundary = 0.001;
    MatrixXd wt = MatrixXd::Identity(nMoments, nMoments); // wt matrix
    MatrixXd GBMAT(0, 0);
   
    VectorXd wmatup(4);
    wmatup << 0.15, 0.3, .45, .6;
    
    /* Solve for Y_t (mu). */
    struct K tru;
    tru.k = VectorXd::Zero(Npars);
    tru.k << 5.0, 0.1, 1.0, 8.69, 0.05, 0.70;
    tru.k /= (9.69);
   
    struct K seed;
    seed.k = VectorXd::Zero(Npars);
    Nonlinear_ODE6 trueSys(tru);
    Protein_Moments Yt(tf, nMoments);
    Mom_ODE_Observer YtObs(Yt);
    Controlled_RK_Stepper_N controlledStepper;
    cout << "362" << endl;
    for (int i = 0; i < N; i++) {
        State_N c0 = gen_multi_norm_iSub(); // Y_0 is simulated using lognorm dist.
        integrate_adaptive(controlledStepper, trueSys, c0, t0, tf, dt, YtObs);
    }
    Yt.mVec /= N;
    //Yt.sec /= N;

    /* PSO costs */
    double gCost = 20000;
    /* Instantiate seedk aka global costs */
    for (int i = 0; i < Npars; i++) { seed.k(i) = unifDist(gen); }
   
    Protein_Moments Xt(tf, nMoments);
    Mom_ODE_Observer XtObs(Xt);
    Nonlinear_ODE6 sys(seed);
    
    for (int i = 0; i < N; i++) {
        State_N c0 = gen_multi_norm_iSub();
        integrate_adaptive(controlledStepper, sys, c0, t0, tf, dt, XtObs);
    }
    Xt.mVec /= N;
    //Xt.sec /= N;
    double costSeedk = calculate_cf2(Yt.mVec, Xt.mVec, wt); 
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
    #pragma omp parallel for
    for(int particle = 0; particle < Nparts; particle++){
        double sfp = 3.0, sfg = 1.0, sfe = 6.0; // initial particle historical weight, global weight social, inertial
        double sfi = sfe, sfc = sfp, sfs = sfg; // below are the variables being used to reiterate weights
        random_device pRanDev;
        mt19937 pGenerator(pRanDev());
        uniform_real_distribution<double> pUnifDist(0.0, 1.0);
        /* instantiate all particle rate constants with unifDist */
        struct K pos;
        pos.k = VectorXd::Zero(Npars);
        for(int i = 0; i < Npars; i++){
            pos.k(i) = pUnifDist(pGenerator);
        }
        
        /* using new rate constants, instantiate particle best values */
        Nonlinear_ODE6 initSys(pos);
        Protein_Moments XtPSO(tf, nMoments);
        Mom_ODE_Observer XtObsPSO(XtPSO);
        for(int i = 0; i < N; i++){
            State_N c0 = gen_multi_norm_iSub();
            integrate_adaptive(controlledStepper, initSys, c0, t0, tf, dt, XtObsPSO);
        }
        XtPSO.mVec/=N;
        //XtPSO.sec/=N;
        double cost = calculate_cf2(Yt.mVec, XtPSO.mVec, wt);
        double partBest = cost; 
        VectorXd PBVEC = pos.k;
        /* step into PSO */
        double w1 = 6, w2 = 1, w3 = 1;
        for(int step = 0; step < Nsteps; step++){
            w1 = sfi * pUnifDist(pGenerator)/ sf2; w2 = sfc * pUnifDist(pGenerator) / sf2; w3 = sfs * pUnifDist(pGenerator)/ sf2;
            double sumw = w1 + w2 + w3; //w1 = inertial, w2 = pbest, w3 = gbest
            w1 = w1 / sumw; w2 = w2 / sumw; w3 = w3 / sumw;
            VectorXd rpoint = comp_vel_vec(pos.k);
            pos.k = w1 * rpoint + w2 * PBVEC + w3 * GBVEC; // update position of particle
            XtPSO.mVec.setZero();
           // XtPSO.sec.setZero();
            Nonlinear_ODE6 stepSys(pos);
            Mom_ODE_Observer XtObsPSO1(XtPSO);
            for(int i = 0; i < N; i++){
                State_N c0 = gen_multi_norm_iSub();
                integrate_adaptive(controlledStepper, initSys, c0, t0, tf, dt, XtObsPSO1);
            }
            XtPSO.mVec /= N;
            //XtPSO.sec /=N; l
            cost = calculate_cf2(Yt.mVec, XtPSO.mVec, wt);
            #pragma omp critical
            {
                if(cost < partBest){
                    PBVEC = pos.k;
                    partBest = cost;
                    if(cost < gCost){
                        gCost = cost;
                        GBVEC = pos.k;
                        GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1);
                        for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = GBVEC(i);}
                        GBMAT(GBMAT.rows() - 1, Npars) = gCost;
                    }   
                }
            }
            sfi = sfi - (sfe - sfg) / Nsteps;   // reduce the inertial weight after each step 
            sfs = sfs + (sfe - sfg) / Nsteps;
        }
        
    }


    cout << "GBMAT from first PSO:" << endl << endl;
    cout << GBMAT << endl << endl;
    cout << "truk" << tru.k.transpose() << endl;
   // cout << " targeted pso has begun!" << endl;
    // /* second targeted PSO */
    // int Nparts2 = 25; // targeted PSO requires far less particles.
    // int nSteps2 = 1500;
    // VectorXd chkPts = wmatup * nSteps2; // compute points in time when to recompute wt. matrix
    // for(int i = 0; i < chkPts.size(); i++){
    //     int val = chkPts(i);
    //     chkPts(i) = val;// round all values down to integer values
    // }

    // #pragma omp parallel for
    // for(int particle = 0; particle < Nparts2; particle++){
    //     int wasflipped = 0;
    //     double nearby = sdbeta;
    //     // reset weights
    //     double sfp = 3.0, sfg = 1.0, sfe = 6.0; // initial particle historical weight, global weight social, inertial
    //     double sfi = sfe, sfc = sfp, sfs = sfg; // below are the variables being used to reiterate weights
    //     double w1 = 6, w2 = 1, w3 = 1;
    //     random_device pRanDev;
    //     mt19937 pGenerator(pRanDev());
    //     uniform_real_distribution<double> pUnifDist(0.0, 1.0);
    //     struct K pos;
    //     pos.k = VectorXd::Zero(Npars);
    //     VectorXd PBVEC = VectorXd::Zero(Npars);
    //     double pBestCost = 0, cost = 0;

    //     // instantiate new position vector and PBVEC/cost based on GBVEC/gcost using markov like movement 
    //     for(int edim = 0; edim < Npars; edim++){
    //         double tmean = GBVEC(edim);
    //         if (GBVEC(edim) > 0.5) {
    //             tmean = 1 - GBVEC(edim);
    //             wasflipped = 1;
    //         }
    //         double myc = (1 - tmean) / tmean;
    //         double alpha = myc / ((1 + myc) * (1 + myc) * (1 + myc)*nearby*nearby);
    //         double beta = myc * alpha;

    //         std::gamma_distribution<double> aDist(alpha, 1);
    //         std::gamma_distribution<double> bDist(beta, 1);

    //         double x = aDist(pGenerator);
    //         double y = bDist(pGenerator);
    //         double myg = x / (x + y);
    
    //         if (wasflipped == 1) {
    //             wasflipped = 0;
    //             myg = 1 - myg;
    //         }
    //         pos.k(edim) = myg;
    //     }
    //     Protein_Moments XtPSOInit(tf, nMoments);
    //     Nonlinear_ODE6 initSys(pos);
    //     Mom_ODE_Observer XtObsPSOInit(XtPSOInit);
    //     for(int i = 0; i < N; i++){
    //         State_N c0 = gen_multi_norm_iSub();
    //         integrate_adaptive(controlledStepper, initSys, c0, t0, tf, dt, XtObsPSOInit);
    //     }
    //     XtPSOInit.mVec/=N;

    //     // use that to instantiate new PBVEC and respective cost using this value
    //     cost = calculate_cf2(Yt.mVec, XtPSOInit.mVec, wt);
    //     pBestCost = cost;
    //     PBVEC = pos.k;
    //     // re enter for loop of # of steps
    //     for(int step = 0; step < nSteps2; step++){

    //         // at each checkpoint, use global best vector to solve ode system and..
    //         if(step == chkPts(0) || step == chkPts(1)  || step == chkPts(2)  || step == chkPts(3)){
    //             nearby = squeeze * nearby;
    //             struct K gBest;
    //             gBest.k = VectorXd::Zero(Npars);
    //             gBest.k = GBVEC;
    //             Nonlinear_ODE6 chkSys(gBest);
    //             Data_Components chkMoments(tf, nMoments, N);
    //             Data_ODE_Observer chkObs(chkMoments);
    //             for(int i = 0; i < N; i++){
    //                 State_N c0 = gen_multi_norm_iSub();
    //                 integrate_adaptive(controlledStepper, chkSys, c0, t0, tf, dt, chkObs);
    //             }
    //             chkMoments.mVec/=N; // compute moment vectors of new GBbest 
                
    //             //compute a new weight matrix using method discussed with sample and inverse omega matrix - huge bottleneck!
    //             #pragma omp critical
    //             {
    //                 wt = calculate_omega_weight_matrix(chkMoments.mat, Yt.mVec, N);
    //                 double cost = calculate_cf2(Yt.mVec, chkMoments.mVec, wt);    
    //                 //  attach new cost with updated w.mat into GBMAT
    //                 if(cost < gCost){
    //                     gCost = cost;
    //                     GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1);
    //                     for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = GBVEC(i);}
    //                     GBMAT(GBMAT.rows() - 1, Npars) = gCost;
    //                 }   
    //             }
    //             // reuse markov like behavior on setting up position vector again
    //             for(int edim = 0; edim < Npars; edim++){
    //                 double tmean = GBVEC(edim);
    //                 if (GBVEC(edim) > 0.5) {
    //                     tmean = 1 - GBVEC(edim);
    //                     wasflipped = 1;
    //                 }
    //                 double myc = (1 - tmean) / tmean;
    //                 double alpha = myc / ((1 + myc) * (1 + myc) * (1 + myc)*nearby*nearby);
    //                 double beta = myc * alpha;

    //                 std::gamma_distribution<double> aDist(alpha, 1);
    //                 std::gamma_distribution<double> bDist(beta, 1);

    //                 double x = aDist(pGenerator);
    //                 double y = bDist(pGenerator);
    //                 double myg = x / (x + y);
            
    //                 if (wasflipped == 1) {
    //                     wasflipped = 0;
    //                     myg = 1 - myg;
    //                 }
    //                 pos.k(edim) = myg;
    //             }
                
    //             // use new position based on global best, solve odes, recompute update particle best moment vectors and /or cost.
    //             Protein_Moments XtPSOPb(tf, nMoments);
    //             Nonlinear_ODE6 pbSys(pos);
    //             Mom_ODE_Observer XtObsPSOPb(XtPSOPb);
    //             for(int i = 0; i < N; i++){
    //                 State_N c0 = gen_multi_norm_iSub();
    //                 integrate_adaptive(controlledStepper, pbSys, c0, t0, tf, dt, XtObsPSOPb);
    //             }
    //             XtPSOPb.mVec /= N;
    //             pBestCost = calculate_cf2(Yt.mVec, XtPSOPb.mVec, wt);
    //             PBVEC = pos.k; // update particle best

    //         }
    //         // and then redo the above inner code for PSO again.
    //         w1 = sfi * pUnifDist(pGenerator)/ sf2; w2 = sfc * pUnifDist(pGenerator) / sf2; w3 = sfs * pUnifDist(pGenerator)/ sf2;
    //         double sumw = w1 + w2 + w3; //w1 = inertial, w2 = pbest, w3 = gbest
    //         w1 = w1 / sumw; w2 = w2 / sumw; w3 = w3 / sumw;
    //         VectorXd rpoint = comp_vel_vec(pos.k);
    //         pos.k = w1 * rpoint + w2 * PBVEC + w3 * GBVEC; // update position of particle
            
    //         Protein_Moments XtPSO(tf, nMoments);
    //         Nonlinear_ODE6 stepSys(pos);
    //         Mom_ODE_Observer XtObsPSO1(XtPSO);
    //         for(int i = 0; i < N; i++){
    //             State_N c0 = gen_multi_norm_iSub();
    //             integrate_adaptive(controlledStepper, initSys, c0, t0, tf, dt, XtObsPSO1);
    //         }
    //         XtPSO.mVec /= N;
    //         //XtPSO.sec /=N; l
    //         cost = calculate_cf2(Yt.mVec, XtPSO.mVec, wt);
    //         #pragma omp critical
    //         {
    //             if(cost < pBestCost){
    //                 PBVEC = pos.k;
    //                 pBestCost = cost;
    //                 if(cost < gCost){
    //                     gCost = cost;
    //                     GBVEC = pos.k;
    //                     GBMAT.conservativeResize(GBMAT.rows() + 1, Npars + 1);
    //                     for (int i = 0; i < Npars; i++) {GBMAT(GBMAT.rows() - 1, i) = GBVEC(i);}
    //                     GBMAT(GBMAT.rows() - 1, Npars) = gCost;
    //                 }   
    //             }
    //         }
    //         sfi = sfi - (sfe - sfg) / Nsteps;   // reduce the inertial weight after each step 
    //         sfs = sfs + (sfe - sfg) / Nsteps;   
    //     }
    // } 

    // cout << GBMAT << endl;
    ofstream plot;
	plot.open("Jay_Global_Best.txt");
	MatrixXd GBMATWithSteps(GBMAT.rows(), GBMAT.cols() + 1);
	VectorXd globalIterations(GBMAT.rows());
	for(int i = 0; i < GBMAT.rows(); i++){
		globalIterations(i) = i;
	}
	GBMATWithSteps << globalIterations, GBMAT;
	plot << GBMATWithSteps << endl;
	plot.close();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << "CODE FINISHED RUNNING IN " << duration << " s TIME!" << endl;

    return 0; // just to close the program at the end.
}





