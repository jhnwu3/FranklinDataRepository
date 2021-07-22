/* This is a test cpp script to test out openMP and show how to use it to solve ODEs, it solves a simple 3 linear ODE system that we eventually use in the actual ODE.cpp, which
may or may not be renamed to main.cpp at some point. 
 */

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
#define N 1500 // # of samples to sample over
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

void doStuff(){
    for(int j = 0; j< 20000; j++){

    }
}
struct Write_File_Plot 
{
    ostream& fOut;
    Write_File_Plot (ostream& out) : fOut( out ) {} 
    void operator()(const State_N &c, const double t){ // write all solved ODE values in GNU plot vals
        fOut << t;
        for(int i = 0; i < 6; i++){
        fOut << "," << c[i];
        }
        fOut << endl;
    }
}; 
struct pVav_Plot 
{
    ostream& fOut;
    pVav_Plot (ostream& out) : fOut( out ) {} 
    void operator()(const State_N &c, const double t){ // write all solved ODE values in GNU plot vals
        fOut << t;
        fOut << "," << c[3];
        fOut << endl;
    }
}; 
struct Syk_Pvav_Plot 
{
    ostream& fOut;
    double tf;
    Syk_Pvav_Plot (ostream& out, double tf1) : fOut( out ), tf ( tf1) {} 
    void operator()(const State_N &c, const double t){ // write all solved ODE values in GNU plot vals
        // fOut << t;
        // for(int i = 0; i < 6; i++){
        // fOut << "," << c[i];
        // }
        // fOut << endl;
        if(t == 0){
            fOut << c[0];
        }
        if(t == tf){ fOut << "," << c[3] << endl; }
    }
}; 

/* Test finding min function */
int main (){
    double mu_x = 1.47, mu_y = 1.74, mu_z = 1.99; // true means for MVN(theta)
    // ode vars
    int nDim = 6;
    int nMoments = (N_SPECIES * (N_SPECIES + 3)) / 2;
    double t0 = 0.0, tf = 50.0, dt = 1.0;
    struct K tru;
    tru.k = VectorXd::Zero(nDim);
    tru.k << 5.0, 0.1, 1.0, 8.69, 0.05, 0.70;
    tru.k /= (9.69);
    MatrixXd wt = MatrixXd::Identity(nMoments, nMoments); // wt matrix
    Controlled_RK_Stepper_N controlledStepper;
    Nonlinear_ODE6 trueSys(tru);
    Protein_Moments Yt(tf, nMoments);
    Mom_ODE_Observer YtObs(Yt);
    for (int i = 0; i < N; i++) {
        //State_N c0 = gen_multi_norm_iSub(); // Y_0 is simulated using norm dist.
        State_N c0 = {80, 250, 0, 0, 85, 0};
        integrate_adaptive(controlledStepper, trueSys, c0, t0, tf, dt, YtObs);
    }
    Yt.mVec /= N;
    cout << "Yt:" << Yt.mVec.transpose() << endl;

    /* Random Number Generator */
    random_device rand_dev;
    mt19937 generator(rand_dev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);
    std::normal_distribution<double> norm(120.0, 120.0);

    ofstream costOut;
    
    int numDataPts = 100;
    int sampleSize = 10;
    double alpha = 0.001;
    string s =  to_string(alpha) + "RateDist_vs_Cost.csv";
    costOut.open(s);
    for(int pt = 0; pt < numDataPts; pt++){
        struct K pos;
        pos.k = VectorXd::Zero(nDim);
        for(int i = 0; i < nDim; i++) { pos.k(i) = tru.k(i) + alpha * (0.5 - unifDist(generator)); }
        double kCost = calculate_cf1(tru.k, pos.k);
        Protein_Moments Xt(tf, nMoments);
        Mom_ODE_Observer XtObs(Xt);
        Nonlinear_ODE6 sys(pos);
        Controlled_RK_Stepper_N controlledStepper1;
        for(int s = 0; s < sampleSize; s++){
            State_N c0 = {80, 250, 0, 0, 85, 0};
            integrate_adaptive(controlledStepper1, sys, c0, t0, tf, dt, XtObs);
        }
        Xt.mVec /= sampleSize;
        double cost = calculate_cf2(Yt.mVec, Xt.mVec, wt);
        costOut << kCost << "," << cost << endl;
    }
    struct K pos;
    pos.k = VectorXd::Zero(nDim);
    pos.k << 0.515694, 0.0607786, 0.103353, 0.897172, 0.05473, 0.690204; 

    Protein_Moments Xt(tf, nMoments);
    Mom_ODE_Observer XtObs(Xt);
    Nonlinear_ODE6 sys(pos);
    Controlled_RK_Stepper_N controlledStepper1;
    for(int s = 0; s < sampleSize; s++){
        State_N c0 = {80, 250, 0, 0, 85, 0};
        integrate_adaptive(controlledStepper1, sys, c0, t0, tf, dt, XtObs);
    }
    Xt.mVec/=sampleSize;
    double kCost = calculate_cf1(tru.k, pos.k);
    double cost = calculate_cf2(Yt.mVec, Xt.mVec, wt);
    costOut << kCost << "," << cost << endl;
    costOut.close();
    //  /* ODE solver variables! */
    // ofstream baseOut;
    // baseOut.open("baseConc.csv");
    // Write_File_Plot baseCsv(baseOut);
    // State_N c0_base = {120 ,41.33, 0, 0, 80, 0}; // baseline
    // integrate_adaptive(controlledStepper, trueSys, c0_base, t0, tf, dt, baseCsv);
    //    baseOut.close();
    // ofstream highOut;
    // highOut.open("highConc.csv");
    // Write_File_Plot hiCsv(highOut);
    // State_N c0_high = {10000, 41.33, 0, 0, 80, 0};
    // integrate_adaptive(controlledStepper, trueSys, c0_high, t0, tf, dt, hiCsv);
    // highOut.close();

    // ofstream lowOut;
    // lowOut.open("lowConc.csv");
    // Write_File_Plot loCsv(lowOut);
    // State_N c0_low = {20, 41.33, 0, 0, 80, 0}; // lowered
    // integrate_adaptive(controlledStepper, trueSys, c0_low, t0, tf, dt, loCsv);
    // lowOut.close();

    // int runs = 10;
    // for(int i = 0; i < runs; i++){
    //     State_N c0 = {(norm(generator)), 41.33, 0, 0, 80, 0};
    //     string s = to_string(i) + "Protein_Concentrations.csv";
    //     ofstream fout;
    //     fout.open(s);
    //     Write_File_Plot obsCsv(fout);
    //     integrate_adaptive(controlledStepper, trueSys, c0, t0, tf, dt, obsCsv);
    //     fout.close();
    // }

    // ofstream fOut; 
    // fOut.open("syk_pVav.csv");
    // Syk_Pvav_Plot fPlot(fOut, tf);
    // for(int i = 0; i < 100; i++){
    //     State_N c0 = { (double) i, 250.0, 0, 0, 85, 0};
    //     integrate_adaptive(controlledStepper, trueSys, c0, t0, tf, dt, fPlot);
    // }
    // fOut.close();

    // ofstream plot;
    // plot.open("Syk60.csv");
    // pVav_Plot obs(plot);
    // State_N sykC0 = { 60, 250.0, 0, 0, 85, 0};
    // integrate_adaptive(controlledStepper, trueSys, sykC0, t0, tf, dt, obs);
    // plot.close();

    // ofstream plot1;
    // plot1.open("Syk80.csv");
    // pVav_Plot obs1(plot1);
    // sykC0 = {80, 250.0, 0, 0, 85, 0};
    // integrate_adaptive(controlledStepper, trueSys, sykC0, t0, tf, dt, obs1);
    // plot1.close();

    // ofstream plot2;
    // plot2.open("Syk100.csv");
    // pVav_Plot obs2(plot2);
    // sykC0 = { 100, 250.0, 0, 0, 85, 0};
    // integrate_adaptive(controlledStepper, trueSys, sykC0, t0, tf, dt, obs2);
    // plot2.close();

    return EXIT_SUCCESS;
}