#include "main.hpp"
#include "fileIO.hpp"
#include "calc.hpp"
#include "ODE.hpp"

/* Global Vectors/Matrices to be accessible by ODEINT solvers */
/* data module */
// VectorXd mVecTrue = VectorXd::Zero(N_SPECIES*(N_SPECIES + 3) / 2); // moment vector for some t
// MatrixXd m2Mat = MatrixXd::Zero(N_SPECIES, N_SPECIES); // second moment matrices

/* Global Variables to be used for parallel computing */
MatrixXd globalSample = MatrixXd::Zero(N, N_SPECIES);// sample matrix
VectorXd bestMomentVector = VectorXd::Zero( N_SPECIES*(N_SPECIES + 3) / 2); // secomd moment vector 

int main(int argc, char **argv)
{   
    auto t1 = std::chrono::high_resolution_clock::now(); // start time

    /* PSO Parameters */
    int nMom = (N_SPECIES * (N_SPECIES + 3)) / 2; // number of moments
    int nSteps = 20;
    int nPart = 20;
    int nDim = N_DIM;
    double t0 = 0, tf = 5.0 * 9.69, dt = 1.0;
    cout << "sample:" << N << " Nparts:" << nPart << " nsteps:" << nSteps << endl;
    random_device rndDev;
    mt19937 gen(rndDev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);

    /* Instantiate Y_t */
    struct K tru;
    tru.k << 5.0, 0.1, 1.0, 8.69, 0.05, 0.70;
    tru.k /= (9.69); // make sure not so close to the boundary
    tru.k(1) += 0.05; tru.k(4) += 0.05;
    Nonlinear_ODE6 trueSys(tru);
    Protein_Moments Yt(tf, nMom);
    Mom_ODE_Observer YtObs(Yt);
    Controlled_RK_Stepper_N controlledStepper;
    for (int i = 0; i < N; i++) {
        //State_N c0 = gen_multi_norm_iSub(); // Y_0 is simulated using norm dist.
        State_N c0 = {80, 250, 0, 0, 85, 0};
        integrate_adaptive(controlledStepper, trueSys, c0, t0, tf, dt, YtObs);
    }
    Yt.mVec /= N;
    /* Instantiate Seed */
    struct K seed;
    for(int i = 0; i < nDim; i++){
        seed.k(i) = unifDist(gen);
    }
    Protein_Moments Xt(tf, nMom);
    Mom_ODE_Observer XtObs(Xt);
    Nonlinear_ODE6 sys(seed);
    for (int i = 0; i < N; i++) {
        //State_N c0 = gen_multi_norm_iSub();
        State_N c0 = {80, 250, 0, 0, 85, 0};
        integrate_adaptive(controlledStepper, sys, c0, t0, tf, dt, XtObs);
    }
    Xt.mVec /= N;
    /* Instantiate PBMAT */


    /* PSO starts */
    auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << " Code Finished Running in " << duration << " seconds time!" << endl;
}

