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
    /**** PART ONE ****/
    auto t1 = std::chrono::high_resolution_clock::now(); // start time
    int sizeSubset = 3;
    int nMom = (N_SPECIES * (N_SPECIES + 3)) / 2; // number of moments
    
    /* RNG */
    random_device ranDev;
    mt19937 generator(ranDev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);

    /* ODE Vars */
    double t0 = 0.0, tf = 4.0, dt = 1.0, tn = 3.0; // times 
    VectorXd sub = VectorXd::Zero(N_DIM); sub << 1,2,0,0,5,0; // subset of proteins to solve for.
    Controlled_RK_Stepper_N controlled_stepper;

    /* PSO Vars */
    int subSize = 3;
    int subMom = (subSize * (subSize + 3)) / 2;
    MatrixXd cov(subMom, subMom); // covar matrix   
    MatrixXd wt = MatrixXd::Identity(subMom, subMom); // wt. matrix
    MatrixXd gBMat;
    VectorXd gBVec = VectorXd::Zero(N_DIM);
    double gCost = 10000000; // some outrageous starting value

    /* Note: Y_0 - either given or generated by lognorm dist  */

    /* Solve ODEs for Y_t or mu "true" moment vectors using exact rate constants */
    cout << "Computing Y_t" << endl;
    struct K exactK; 
    exactK.k << 5.0, 0.10, 1.00, 8.69, 0.05, 0.70; // true k vector
    Nonlinear_ODE6 ode6Sys(exactK); // ode sys to evolve
    Data_Components Y_t(sub, tf, nMom); // Y_t = mu
    Data_ODE_Observer YtObs6(Y_t); // obs sums over subset of values
    State_N c0 = {120, 250, 0, 0, 80, 0}; //gen_multi_lognorm_init6();
    for(int i = 0; i < N; i++){
        integrate_adaptive(controlled_stepper, ode6Sys, c0, t0, tf, dt, YtObs6);
    }
    Y_t.secondMoments/=N; // average moments
    Y_t.moments /= N;
    VectorXd mu = gen_sub_mom_vec(Y_t.moments); // filter out zero moments due to subset.
    cout << "Y_t moment vector" << mu.transpose() << endl << endl;

    /****************************** Parallel Computing - Particles/PSO ******************************/
    int particle = 0; // private variable used in parallel computing.
    cout << "Parallel Computing Has Started!" << endl << endl;
#pragma omp parallel for
    for(particle = 0; particle < N_PARTICLES; particle++){
       
        int nSteps = 400;
        struct K pos; // particle k vals
        /* rng */
        random_device pRanDev;
        mt19937 pGenerator(pRanDev());
        uniform_real_distribution<double> pUnifDist(0.0, 1.0);
        /* ODE */
        for(int i = 0; i < pos.k.size(); i++){ pos.k(i) = pUnifDist(generator); } 
        Nonlinear_ODE6 pOdeSys(pos);
        Controlled_RK_Stepper_N pControlledStepper;
        Data_Components X_t(sub, tf, nMom); // System for Y_t = mu
        Data_ODE_Observer XtObs6(X_t); // obs sums over subset of values
        double pt0 = t0, ptf = tf, pdt = dt;
        

        /* PSO */
        /* instantiate values before PSO */  
        double w = 1.0, wS = 2.0, wC = 2.0; //  w - inertial weight, cS - social weight 
        double pCurrCost;
        VectorXd pBVec = pos.k; // best particle k rates
        double pBCost; // best cost in particle history
        for(int i = 0; i < N; i++){
            State_N pC0 = gen_multi_lognorm_init6();
            integrate_adaptive(pControlledStepper, pOdeSys, pC0, pt0, ptf, pdt, XtObs6);
        }
        VectorXd pMoments = gen_sub_mom_vec(X_t.moments);
        /* instantiate custom velocity markov component */
        VectorXd vj = comp_vel_vec(pos.k); 
        pCurrCost = calculate_cf1(mu, pMoments, subMom);

        /* Instantiate inertial component aka original velocity vector */
        for(int jjj = 0; jjj < nSteps; jjj++){
            double sumW = w + wS + wC;
            w = w * pUnifDist(generator) / sumW; //redeem weights 
            wS = wS * pUnifDist(generator) / sumW;
            wC = wC * pUnifDist(generator) / sumW;

            pos.k = w * vj + wC * pBVec + wS * gBVec;

            /* cost comparisons */
            #pragma omp critical
            {     
                
            }
        }
        /* 2nd iteration - PSO*/
        /* using CF2 compute next cost function and recompute weight */
    }

    /* Check - Find average of 9 moments of nonlinear 6 system at 3 dif times */
    ofstream average9Moments;
    average9Moments.open("Average_9_Mom_3times.txt");
    Data_Components time_tf(sub, 0.5, nMom); // Y_t = mu
    Data_ODE_Observer tfObs6(time_tf); // obs sums over subset of values
    Data_Components time_5(sub, 1.0, nMom); // Y_t = mu
    Data_ODE_Observer t5Obs6(time_5); // obs sums over subset of values
    Data_Components time_10(sub, 2.0, nMom); // Y_t = mu
    Data_ODE_Observer t10Obs6(time_10); // obs sums over subset of values
    
    for(int i = 0; i < N; i++){
        State_N c0t = gen_multi_lognorm_init6(); //gen_multi_lognorm_init6();
        integrate_adaptive(controlled_stepper, ode6Sys, c0t, t0, 0.5, dt, tfObs6);
        integrate_adaptive(controlled_stepper, ode6Sys, c0t, t0, 1.0, dt, t5Obs6);
        integrate_adaptive(controlled_stepper, ode6Sys, c0t, t0, 2.0, dt, t10Obs6);
    }
    time_tf.secondMoments/=N; // average moments
    time_tf.moments /= N;
    time_5.secondMoments /= N; // average moments
    time_5.moments /= N;
    time_10.secondMoments /= N; // average moments
    time_10.moments /= N;
    average9Moments << "tf 0.5:" << gen_sub_mom_vec(time_tf.moments).transpose() << endl << endl;
    average9Moments << "1.0:" << gen_sub_mom_vec(time_5.moments).transpose()<< endl << endl;
    average9Moments << "2.0:" << gen_sub_mom_vec(time_10.moments).transpose() << endl;


    average9Moments.close();
    auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << " Code Finished Running in " << duration << " seconds time!" << endl;
}

