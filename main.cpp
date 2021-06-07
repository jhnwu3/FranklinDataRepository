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
MatrixXd w = MatrixXd::Identity( (N_SPECIES * (N_SPECIES + 3)) / 2,  (N_SPECIES * (N_SPECIES + 3)) / 2); // Global Weight/Identity Matrix, nMoments x nMoments
double globalCost = 10000000; // some outrageous starting value
int particleIterator = 0;

int main(int argc, char **argv)
{   
    /**** PART ONE ****/
    auto t1 = std::chrono::high_resolution_clock::now(); // start time
    int nMom = (N_SPECIES * (N_SPECIES + 3)) / 2; // number of moments
    
    /* RNG */
    random_device ranDev;
    mt19937 generator(ranDev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);

    /* instantiate data module / true values class */
    // Data_Components mTrue;
    // mTrue.subset = VectorXd::Zero(N_SPECIES);// subset of values we want to store.
    // mTrue.subset << 1,2,3; // store the indices in specific order.
    // mTrue.moments = VectorXd::Zero(nMom);
    // mTrue.secondMoments = MatrixXd::Zero(N_SPECIES, N_SPECIES);

    /* triple file streams for 3 times */
    ofstream oFile, oFile1, oFile2; 
    open_files(oFile, oFile1, oFile2); 

    /* Variables used for multivariate log normal distribution */
    VectorXd mu(N_SPECIES);
    MatrixXd sigma  = MatrixXd::Zero(N_SPECIES, N_SPECIES);
    MatrixXd sampleSpace(N, N_SPECIES);
    /* Covariance Matrix to be Calculated from Moments! */
    MatrixXd cov(N_SPECIES, N_SPECIES);
    /* weight matrix */
    MatrixXd w = MatrixXd::Identity(nMom, nMom);

    /* ODE solver variables! */
    VectorXd initCon(N_SPECIES); // temp vector to be used for initiation conditions

    Controlled_RK_Stepper_N controlled_stepper;
    /* mu vector and covariance (sigma) original values */
    // mu << mu_x, mu_y, mu_z;
    // sigma << 0.77, 0.0873098, 0.046225, 
    //             0.0873098, 0.99, 0.104828, 
    //             0.046225, 0.104828, 1.11; 
    /* Compute mu and covar matrix required for multivar norm dist 
    sampleSpace = generate_sample_space(N_SPECIES, N);
    mu = sampleSpace.colwise().mean(); 
    sigma = create_covariance_matrix(sampleSpace, mu, N_SPECIES);*/
    //cout << "mu:" << mu.transpose() << endl << endl << "sigma:" << endl << sigma << endl << endl; 

    cout << "beginning to do nonlinear6 for given parameters!" << endl;
    /* For Checking Purposes - Graph Vav, p-Vav .., SHP1 */
    Controlled_RK_Stepper_6 controlled_6stepper;
    struct K jayK;
    jayK.k = VectorXd::Zero(N_DIM);
    jayK.k << 5.0, 0.10, 1.00, 8.69, 0.05, 0.07; // given rate constants
    Nonlinear_ODE6 ODE6System(jayK);
    ofstream gnu;
    gnu.open("NonlinODE6_Syk_Vav_pVav_SHP1.txt"); 
    Write_File_Plot writeFile(gnu);
    State_N jc0 = {120.0, 41.33, 0, 0, 80.0, 0};
    integrate_adaptive(controlled_6stepper, ODE6System, jc0, t0, tf, dt, writeFile);
    
    /* Calculate cov and cor matrix for several thousand samples at 3 different times */
    normal_distribution<double> normC1{120.0, 120.0};
    normal_distribution<double> normC2{41.33, 5.0};
    normal_distribution<double> normC5{80.0, 6.0};  
    Data_Components data6;
    Data_Components data6T2;
    Data_Components data6T3;
    data6.subset = VectorXd::Zero(N_DIM); data6.subset << 1,2,0,0,5,0;
    data6.moments = VectorXd::Zero(nMom);
    data6.secondMoments = MatrixXd::Zero(N_SPECIES, N_SPECIES);
    
    data6T2 = data6;
    data6T3 = data6;
    data6.timeToRecord = tf; 
    data6T2.timeToRecord = 1.0;
    data6T3.timeToRecord = 5.0;
    Data_ODE_Observer dataOBS6(data6);
    Data_ODE_Observer dataOBS6T2(data6T2);
    Data_ODE_Observer dataOBS6T3(data6T3);

    cout << "Data Module: Beginning to solve nonlinear6 for 10000 samples at 3 times!" << endl;
    for(int i = 0; i < N; i++){
        State_N nC0 = {normC1(generator), normC2(generator), 0, 0, normC5(generator), 0};
        State_N nC01 = {normC1(generator), normC2(generator), 0, 0, normC5(generator), 0};
        State_N nC02 = {normC1(generator), normC2(generator), 0, 0, normC5(generator), 0};
        integrate_adaptive(controlled_6stepper, ODE6System, nC0, t0, tf, dt, dataOBS6); 
        integrate_adaptive(controlled_6stepper, ODE6System, nC01, t0, 1.0, dt, dataOBS6T2); 
        integrate_adaptive(controlled_6stepper, ODE6System, nC02, t0, 5.0, dt, dataOBS6T3); 
    }
    data6.moments /= N;
    data6.secondMoments /= N;
    data6T2.moments /= N;
    data6T2.secondMoments /= N;
    data6T3.moments /= N;
    data6T3.secondMoments /= N;
    
    cout << "calculating cov matrices!" << endl;
    /* use for Cost Function below ~ tf */
    cout <<"Correlation matrix i.e <ci(t)cj(t)> :"<< endl << data6.secondMoments << endl;
    oFile <<"Correlation matrix i.e <ci(t)cj(t)> :"<< endl << data6.secondMoments << endl;
    cov = calculate_covariance_matrix(data6.secondMoments, data6.moments, N_SPECIES);
    oFile <<"Cov mat :" << endl << cov << endl; 
    cout << "tf Cov Mat:" << endl << cov << endl;
    cout << endl << "Test:" << endl << data6T2.secondMoments << endl << endl;
    oFile1 <<"Correlation matrix i.e <ci(t)cj(t)> tf = 1.0:"<< endl << data6T2.secondMoments << endl;
    oFile1 <<"Cov Mat :"<< endl << calculate_covariance_matrix(data6T2.secondMoments, data6T2.moments, N_SPECIES) << endl;
    oFile2 <<"Correlation matrix i.e <ci(t)cj(t)> tf = 5.0:"<< endl << data6T3.secondMoments << endl;
    oFile2 <<"Cov Mat :"<< endl << calculate_covariance_matrix(data6T3.secondMoments, data6T3.moments, N_SPECIES) << endl;

    /**** parallel computing ****/
    cout << "Parallel Computing Has Started!" << endl << endl;
#pragma omp parallel for
    for(particleIterator = 0; particleIterator < N_PARTICLES; particleIterator++){
        /* first iteration */
        double pCost;
        State_N particleC0; // initial conditions for part
        VectorXd pInit(N_SPECIES); 
        
        Particle_Components pComp; // particle components
        pComp.subset = data6.subset;
        pComp.momVec = VectorXd::Zero(nMom);
        pComp.sampleMat = MatrixXd(1, N_SPECIES); // start off with 1 row for initial sample size
        pComp.timeToRecord = tf;
        Multi_Normal_Random_Variable sampleParticle{mu, sigma}; 

        /* Generate rate constants from uniform dist (0,1) for 6-dim hypercube */
        struct K pK; // structure for particle rate constants
        pK.k = VectorXd::Zero(N_DIM);
        for(int i = 0; i < N_DIM; i++){
            pK.k(i) = unifDist(generator);                        
        }
        if(particleIterator == 0){
            pK.k = jayK.k; // @Check, for first module, use exact rates constants specified for n=6
        }
        Nonlinear_ODE6 pSys(pK); // instantiate ODE System

        /* solve N-samples of ODEs */
        // for(int i = 0; i < N; i++){
        //     pInit = sampleParticle(); // sample from normal dist
        //     for(int a = 0; a < N_SPECIES; a++){
        //         particleC0[a] = exp(pInit(a)); // convert to lognorm
        //         pInit(a) = particleC0[a];
        //     }
        //     integrate_adaptive(controlled_stepper, pSys, particleC0, t0, tf, dt, Particle_Observer(pComp));
        // }  
        for(int i = 0; i < N; i++){
            particleC0 = {normC1(generator), normC2(generator), 0, 0, normC5(generator), 0};
            integrate_adaptive(controlled_stepper, pSys, particleC0, t0, tf, dt, Particle_Observer(pComp));
        }  
        pComp.momVec /= N; 
        pCost = calculate_cf1(data6.moments, pComp.momVec, nMom); // cost

        /* cost comparisons */
        #pragma omp critical
        {     
            if(particleIterator == 0){
                cout << endl << endl << "Writing First Particle data!" << endl << endl;
                write_particle_data(pK.k, pInit, pComp.momVec, data6.moments ,pCost);
            }
            cout << "protein moment vector: "<< pComp.momVec.transpose() << "from thread: " << omp_get_thread_num << endl;
            if(pCost < globalCost){
                globalCost = pCost;
                bestMomentVector = pComp.momVec;
                globalSample = pComp.sampleMat;
            }
        }
 
        w = calculate_weight_matrix(pComp.sampleMat, data6.moments, nMom, N);  // calc inverse wt. matrix
        /* 2nd iteration - PSO*/
        /* using CF2 compute next cost function and recompute weight */
    }

    cout << "bestMomentVector: " << bestMomentVector.transpose() << endl << endl;
    cout << "Global Best Cost: " << globalCost << endl;

    close_files(oFile, oFile1, oFile2); 

    auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << " Code Finished Running in " << duration << " seconds time!" << endl;
}

/* examples of integrate functions: */
// integrate(tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);
// integrate_adaptive(controlled_stepper, tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);

/* mu vector and covariance (sigma) original values
mu << mu_x, mu_y, mu_z;
sigma << 0.77, 0.0873098, 0.046225, 
            0.0873098, 0.99, 0.104828, 
            0.046225, 0.104828, 1.11; */

/* Og method of pulling from multilognorm dist */
// //@TODO - need to find a better way to fit multiple eqs
// /* multivar norm gen */
// Multi_Normal_Random_Variable sample{mu, sigma};
// K dataK;
// dataK.k = VectorXd::Zero(N_DIM);
// dataK.k << k1, k2, k3, k4, k5, 0;
// Linear_ODE3 linSys3(dataK); 
// Data_ODE_Observer dataObs(mTrue); // data observer class to fill values in mTrue.
// State_N c0;
// /* Solve for moments using ODE system */
// for(int i = 0; i < N; i++){
//    if(i % 1000 == 0){ cout << i << endl;  }
//     initCon = sample(); // sample from multilognormal dist
//     for(int a = 0; a < N_SPECIES; a++){
//         c0[a] = exp(initCon(a)); // assign vector for use in ODE solns.
//     }
//     integrate_adaptive(controlled_stepper, linSys3, c0, t0, tf, dt, dataObs);
// }
// /* avg for moments */
// mTrue.moments /= N;
// mTrue.secondMoments /= N;
// cov = calculate_covariance_matrix(mTrue.secondMoments, mTrue.moments, N_SPECIES);