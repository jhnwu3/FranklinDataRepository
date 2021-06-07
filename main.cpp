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

    /* instantiate data module / true values class */
    Data_Components mTrue;
    mTrue.subset = VectorXd::Zero(SUBSET_SIZE);// subset of values we want to store.
    mTrue.subset << 1,2,3; // store the indices in specific order.
    mTrue.moments = VectorXd::Zero(nMom);
    mTrue.secondMoments = MatrixXd::Zero(N_SPECIES, N_SPECIES);

    /* global file streams to be able to access all files */
    ofstream oFile, oFile1, oFileMAV; 
    open_files(oFile, oFile1, oFileMAV); 
    /* Random Number Generator */
    random_device ranDev;
    mt19937 generator(ranDev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);
    /* Variables used for multivariate log normal distribution */
    VectorXd mu(N_SPECIES);
    MatrixXd sigma  = MatrixXd::Zero(N_SPECIES, N_SPECIES);
    MatrixXd sampleSpace(N, N_SPECIES);
    /* Covariance Matrix to be Calculated from Moments! */
    MatrixXd cov(N_SPECIES, N_SPECIES);
    /* weight matrix */
    MatrixXd w = MatrixXd::Identity(nMom, nMom);
    /* rate constants vectors */
    VectorXd kTrue(nMom);
    VectorXd kEst(nMom);
    VectorXd kEst1(nMom);

    /* Bill's initial k - values in var.cpp*/
    kTrue(0) = k1; kTrue(1) = k2; kTrue(2) = k3; kTrue(3) = k4; kTrue(4) = k5;

    /* Fill with temporary random vals for k vecs, cost testing */
    for(int i = 0; i < nMom; i++){
        if(i > 4){ kTrue(i) = 0; } // generate a random Ktrue even tho will be given later
        kEst(i) = unifDist(generator); 
        kEst1(i) = kTrue(i) + 0.1 * unifDist(generator); // another that differs from exact one by 0.1
    }
    
    /* ODE solver variables! */
    VectorXd initCon(N_SPECIES); // temp vector to be used for initiation conditions

    Controlled_RK_Stepper_N controlled_stepper;
    /* mu vector and covariance (sigma) original values */
    mu << mu_x, mu_y, mu_z;
    sigma << 0.77, 0.0873098, 0.046225, 
                0.0873098, 0.99, 0.104828, 
                0.046225, 0.104828, 1.11; 
    /* Compute mu and covar matrix required for multivar norm dist 
    sampleSpace = generate_sample_space(N_SPECIES, N);
    mu = sampleSpace.colwise().mean(); 
    sigma = create_covariance_matrix(sampleSpace, mu, N_SPECIES);*/
    cout << "mu:" << mu.transpose() << endl << endl << "sigma:" << endl << sigma << endl << endl; 

    /* For Checking Purposes - Graph Vav, p-Vav .., SHP1 */
    Controlled_RK_Stepper_6 controlled_6stepper;
    struct K jayK;
    jayK.k = VectorXd::Zero(N_DIM);
    jayK.k << 5.0, 0.10, 1.00, 8.69, 0.05, 0.07; // write 6 values.
    Nonlinear_ODE6 ODE6System(jayK);
    ofstream gnu;
    gnu.open("NonlinODE6_Syk_Vav_pVav_SHP1.txt"); 
    Write_File_Plot writeFile(gnu);
    State_6 jc0 = {120.0, 41.33, 0, 0, 80.0, 0};
    integrate_adaptive(controlled_6stepper, ODE6System, jc0, t0, tf, dt, writeFile);
    
    /* Now do it for several thousand samples */
    normal_distribution<double> normC1{120.0, 120.0};
    normal_distribution<double> normC2{41.33, 5.0};
    normal_distribution<double> normC5{80.0, 6.0};  
    Data_Components data6;
    data6.subset = VectorXd::Zero(6); data6.subset << 1,2,0,0,5,0;
    data6.moments = VectorXd::Zero(6);
    data6.secondMoments = MatrixXd::Zero(6,6);

    Data_ODE_Observer6 dataOBS6(data6);
    for(int i = 0; i < 3000; i++){
        State_6 nC0 = {normC1(generator), normC2(generator), 0, 0, normC5(generator), 0};
        integrate_adaptive(controlled_6stepper, ODE6System, nC0, t0, tf, dt, dataOBS6); 
    }
    oFile <<"Correlation matrix i.e <ci(t)cj(t)> :"<< endl <<data6.secondMoments << endl;
    oFile <<"Cov mat :" << endl << calculate_covariance_matrix(data6.secondMoments, data6.moments, 6) << endl; 
    //@TODO - need to find a better way to fit multiple eqs
    /* multivar norm gen */
    Multi_Normal_Random_Variable sample{mu, sigma};
    K dataK;
    dataK.k = VectorXd::Zero(N_DIM);
    dataK.k << k1, k2, k3, k4, k5, 0;
    Linear_ODE3 linSys3(dataK); 
    Data_ODE_Observer dataObs(mTrue); // data observer class to fill values in mTrue.
    State_N c0;
    /* Solve for <Ci> using ODE system */
    for(int i = 0; i < N; i++){
       if(i % 1000 == 0){ cout << i << endl;  }
        initCon = sample(); // sample from multilognormal dist
        for(int a = 0; a < N_SPECIES; a++){
            c0[a] = exp(initCon(a)); // assign vector for use in ODE solns.
        }
        integrate_adaptive(controlled_stepper, linSys3, c0, t0, tf, dt, dataObs);
    }
    /* avg for moments */
    mTrue.moments /= N;
    mTrue.secondMoments /= N;
    cov = calculate_covariance_matrix(mTrue.secondMoments, mTrue.moments, N_SPECIES);
    

    /**** parallel computing ****/
    cout << "Parallel Computing Has Started!" << endl << endl;
#pragma omp parallel for
    for(particleIterator = 0; particleIterator < N_PARTICLES; particleIterator++){
        /* first iteration */
        double pCost;
        State_N particleC0; // initial conditions for part
        VectorXd pInit(N_SPECIES); 
        
        Particle_Components pComp; // particle components
        pComp.subset = mTrue.subset;
        pComp.momVec = VectorXd::Zero(nMom);
        pComp.sampleMat = MatrixXd(1, N_SPECIES); // start off with 1 row for initial sample size

        Multi_Normal_Random_Variable sampleParticle{mu, sigma}; 

        /* Generate rate constants from uniform dist (0,1) for 6-dim hypercube */
        struct K pK; // structure for particle rate constants
        pK.k = VectorXd::Zero(N_DIM);
        for(int i = 0; i < N_DIM; i++){
            pK.k(i) = unifDist(generator);                        
        }
        Linear_ODE3 pSys(pK); // instantiate ODE System

        /* solve N-samples of ODEs */
        for(int i = 0; i < N; i++){
            pInit = sampleParticle(); // sample from normal dist
            for(int a = 0; a < N_SPECIES; a++){
                particleC0[a] = exp(pInit(a)); // convert to lognorm
                pInit(a) = particleC0[a];
            }
            integrate_adaptive(controlled_stepper, pSys, particleC0, t0, tf, dt, Particle_Observer(pComp));
        }    
        pComp.momVec /= N; 
        pCost = calculate_cf1(mTrue.moments, pComp.momVec, nMom); // cost

        /* cost comparisons */
        #pragma omp critical
        {     
            if(particleIterator == 0){
                cout << endl << endl << "Writing First Particle data!" << endl << endl;
                write_particle_data(pK.k, pInit, pComp.momVec, mTrue.moments ,pCost);
            }
            cout << "protein moment vector: "<< pComp.momVec.transpose() << "from thread: " << omp_get_thread_num << endl;
            if(pCost < globalCost){
                globalCost = pCost;
                bestMomentVector = pComp.momVec;
                globalSample = pComp.sampleMat;
            }
        }
 
        w = calculate_weight_matrix(pComp.sampleMat, mTrue.moments, nMom, N);  // calc inverse wt. matrix
        /* 2nd iteration - PSO*/
        /* using CF2 compute next cost function and recompute weight */
    }

    cout << "bestMomentVector: " << bestMomentVector.transpose() << endl << endl;
    cout << "Global Best Cost: " << globalCost << endl;

    /**** Print Statements ****/
    cout << "kTrue:" << endl << kTrue.transpose() << endl;
    cout << "kEst between 0 and 1s:" << endl << kEst.transpose() << endl;
    cout << "kEst1 0.1 * rand(0,1) away:" << endl << kEst1.transpose() << endl;
   
    cout << "kCost for a set of k estimates between 0 and 1s: " << calculate_cf1(kTrue, kEst, nMom) << endl;
    cout << "kCost for a set of k estimates 0.1 * rand(0,1) away from true: " << calculate_cf1(kTrue, kEst1, N_SPECIES) << endl << endl;
   
    cout << "kCostMat for a set of k estimates between 0 and 1s: " << calculate_cf2(kTrue, kEst, w, N_SPECIES) << endl;
    cout << "kCostMat for a set of k estimates 0.1 * rand(0,1) away from true: " << calculate_cf2(kTrue, kEst1, w, N_SPECIES) << endl << endl;
    /* Print statement for the moments */
    oFileMAV << "2nd moment matrix:" << endl;
    oFileMAV << mTrue.secondMoments << endl << endl;
    cout << "2nd moment matrix:" << endl;
    cout << mTrue.secondMoments << endl << endl;

    oFileMAV << "Full " << N_SPECIES << "protein moment vector" << endl;
    oFileMAV << mTrue.moments.transpose() << endl;
    cout << "Full " << N_SPECIES << " protein moment vector" << endl;
    cout << mTrue.moments.transpose() << endl;

    oFileMAV << "Cov Matrix" << endl << cov << endl;
    cout << "Cov Matrix:" << endl << cov << endl;

    close_files(oFile, oFile1, oFileMAV); 

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