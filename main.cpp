#include "main.hpp"
#include "fileIO.hpp"
#include "calc.hpp"
#include "ODE.hpp"

/* Global Vectors/Matrices to be accessible by ODEINT solvers */
/* data module */
VectorXd mVecTrue = VectorXd::Zero(N_SPECIES*(N_SPECIES + 3) / 2); // moment vector for some t
MatrixXd m2Mat = MatrixXd::Zero(N_SPECIES, N_SPECIES); // second moment matrices

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

    /* global file streams to be able to access all files */
    ofstream oFile, oFile1, oFileMAV; 
    open_files(oFile, oFile1, oFileMAV); 
    /* Random Number Generator */
    random_device rand_dev;
    mt19937 generator(rand_dev());
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
    state_type c0;
    controlled_stepper_type controlled_stepper;
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

    /* multivar norm gen */
    normal_random_variable sample{mu, sigma};

    /* Solve for <Ci> using ODE system */
    for(int i = 0; i < N; i++){
       if(i % 1000 == 0){ cout << i << endl;  }
        initCon = sample(); // sample from multilognormal dist
        for(int a = 0; a < N_SPECIES; a++){
            c0[a] = exp(initCon(a)); // assign vector for use in ODE solns.
        }
        integrate_adaptive(controlled_stepper, linearODE3_true, c0, t0, tf, dt, sample_adapt);
    }
    /* avg for moments */
    mVecTrue /= N;
    m2Mat /= N;
  
    /* Fill moment vector with diagonals and unique values of the matrix */
    for(int i = 0; i < N_SPECIES; i++){ mVecTrue(N_SPECIES + i) = m2Mat.diagonal()(i); }
    for(int row = 0; row < N_SPECIES - 1; row++){
        for(int col = row + 1; col < N_SPECIES; col++){
            mVecTrue(2*N_SPECIES + (row + col - 1)) = m2Mat(row, col);
        }
    }
    cov = calculate_covariance_matrix(m2Mat, mVecTrue, N_SPECIES);
    

    /**** parallel computing ****/
    cout << "Parallel Computing Has Started!" << endl << endl;
#pragma omp parallel for
    for(particleIterator = 0; particleIterator < N_PARTICLES; particleIterator++){
        /* first iteration */
        double pCost;
        state_type particleC0; // initial conditions for part
        VectorXd pInit(N_SPECIES); 
        
        Particle_Components pComp; // particle components
        pComp.momVec = VectorXd::Zero(nMom);
        pComp.sampleMat = MatrixXd(1, N_SPECIES); // start off with 1 row for initial sample size

        normal_random_variable sampleParticle{mu, sigma}; 

        /* Generate rate constants from uniform dist (0,1) for 6-dim hypercube */
        struct K kS; // structure for particle rate constants
        kS.k = VectorXd::Zero(N_DIM);
        for(int i = 0; i < N_DIM; i++){
            kS.k(i) = unifDist(generator);                        
        }
        Particle_Linear sys(kS); // instantiate ODE System

        /* solve N-samples of ODEs */
        for(int i = 0; i < N; i++){
            pInit = sampleParticle(); // sample from normal dist
            for(int a = 0; a < N_SPECIES; a++){
                particleC0[a] = exp(pInit(a)); // convert to lognorm
                pInit(a) = particleC0[a];
            }
            integrate_adaptive(controlled_stepper, sys, particleC0, t0, tf, dt, Particle_Observer(pComp));
        }    
        pComp.momVec /= N; 
        pCost = CF1(mVecTrue, pComp.momVec, nMom); // cost

        /* cost comparisons */
        #pragma omp critical
        {     
            if(particleIterator == 0){
                cout << endl << endl <<"Writing First Particle data!" << endl << endl;
                ofstream oParticle;
                oParticle.open("First_Particle.txt");
                write_particle_data(oParticle, kS.k, pInit, pComp.momVec, pCost);
                oParticle.close();
            }
            cout << "protein moment vector: "<< pComp.momVec.transpose() << "from thread: " << omp_get_thread_num << endl;
            if(pCost < globalCost){
                globalCost = pCost;
                bestMomentVector = pComp.momVec;
                globalSample = pComp.sampleMat;
            }
        }
 
        w = calculate_weight_matrix(pComp.sampleMat, mVecTrue, nMom, N);  // calc inverse wt. matrix
        /* 2nd iteration - PSO*/
        /* using CF2 compute next cost function and recompute weight */
    }

    cout << "bestMomentVector: " << bestMomentVector.transpose() << endl << endl;
    cout << "Global Best Cost: " << globalCost << endl;

    /**** Print Statements ****/
    cout << "kTrue:" << endl << kTrue.transpose() << endl;
    cout << "kEst between 0 and 1s:" << endl << kEst.transpose() << endl;
    cout << "kEst1 0.1 * rand(0,1) away:" << endl << kEst1.transpose() << endl;
   
    cout << "kCost for a set of k estimates between 0 and 1s: " << CF1(kTrue, kEst, nMom) << endl;
    cout << "kCost for a set of k estimates 0.1 * rand(0,1) away from true: " << CF1(kTrue, kEst1, N_SPECIES) << endl << endl;
   
    cout << "kCostMat for a set of k estimates between 0 and 1s: " << CF2(kTrue, kEst, w, N_SPECIES) << endl;
    cout << "kCostMat for a set of k estimates 0.1 * rand(0,1) away from true: " << CF2(kTrue, kEst1, w, N_SPECIES) << endl << endl;
    /* Print statement for the moments */
    oFileMAV << "2nd moment matrix:" << endl;
    oFileMAV << m2Mat << endl << endl;
    cout << "2nd moment matrix:" << endl;
    cout << m2Mat << endl << endl;

    oFileMAV << "Full " << N_SPECIES << "protein moment vector" << endl;
    oFileMAV << mVecTrue.transpose() << endl;
    cout << "Full " << N_SPECIES << " protein moment vector" << endl;
    cout << mVecTrue.transpose() << endl;

    oFileMAV << "Cov Matrix" << endl << cov << endl;
    cout << "Cov Matrix:" << endl << cov << endl;

    close_files(oFile, oFile1, oFileMAV); 

    auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << " Code Finished Running in " << duration << " seconds time!" << endl;
}



/**** ODE-INT OBSERVER FUNCTIONS ****/
/* const integrations observer funcs */
void sample_const( const state_type &c , const double t){
    if(t == tn){       // sample for some time
        for(int row = 0; row < N_SPECIES; row++){
            mVecTrue(row) += c[row]; // store all first moments in the first part of the moment vec
            for(int col = row; col < N_SPECIES; col++){
                m2Mat(row,col) += (c[row] * c[col]);   // store in a 2nd moment matrix
            }
        }
    }
    if( c[0] - c[1] < 1e-10 && c[0] - c[1] > -1e-10){
        cout << "Out of bounds!" << endl;
        return; // break out for loop
    }
}
/* adaptive observer func */
void sample_adapt( const state_type &c , const double t){
    /* We will have some time we are sampling towards */
    if(t == tf){
        for(int row = 0; row < N_SPECIES; row++){
            mVecTrue(row) += c[row]; // store all first moments in the first part of the moment vec
            for(int col = row; col < N_SPECIES; col++){
                m2Mat(row,col) += (c[row] * c[col]);   // store in a 2nd moment matrix
                m2Mat(col,row) = m2Mat(row,col);   // store in a 2nd moment matrix
            }
        }
    }
    if( c[0] - c[1] < 1e-10 && c[0] - c[1] > -1e-10){
        cout << "Out of bounds!" << endl;
        return; // break out for loop
    }
}
/* Only to be used with integrate/integrate_adaptive - linear */
void sample_adapt_linear( const state_type &c , const double t){
    /* We will have some time we are sampling towards */
    if(t == tf){
        for(int row = 0; row < N_SPECIES - 1; row++){
            mVecTrue(row) += c[row]; // store all first moments in the first part of the moment vec
            for(int col = row; col < N_SPECIES - 1; col++){
                m2Mat(row,col) += (c[row] * c[col]);   // store in a 2nd moment matrix
                m2Mat(col,row) = m2Mat(row,col);   // store in a 2nd moment matrix
            }
        }
    }
    if( c[0] - c[1] < 1e-10 && c[0] - c[1] > -1e-10){
        cout << "Out of bounds!" << endl;
        return; // break out for loop
    }
}

/* examples of integrate functions: */
// integrate(tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);
// integrate_adaptive(controlled_stepper, tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);

/* mu vector and covariance (sigma) original values
mu << mu_x, mu_y, mu_z;
sigma << 0.77, 0.0873098, 0.046225, 
            0.0873098, 0.99, 0.104828, 
            0.046225, 0.104828, 1.11; */