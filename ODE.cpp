#include "ODE.hpp"

/* global file streams to be able to access all files */
ofstream oFile; 
ofstream oFile1; 
ofstream oFileMAV; 

/* Global Vectors/Matrices to be accessible by ODEINT solvers */
/* data module */
/* moment vector */
VectorXd mVecTrue = VectorXd::Zero(N_SPECIES*(N_SPECIES + 3) / 2); // for some t
/* Second moment matrix. */
MatrixXd m2Mat = MatrixXd::Zero(N_SPECIES, N_SPECIES); // secomd moment vector


/* Variables to be used for parallel computing*/
MatrixXd xs = MatrixXd::Zero(N, N_SPECIES);// sample matrix
VectorXd bestMomentVector = VectorXd::Zero( N_SPECIES*(N_SPECIES + 3) / 2); // secomd moment vector 
MatrixXd w = MatrixXd::Identity( (N_SPECIES * (N_SPECIES + 3)) / 2,  (N_SPECIES * (N_SPECIES + 3)) / 2); // Global Weight/Identity Matrix, nMoments x nMoments
double globalCost = 10000000; // some outrageous starting value
int particleIterator = 0;

/**** ODE-INT OBSERVER FUNCTIONS ****/
/* Only to be used with integrate_const(), solves the ODE's defined in ODESys.cpp*/
void sample_const( const state_type &c , const double t){
    /* We will have some time we are sampling for */
    if(t == tn){       
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
/* Only to be used with integrate/integrate_adaptive - nonlinear */
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

int main(int argc, char **argv)
{   

    /************ FIRST PART DATA MODULE INITIALLY *********/
    auto t1 = std::chrono::high_resolution_clock::now(); // start time
    int nMom = (N_SPECIES * (N_SPECIES + 3)) / 2; // number of moments

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
    /* Bill's initial k*/
    kTrue(0) = k1;
    kTrue(1) = k2;
    kTrue(2) = k3;
    kTrue(3) = k4;
    kTrue(4) = k5;
    /* Fill with temporary random vals for k vecs, cost testing */
    for(int i = 0; i < nMom; i++){
        if(i > 4){
            kTrue(i) = 0; // generate a random Ktrue even tho will be given
        }
        kEst(i) = unifDist(generator); // one random between 0 and 1
        kEst1(i) = kTrue(i) + 0.1 * unifDist(generator); // another that differs from exact one by 0.1
    }
    
    /* ODE solver variables! */
    VectorXd initCon(N_SPECIES); // temp vector to be used for initiation conditions
    state_type c0;
    controlled_stepper_type controlled_stepper;

    /* assign mu vector and sigma matrix values   
    mu << mu_x, mu_y, mu_z;
    sigma << 0.77, 0.0873098, 0.046225, 
             0.0873098, 0.99, 0.104828, 
             0.046225, 0.104828, 1.11; */
    
      /* Calculate averages */
    sampleSpace = generate_sample_space(N_SPECIES, N);
    mu = sampleSpace.colwise().mean(); 
    /* Calculate covar matrix labeled sigma */
    sigma = create_covariance_matrix(sampleSpace, mu, N_SPECIES);
    cout << "mu:" << mu.transpose() << endl << endl << "sigma:" << endl << sigma << endl << endl; 
    /* multivariate /normal distribution generator */
    normal_random_variable sample{mu, sigma};
    open_files(); 

    /* average randomized sample/initial conditions from unif dist, N=10,000, CALL ODE SOLVER HERE! */
   for(int i = 0; i < N; i++){
       if(i % 1000 == 0){ cout << i << endl;  }
        initCon = sample(); // sample from multilognormal dist
        for(int a = 0; a < N_SPECIES; a++){
            c0[a] = exp(initCon(a)); // assign vector for use in ODE solns.
        }
        integrate_adaptive(controlled_stepper, linearODE3_true, c0, t0, tf, dt, sample_adapt);
   }

     /* Divide the sums at the end to reduce number of needed division ops */
    mVecTrue /= N;
    m2Mat /= N;
  
    /* Fill moment vector with diagonals and unique values of the matrix */
    for(int i = 0; i < N_SPECIES; i++){
        mVecTrue(N_SPECIES + i) = m2Mat.diagonal()(i);
    }
    for(int row = 0; row < N_SPECIES - 1; row++){
        for(int col = row + 1; col < N_SPECIES; col++){
            mVecTrue(2*N_SPECIES + (row + col - 1)) = m2Mat(row, col);
        }
    }
    cov = calculate_covariance_matrix(m2Mat, mVecTrue, N_SPECIES);

    /*******************************************************/
    VectorXd kFinal(5);
    
    /* parallel computing */
    cout << "Parallel Computing starts here!" << endl << endl;
    //  #pragma omp parallel for
    //  {
    //     for(particleIterator = 0; particleIterator < N_PARTICLES; particleIterator++){
            /* variables */
            int nIter = 2;
            double pCost;
            state_type particleC0;
            struct K kParticle; // structure for particle rate constants
            kParticle.k = VectorXd::Zero(N_DIM);
            VectorXd initConditions(N_SPECIES);
            Particle_Components pComp;
            pComp.momentVector = VectorXd::Zero(nMom);
            pComp.sampleMat = MatrixXd(1, N_SPECIES); // start off with 1 row for initial sample size
            normal_random_variable sampleParticle{mu, sigma}; // placed input
            /* 2 iterations for each particle module */
            /* Generate rate constants from uniform dist (0,1) for 5-dim hypercube */
            for(int i = 0; i < N_DIM; i++){
                kParticle.k(i) = unifDist(generator);                        
            }
            cout << "k rate vector generated: " << kParticle.k.transpose() << endl;
            Particle_Linear sys(kParticle); // plug rate constants into ode sys to solve
            /* solve ODEs for fixed number of samples using ODEs, use linearODE3 sys for now & compute moments. */
            for(int i = 0; i < N; i++){
                initConditions = sampleParticle(); // sample from multilognormal dist
                for(int a = 0; a < N_SPECIES; a++){
                    particleC0[a] = exp(initConditions(a)); // assign vector for use in ODE solns.
                }
                integrate_adaptive(controlled_stepper, sys, particleC0, t0, tf, dt, Particle_Observer(pComp));
            }
            
            pComp.momentVector /= N; 
            cout <<"mvec: " << pComp.momentVector.transpose() << endl<<endl;
            cout << "sampleMat: "<< endl << pComp.sampleMat << endl << endl;
            pCost = CF1(mVecTrue, pComp.momentVector, nMom);
            
            /* do cost comparisons with global cost using a 1 thread at a time to make sure to properly update global values*/
            // #pragma omp critical
            // {   
            //     cout << "protein moment vector: "<< pComp.momentVector.transpose() << "from thread: " << omp_get_thread_num << endl;
            //     if(pCost < globalCost){
            //         globalCost = pCost;

            //     }
            // }
            /* Calculate CF1 for moments */ 

            /* Calculate inverse weight matrix */
                
    //     }
    //  }
    cout << "Global Best Cost: " << globalCost << endl;
    /* 2nd iteration */
    /* using CF2 compute next cost function and recompute weight */


    /***** printf statements ******/
    /* Print statement for the rates */
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


    close_files(); 

    auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << " Code Finished Running in " << duration << " seconds time!" << endl;
}


// examples of integrate functions:
// integrate(tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);
// integrate_adaptive(controlled_stepper, tripleNonlinearODE, c0, 0.0, 500.0, 10.0, write_file);

/********** File IO **********/

/* open files for writing */
void open_files(){
    oFile.open("ODE_Soln.csv");
    oFile1.open("ODE_Const_Soln.csv"); 
    oFileMAV.open("mAv.csv");
}
/* write data to specific csv functions */
void write_file( const state_type &c , const double t ){ oFile << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl; }
void write_file_const( const state_type &c , const double t ){ oFile1 << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl; }
/* close files */
void close_files(){
    oFile.close();
    oFile1.close();
    oFileMAV.close();
}