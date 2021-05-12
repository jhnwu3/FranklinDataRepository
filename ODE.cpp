#include "ODE.hpp"

/* global file streams to be able to access all files */
ofstream oFile; 
ofstream oFile1; 
ofstream oFileMAV; 

/* Global Vectors/Matrices to be accessible by ODEINT solvers */
/* moment vector */
VectorXd mVec = VectorXd::Zero(N_PROTEINS*(N_PROTEINS + 3) / 2); // for some t
/* Second moment matrix. */
MatrixXd m2Mat = MatrixXd::Zero(N_PROTEINS, N_PROTEINS); // secomd moment vector
/* Weight/Identity Matrix */
MatrixXd w = MatrixXd::Identity( (N_PROTEINS * (N_PROTEINS + 3)) / 2,  (N_PROTEINS * (N_PROTEINS + 3)) / 2);


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

/**** ODE-INT OBSERVER FUNCTIONS ****/
/* Only to be used with integrate_const(), solves the ODE's defined in ODESys.cpp*/
void sample_const( const state_type &c , const double t){

    /* We will have some time we are sampling for */
    if(t == tn){
        for(int row = 0; row < N_PROTEINS; row++){
            mVec(row) += c[row]; // store all first moments in the first part of the moment vec
            for(int col = 0; col < N_PROTEINS; col++){
                m2Mat(row,col) += (c[row] * c[col]);   // store in 
            }
        }
    }
}
/* Only to be used with integrate/integrate_adaptive - @TODO */
void sample_adapt( const state_type &c , const double t){}

int main(int argc, char **argv)
{   
    auto t1 = std::chrono::high_resolution_clock::now(); // start time
    int nMom = (N_PROTEINS * (N_PROTEINS + 3)) / 2; // number of moments

    /* Random Number Generator */
    random_device rand_dev;
    mt19937 generator(rand_dev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);
    /* Variables used for multivariate log normal distribution */
    VectorXd mu(N_PROTEINS);
    MatrixXd sigma  = MatrixXd::Zero(N_PROTEINS, N_PROTEINS);
    /* Covariance Matrix to be Calculated from Moments! */
    MatrixXd cov(N_PROTEINS, N_PROTEINS);
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
    VectorXd initCon(N_PROTEINS); // temp vector to be used for initiation conditions
    state_type c0;
    controlled_stepper_type controlled_stepper;

    /* assign mu vector and sigma matrix values 
    mu << mu_x, mu_y, mu_z;
    sigma << 0.77, 0.0873098, 0.046225, 
             0.0873098, 0.99, 0.104828, 
             0.046225, 0.104828, 1.11; */
    
    for(int row = 0; row < N_PROTEINS; row++){
         mu(row) = 1 + unifDist(generator);
         for(int col = 0; col < N_PROTEINS; col++){
             sigma(row,col) = exp(unifDist(generator));
         }
    }
    cout << "mu:" << mu.transpose() << endl << endl << "sigma:" << endl << sigma << endl << endl; 
    /* multivariate /normal distribution generator */
    normal_random_variable sample{mu, sigma};

    open_files(); 

    /* average randomized sample/initial conditions from unif dist, N=10,000, CALL ODE SOLVER HERE! */
   for(int i = 0; i < N; i++){
       if(i % 1000 == 0){ cout << i << endl;  }

        initCon = sample(); // sample from multilognormal dist
        for(int a = 0; a < N_PROTEINS; a++){
            c0[a] = exp(initCon(a)); // assign vector for use in ODE solns.
        }
        integrate_const(controlled_stepper, nonlinearODE6, c0, t0, tf, dt, sample_const);
   }
    
    /* Divide the sums at the end to reduce number of needed division ops */
    for(int row = 0; row  < N_PROTEINS; row++){
        mVec(row) /= N;  
        for(int col = 0; col < N_PROTEINS; col++){
            m2Mat(row, col) /= N;
        }
    }

    /* Fill moment vector with diagonals and unique values of the matrix */
    for(int i = 0; i < N_PROTEINS; i++){
        mVec(N_PROTEINS + i) = m2Mat.diagonal()(i);
    }
    for(int row = 0; row < N_PROTEINS - 1; row++){
        for(int col = row + 1; col < N_PROTEINS; col++){
            mVec(2*N_PROTEINS + (row + col - 1)) = m2Mat(row, col);
        }
    }
    cov = calculate_covariance_matrix(m2Mat, mVec, N_PROTEINS);
    /***** printf statements ******/
    /* Print statement for the rates */
    cout << "kTrue:" << endl << kTrue.transpose() << endl;
    cout << "kEst between 0 and 1s:" << endl << kEst.transpose() << endl;
    cout << "kEst1 0.1 * rand(0,1) away:" << endl << kEst1.transpose() << endl;
    cout << "kCost for a set of k estimates between 0 and 1s: " << kCost(kTrue, kEst, nMom) << endl;
    cout << "kCost for a set of k estimates 0.1 * rand(0,1) away from true: " << kCost(kTrue, kEst1, N_PROTEINS) << endl << endl;
    cout << "kCostMat for a set of k estimates between 0 and 1s: " << kCostMat(kTrue, kEst, w, N_PROTEINS) << endl;
    cout << "kCostMat for a set of k estimates 0.1 * rand(0,1) away from true: " << kCostMat(kTrue, kEst1, w, N_PROTEINS) << endl << endl;
    /* Print statement for the moments */

    oFileMAV << "2nd moment matrix:" << endl;
    oFileMAV << m2Mat << endl << endl;
    cout << "2nd moment matrix:" << endl;
    cout << m2Mat << endl << endl;

    oFileMAV << "Full " << N_PROTEINS << "protein moment vector" << endl;
    oFileMAV << mVec.transpose() << endl;
    cout << "Full " << N_PROTEINS << " protein moment vector" << endl;
    cout << mVec.transpose() << endl;

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