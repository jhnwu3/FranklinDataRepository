#include "ODE.hpp"

ofstream oFile; 
ofstream oFile1; 
ofstream oFileMAV; 

/* Specify Variables here before needing to define them into the var.cpp file, especially for Matrix math */
/* Variables for RNG */
VectorXd mu(3);
MatrixXd sigma  = MatrixXd::Zero(nProt, nProt);
/* moment vector */
VectorXd mVec = VectorXd::Zero(nProt*(nProt + 3) / 2); // for some t

/* Second moment matrix. */
MatrixXd m2 = MatrixXd::Zero(nProt, nProt); // for some t

/* Covariance Matrix */
MatrixXd cov(nProt, nProt);

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


/* Only to be used with integrate_const -, solves the ODE's defined in ODESys.cpp*/
void sample_const( const state_type &c , const double t){

    /* We will have some number of time steps */
    if(t == 10){
        mVec(0) += c[0]; // store all first moments in the first part of the moment vec
        mVec(1) += c[1];
        mVec(2) += c[2];
        /* form second moment symmetric matrix*/
        for(int row = 0; row < nProt; row++){
            for(int col = 0; col < nProt; col++){
                m2(row,col) += (c[row] * c[col]);    
            }
        }
    }
}
/* Only to be used with integrate/integrate_adaptive - @TODO */
void sample_adapt( const state_type &c , const double t){}

/* Temporary Location for Calculation Functions */
double kCost (const VectorXd& kTrueVec, const VectorXd& kEstVec){
    double cost = 0;
    for(int i = 0; i <nProt; i++){
        cost += (kEstVec(i) - kTrueVec(i)) * (kEstVec(i) - kTrueVec(i));
    }
    return cost;
}

int main(int argc, char **argv)
{   
    /* Random Number Generator */
    random_device rand_dev;
    mt19937 generator(rand_dev());
    uniform_real_distribution<double> unifDist(0.0, 1.0);
    int nMoments = (nProt * (nProt + 3)) / 2;
    auto t1 = std::chrono::high_resolution_clock::now(); // start time
    /* rate constants vectors */
    VectorXd kTrue(nMoments);
    kTrue << k1, k2, k3, k4, k5, 0, 0, 0, 0;
    cout << "kTrue:" << kTrue.transpose() << endl;
    VectorXd kEst(nMoments);
    VectorXd kEst1(nMoments);

    /* Fill with temporary random vals */
    for(int i = 0; i < nMoments; i++){
        kEst(i) = unifDist(generator); // one random between 0 and 1
        kEst1(i) = kTrue(i) + 0.1 * unifDist(generator); // another that differs from exact one by 0.1
    }
    
    /* ODE solver variables! */
    VectorXd initCon(3); // temp vector to be used for initiation conditions
    state_type c0 = {10.0 , 0.0 , 0.0 };
    controlled_stepper_type controlled_stepper;

    /* assign mu vector and sigma matrix values */
    mu << mu_x, mu_y, mu_z;
    sigma << 0.77, 0.0873098, 0.046225, 
             0.0873098, 0.99, 0.104828, 
             0.046225, 0.104828, 1.11;

    /* multivariate /normal distribution generator */
    normal_random_variable sample{mu, sigma};

    open_files(); // open file for writing

    /* average randomized sample/initial conditions from unif dist, N=10,000, CALL ODE SOLVER HERE! */
   for(int i = 0; i < N; i++){
       if(i % 1000 == 0){
           cout << i << endl; 
       }
        initCon = sample();
        c0 = { exp(initCon(0)), exp(initCon(1)), exp(initCon(2))}; // assign vector for use in ODE solns.
        integrate_const(controlled_stepper, tripleNonlinearODE, c0, t0, tf, dt, sample_const);
   }
    
    /* Divide the sums at the end to reduce number of needed division ops */
    for(int row = 0; row  < nProt; row++){
        mVec(row) /= N;  
        for(int col = 0; col < nProt; col++){
            m2(row, col) /= N;
        }
    }

    /* Fill moment vector with diagonals and unique values of the matrix */
    for(int i = 0; i < nProt; i++){
        mVec(nProt + i) = m2.diagonal()(i);
    }
    for(int row = 0; row < nProt - 1; row++){
        for(int col = row + 1; col < nProt; col++){
            mVec(2*nProt + (row + col - 1)) = m2(row, col);
        }
    }

    /* calculate covariance matrix */
    for(int row = 0; row < nProt; row++){
        for(int col = 0; col < nProt; col++){
            cov(row, col) = m2(row,col) - mVec(row)*mVec(col);
        }
    }

    /* Print statement for the rates */
    cout << "kCost for a set of k estimates between 0 and 1s:" << kCost(kTrue, kEst) << endl;
    cout << "kCost for a set of k estimates 0.1 * rand(0,1) away from true:" << kCost(kTrue, kEst1) << endl;
    /* Print statement for the moments */
    oFileMAV << "2nd moment matrix:" << endl;
    oFileMAV << m2 << endl << endl;

    cout << "2nd moment matrix:" << endl;
    cout << m2 << endl << endl;

    oFileMAV << "Full " << nProt << " moment vector" << endl;
    oFileMAV << mVec.transpose() << endl;

    cout << "Full " << nProt << " protein moment vector" << endl;
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