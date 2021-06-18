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
double globalCost = 10000000; // some outrageous starting value
int particleIterator = 0;

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

    /* triple file streams for 3 times */
    ofstream covCorMatTf, covCorMat1, covCorMat5; 
    open_files(covCorMatTf, covCorMat1, covCorMat5); 

    /* calculation variables*/
    MatrixXd cov(N_SPECIES, N_SPECIES); // covar matrix   
    MatrixXd w = MatrixXd::Identity(nMom, nMom); // wt. matrix
    VectorXd sub = VectorXd::Zero(N_DIM); sub << 1,2,0,0,5,0;
    /* ODE solver variables */
    Controlled_RK_Stepper_N controlled_stepper;

    cout << "beginning to do nonlinear6 for given parameters!" << endl;
    /* For Checking Purposes - Graph Vav, p-Vav .., SHP1 */
    struct K trueK; 
    trueK.k << 5.0, 0.10, 1.00, 8.69, 0.05, 0.70; // GRAPH
    Nonlinear_ODE6 ODE6System(trueK);
    ofstream gnu;
    gnu.open("NonlinODE6_Syk_Vav_pVav_SHP1.txt"); 
    Write_File_Plot graphFile(gnu);
    State_N observedC0 = {120.0, 41.33, 0, 0, 80.0, 0}; // X_0, observed initial conditions!
    integrate_adaptive(controlled_stepper, ODE6System, observedC0, t0, tf, dt, graphFile);
    
    /* Calculate cov and cor matrix for several thousand samples at 3 different times */
    Data_Components data6(sub, tf, nMom);
    Data_Components data6T2(sub, 1.0, nMom);
    Data_Components data6T3(sub, 5.0, nMom);
    Data_ODE_Observer dataOBS6(data6);
    Data_ODE_Observer dataOBS6T2(data6T2);
    Data_ODE_Observer dataOBS6T3(data6T3);
    cout << "Data Module: Beginning to solve nonlinear6 for 10000 samples at 3 times!" << endl;
    for(int i = 0; i < N; i++){
        State_N c0 = gen_multi_lognorm_init6(); 
        integrate_adaptive(controlled_stepper, ODE6System, c0, t0, tf, dt, dataOBS6); 
        integrate_adaptive(controlled_stepper, ODE6System, c0, t0, 1.0, dt, dataOBS6T2); 
        integrate_adaptive(controlled_stepper, ODE6System, c0, t0, 5.0, dt, dataOBS6T3); 
    }
    data6.moments /= N;
    data6.secondMoments /= N;
    data6T2.moments /= N;
    data6T2.secondMoments /= N;
    data6T3.moments /= N;
    data6T3.secondMoments /= N;
    
    cout << "calculating cov matrices!" << endl;

    /* record data */
    cout <<"Correlation matrix i.e <ci(t)cj(t)> :"<< endl << data6.secondMoments << endl;
    covCorMatTf <<"Correlation matrix i.e <ci(t)cj(t)> :"<< endl << data6.secondMoments << endl;
    cov = calculate_covariance_matrix(data6.secondMoments, data6.moments, N_SPECIES);
    covCorMatTf <<"Cov mat :" << endl << cov << endl; 
    cout << "tf Cov Mat:" << endl << cov << endl;
    cout << endl << "Test:" << endl << data6T2.secondMoments << endl << endl;
    covCorMat1 <<"Correlation matrix i.e <ci(t)cj(t)> tf = 1.0:"<< endl << data6T2.secondMoments << endl;
    covCorMat1 <<"Cov Mat :"<< endl << calculate_covariance_matrix(data6T2.secondMoments, data6T2.moments, N_SPECIES) << endl;
    covCorMat5 <<"Correlation matrix i.e <ci(t)cj(t)> tf = 5.0:"<< endl << data6T3.secondMoments << endl;
    covCorMat5 <<"Cov Mat :"<< endl << calculate_covariance_matrix(data6T3.secondMoments, data6T3.moments, N_SPECIES) << endl;

    ofstream pFile;
    pFile.open("Protein_Cost_Dist.txt");
    ofstream pCostLabelledFile;
    pCostLabelledFile.open("Protein_Cost_Labeled.txt");
    ofstream pFileRand;
    pFileRand.open("Protein_Cost_Dist_Rand.txt");

    /* check by doing another set of costs 1% off for one particle */
    VectorXd pInit(N_SPECIES);  
    Particle_Components pComp1(sub, tf, nMom);
    struct K trueKOnePercentOff; // structure for particle rate constants
    trueKOnePercentOff.k = trueK.k;        
    Nonlinear_ODE6 pSys1(trueKOnePercentOff); // instantiate ODE System
    for(int i = 0; i < N/2; i++){
        State_N c0 = gen_multi_lognorm_init6();
        integrate_adaptive(controlled_stepper, pSys1, c0, t0, tf, dt, Particle_Observer(pComp1));
    } 
    pComp1.momVec /= (N/2);
    double pCost1 = calculate_cf1(data6.moments, pComp1.momVec, nMom);
    pCostLabelledFile << "pCost1 with exact k's:" << pCost1 << endl;
    cout << endl << endl << "Writing First Particle data!" << endl << endl;
    write_particle_data(trueKOnePercentOff.k, pInit, pComp1.momVec, data6.moments ,pCost1);

    pComp1.momVec = VectorXd::Zero(nMom);
    pComp1.sampleMat = MatrixXd(1, N_SPECIES); // start off with 1 row for initial sample size
    pComp1.timeToRecord = tf;
    trueKOnePercentOff.k = trueK.k * 1.01; 
    Nonlinear_ODE6 pSys2(trueKOnePercentOff);
    for(int i = 0; i < N/2; i++){
        State_N c0 = gen_multi_lognorm_init6();
        integrate_adaptive(controlled_stepper, pSys2, c0, t0, tf, dt, Particle_Observer(pComp1));
    } 
    pComp1.momVec /= (N/2); // now find cost for 0.1% difference for first part
    pCostLabelledFile << "pCost1 with exact k * 1.01's:" << calculate_cf1(data6.moments, pComp1.momVec, nMom) << endl;

    /**** parallel computing ****/
    cout << "Parallel Computing Has Started!" << endl << endl;
#pragma omp parallel for
    for(particleIterator = 0; particleIterator < N_PARTICLES; particleIterator++){
        /* RNG for individual particle elements */
        random_device pRanDev;
        mt19937 pGenerator(pRanDev());
        uniform_real_distribution<double> pUnifDist(0.0, 1.0);

        /* first iteration */
        double pCost;
        Particle_Components pComp(sub, tf, nMom); // instantiate particle component values

        /* Generate rate constants from uniform dist (0,1) for 6-dim hypercube */
        struct K pK; // structure for particle rate constants
        for(int i = 0; i < N_DIM; i++) {pK.k(i) = trueK.k(i) + 0.1*pUnifDist(pGenerator); } // new rate constants within 10%                  
        Nonlinear_ODE6 pSys(pK); // instantiate ODE System

        for(int i = 0; i < N/2; i++){
            State_N c0 = gen_multi_lognorm_init6();
            integrate_adaptive(controlled_stepper, pSys, c0, t0, tf, dt, Particle_Observer(pComp));
        } 
        pComp.momVec /= (N/2); 
        pCost = calculate_cf1(data6.moments, pComp.momVec, nMom); // cost
        pFile << pCost << endl; // for distribution of pCosts within 5-10% 
        
        /* Random Case */
        Particle_Components pCompRand(sub, tf, nMom);
        struct K pKRand;
        pKRand.k = VectorXd::Zero(N_DIM);
        Nonlinear_ODE6 randSys(pKRand);
        for(int i = 0; i < N_DIM; i++){
            pKRand.k(i) = pUnifDist(pGenerator);
        }
        for(int i = 0; i < N/2; i++){
            State_N c0 = gen_multi_lognorm_init6();
            integrate_adaptive(controlled_stepper, randSys, c0, t0, tf, dt, Particle_Observer(pCompRand));
        } 
        pCompRand.momVec /= (N/2); 
        double pCostRan = calculate_cf1(data6.moments, pCompRand.momVec, nMom); // cost
        pFileRand << pCostRan << endl;

        /* cost comparisons */
        #pragma omp critical
        {     
            cout << "protein moment vector: "<< pComp.momVec.transpose() << endl << "given mu: " << data6.moments.transpose() << endl;
            cout << "with cost:" << pCost << endl << endl << endl;
            
            if(pCost < globalCost){
                globalCost = pCost;
                bestMomentVector = pComp.momVec;
                globalSample = pComp.sampleMat;
            }
        }
 
        /* 2nd iteration - PSO*/
        /* using CF2 compute next cost function and recompute weight */
    }
   
    cout << "bestMomentVector: " << bestMomentVector.transpose() << endl << endl;
    cout << "Global Best Cost: " << globalCost << endl;
    cout << "Subset of best moment vector:" << gen_sub_mom_vec(bestMomentVector).transpose() << endl;
    cout << "with size:" << gen_sub_mom_vec(bestMomentVector).size();
    pFile.close();
    pFileRand.close();
    pCostLabelledFile.close();
    gnu.close();
    close_files(covCorMatTf, covCorMat1, covCorMat5); 

    auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    cout << " Code Finished Running in " << duration << " seconds time!" << endl;
}

