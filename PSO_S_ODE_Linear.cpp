// PSO.cpp : Replacing Dr. Stewarts linear 3 ODE's with the nonlinear3 ODE system provided way earlier
//

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

#define N_SPECIES 3
#define N 2000 // # of samples to sample over
#define N_DIM 5 // dim of PSO hypercube
#define N_PARTICLES 20 


using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace boost;
using namespace boost::math;
using namespace boost::numeric::odeint;

/* typedefs for boost ODE-ints */
typedef boost::array< double , N_SPECIES > State_N;
typedef runge_kutta_cash_karp54< State_N > Error_RK_Stepper_N;
typedef controlled_runge_kutta< Error_RK_Stepper_N > Controlled_RK_Stepper_N;

const double ke = 0.0001, kme = 20, kf = 0.01, kmf = 18, kd = 0.03, kmd = 1, 
ka2 = 0.01, ka3 = 0.01, C1T = 20, C2T = 5, C3T = 4;

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

    void operator() (  const State_N &c , State_N &dcdt , double t)
    {
        MatrixXd kr(3, 3); 
        kr << 0, bill.k(1), bill.k(3),
            bill.k(2), 0, bill.k(0),
            0, bill.k(4), 0;
        dcdt[0] = (kr(0,0) * c[0] - kr(0,0) * c[0]) +
              (kr(0,1) * c[1] - kr(1,0) * c[0]) + 
              (kr(0,2) * c[2] - kr(2,0) * c[0]);

        dcdt[1] = (kr(1,0) * c[0] - kr(0,1) * c[1]) +
                (kr(1,1) * c[1] - kr(1,1) * c[1]) + 
                (kr(1,2) * c[2] - kr(2,1) * c[1]);

        dcdt[2] = (kr(2,0) * c[0] - kr(0,2) * c[2]) + 
                (kr(2,1) * c[1] - kr(1,2) * c[2]) + 
                (kr(2,2) * c[2] - kr(2,2) * c[2]);
    }
};


class Nonlinear_ODE6
{
    struct K jay;

public:
    Nonlinear_ODE6(struct K G) : jay(G) {}

    void operator() (  const State_N &c , State_N &dcdt , double t)
    {   
        dcdt[0] = - (jay.k(0) * c[0] * c[1])  // Syk
                  + jay.k(1) * c[2] 
                  + jay.k(2) * c[2];
           
        dcdt[1] = - (jay.k(0) * c[0] * c[1]) // Vav
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

struct Data_Components{
    int index;
    MatrixXd mat;
    double timeToRecord;
};
struct Data_ODE_Observer 
{
    struct Data_Components &dComp;
    Data_ODE_Observer( struct Data_Components &dCom) : dComp( dCom ) {}
    void operator()( State_N const& c, const double t ) const 
    {
        if(t == dComp.timeToRecord){
            for(int i = 0; i < dComp.mat.cols(); i++){ dComp.mat(dComp.index, i) = c[i]; }
        }
    }
};
struct Data_Components6{
    int index;
    MatrixXd mat;
	VectorXd sub;
    double timeToRecord;
};
struct Data_ODE_Observer6 
{
    struct Data_Components6 &dComp;
    Data_ODE_Observer6( struct Data_Components6 &dCom) : dComp( dCom ) {}
    void operator()( State_N const& c, const double t ) const 
    {
        if(t == dComp.timeToRecord){
			int i = 0, j = 0;
			while( i < 6 && j < dComp.sub.size()){
				if(i == dComp.sub(j)){
					dComp.mat(dComp.index,j) = c[i];
					j++;
				}
				i++;
			}
        }
    }
};

void nonlinearODE3( const State_N &c , State_N &dcdt , double t )
{
    dcdt[0] =  ((ke*(C1T - c[0]))/(kme + (C1T - c[0]))) + ((kf * (C1T - c[0]) * c[0] * c[1]) / (kmf + (C1T - c[0]))) - ((kd*c[0]*c[2])/(kmd + c[0])); // dc1dt = ke*(C1T-C1).... (in document)
    dcdt[1] =  ka2 *(C2T - c[1]); // dc2/dt = ka2 * (C2T - c2)
    dcdt[2] =  ka3*(C3T - c[2]); // dc3/dt = ka3 * (C3t - c3)
}
int main() {
	
	auto t1 = std::chrono::high_resolution_clock::now();
	/*---------------------- Setup ------------------------ */
	int bsi = 1, Nterms = 9, useEqual = 0, Niter = 1, Biter = 1; 

	/* Variables (global) */
	double t0 = 0, tf = 3.0, dt = 0.1;
	int wasflipped = 0, Nprots = 3, Npars = 5;
	double squeeze = 0.96, sdbeta = 0.05;

	/* SETUP */
	int useDiag = 0;
	int sf1 = 1;
	int sf2 = 1;
	
	int Nparts_1 = 1000;
	int Nsteps_1 = 5;

	int Nparts_2 = 5;
	int Nsteps_2 = 1000;

	// note for coder: wmatup is a list 
	vector<double> wmatup; 
	wmatup.push_back(0.15);
	wmatup.push_back(0.30);
	wmatup.push_back(0.45);
	wmatup.push_back(0.60);

	double dp = 1, sfp = 3, sfg = 1, sfe = 6;

	K trueK;
	trueK.k = VectorXd::Zero(Npars);
	trueK.k(0) =  0.27678200 / sf1;
	trueK.k(1) = 0.83708059 / sf1;
	trueK.k(2) = 0.44321700 / sf1;
	trueK.k(3) = 0.04244124 / sf1;
	trueK.k(4) = 0.30464502 / sf1;
	//trueK.k << 5.0, 0.1, 1.0, 8.69, 0.05, 0.70;

	vector<double> truk; // make a copy of a vector/ array/list 

	for (unsigned int i = 0; i < trueK.k.size(); i++) {
		truk.push_back(trueK.k(i));
	}

	// print truk values to a .par file w/ 5 columns? 
	ofstream truk_file("truk.par");
	for (unsigned int i = 0; i < truk.size(); i++) {
		truk_file << " " << truk.at(i);
	}
	truk_file.close();

	
	double mu_x = 1.47, mu_y = 1.74, mu_z = 1.99; // true means for MVN(theta)

	double var_x = 0.77, var_y = 0.99, var_z = 1.11; // true variances for MVN(theta);

	double rho_xy = 0.10, rho_xz = 0.05, rho_yz = 0.10; // true correlations for MVN

	double sigma_x = sqrt(var_x), sigma_y = sqrt(var_y), sigma_z = sqrt(var_z);

	double cov_xy = rho_xy * sigma_x * sigma_y;
	double cov_xz = rho_xz * sigma_x * sigma_z;
	double cov_yz = rho_yz * sigma_y * sigma_z;

	/* sigma matrices */
	MatrixXd sigma_12(1, 2);
	sigma_12 << cov_xz, cov_yz;
	
	MatrixXd sigma_22(2, 2);
	sigma_22 << var_x, cov_xy,
		cov_xy, var_y;
	
	MatrixXd sigma_21(2, 1);
	sigma_21 = sigma_12.transpose();

	/* conditional variances of proteins*/
	double cvar_ygx = (1 - (rho_xy * rho_xy)) * (sigma_y * sigma_y);

	double cvar_zgxy = var_z - (sigma_12 * sigma_22.inverse() * sigma_21)(0,0); // note: since matrix types are incompatible witgh doubles, we must do matrix math first, then convert to double.

	int t = 3; // num steps away from initial state

	// Linear ODE3
	
	/* global variables that are being recalculated in PSO */
	double omp_1, omp_2, omp_3, ovp_1 = 0, ovp_2 = 0, ovp_3 = 0, ocov_12, ocov_13, ocov_23;
	double pmp_1, pmp_2, pmp_3, pvp_1 = 0, pvp_2 = 0, pvp_3 = 0, pcov_12, pcov_13, pcov_23;
	double cost_seedk, cost_gbest, cost_sofar;
	MatrixXd X_0(N, 3);
	MatrixXd X_0_obs(N, 3);
	MatrixXd Y_t_obs(N, 3);
	MatrixXd Y_t(N, 3);
	VectorXd pmpV(3);

	MatrixXd GBMAT;
	MatrixXd w_mat = MatrixXd::Identity(9,9);

	VectorXd sub(3); sub << 0,1,4; //subset of protein indices we want to solve for!
	VectorXd gbest(Npars), best_sofar(Npars);

	VectorXd x(N); //note the data sample x is a list of 10000 RV from normal dist
	VectorXd pa_x(N);
	VectorXd y(N);
	VectorXd pa_y(N);
	VectorXd z(N); // z big questions about how to get the data values for it. It takes in a far bigger value???
	VectorXd pa_z(N);

	VectorXd all_terms(9);
	VectorXd term_vec(9);

	/* IMPORTANT THAT YOU INSTANTIATE THE RANDOM GENERATOR LIKE THIS!*/
	std::random_device rand_dev;
	std::mt19937 generator(rand_dev());
	uniform_real_distribution<double> unifDist(0.0, 1.0);

	// take 

	for (int q = 1; q <= Niter; q++) {

		int dpFlag = 1;   // Unsure what this print statement does, will ask later.
		if (q % 10 == 0) {
			cout << "Working on replicate " << q << "\n";
		}

		if (bsi == 0 || q == 1) {
			/* Simulate Y(t) and X(0) */
			
			std::normal_distribution<double> xNorm(mu_x, sigma_x);

			for (int i = 0; i < N; i++) {
				x(i) = (xNorm(generator));
				pa_x(i) = (exp(x(i)));
			}
			
			for (int i = 0; i < x.size(); i++) {
				std::normal_distribution<double> yNorm(mu_y + sigma_y * rho_xy * (x(i) - mu_x) / sigma_x, sqrt(cvar_ygx));
				y(i) = (yNorm(generator));
				pa_y(i) = (exp(y(i))); // convert to lognormal distribution!
			}
			
			/* matrix math for the z random vals. */
			MatrixXd rbind(2, N); // first calculate a 2xN rbind matrix
			for (int i = 0; i < x.size(); i++) {
				rbind(0, i) = x(i) - mu_x;
				rbind(1, i) = y(i) - mu_y;
			}
			MatrixXd zMean(1, N); // calculate the vector of means
			zMean = sigma_12 * sigma_22.inverse() * rbind;
			for (int i = 0; i < zMean.size(); i++) {
				zMean(0, i) = zMean(0, i) + mu_z;
			}
			// finally actually calculate z and pa_z vectors
			for (int i = 0; i < N; i++) {
				std::normal_distribution<double> zNorm(zMean(0,i), sqrt(cvar_zgxy));
				z(i) = (zNorm(generator));
				pa_z(i) = (exp(z(i)));
			}
			cout << "line 314" << endl;
			/* Create Y.0 */
			MatrixXd Y_0(N, 3);
			/*for (int i = 0; i < N; i++) {
				// fill it up from vectors
				Y_0(i, 0) = pa_x(i);
				Y_0(i, 1) = pa_y(i);
				Y_0(i, 2) = pa_z(i);
			}*/
			Y_0.col(0) = pa_x;
			Y_0.col(1) = pa_y;
			Y_0.col(2) = pa_z;

			/* COMPUTE ODES! */ // Y_t = (EMT * Y_0.transpose()).transpose(); - Convert to Y_t
			Data_Components dCom;
			Data_ODE_Observer obs(dCom);
			dCom.mat = MatrixXd::Zero(N, 3);
			dCom.timeToRecord = tf;
			cout << "line 363" << endl;
			State_N c0;
			Controlled_RK_Stepper_N controlledStepper;
			Linear_ODE3 ode3LinSys(trueK);
			//Nonlinear_ODE6 nonlinODE6(trueK);
			for(int i = 0; i < N; i++){
				dCom.index = i;
				int k = 0;
				for(int j = 0; j < N_SPECIES; j++){ 
					c0[j] = Y_0(i,j);
				}
				integrate_adaptive(controlledStepper, ode3LinSys, c0, t0, tf, dt, obs); 
				Y_t.row(i) = dCom.mat.row(i);
			}
			cout << "line 383" << endl;
			if (bsi == 1 && q == 1) {
				Y_t_obs = Y_t;
			}

			/*  # Compute the observed means, variances, and covariances
				# Add random noise to Y.t
				# trusd < -apply(Y.t, 2, sd)
				# error     <- t(matrix(rnorm(N*Nprots,rep(0,Nprots),trusd*0.01),nrow=3))
				# Y.t < -Y.t + error */

			/* means */
			VectorXd ompV = Y_t.colwise().mean();
		
			omp_1 = ompV(0);
			omp_2 = ompV(1);
			omp_3 = ompV(2);

			/* variances - actually have to manually calculate it, no easy library  */
			ovp_1 = (Y_t.col(0).array() - Y_t.col(0).array().mean()).square().sum() / ((double)Y_t.col(0).array().size() - 1);
			ovp_2 = (Y_t.col(1).array() - Y_t.col(1).array().mean()).square().sum() / ((double)Y_t.col(1).array().size() - 1);
			ovp_3 = (Y_t.col(2).array() - Y_t.col(2).array().mean()).square().sum() / ((double)Y_t.col(2).array().size() - 1);

			/* covariances - also requires manual calculation*/
			double sum12 = 0, sum13 = 0, sum23 = 0;
			for (int n = 0; n < N; n++)
			{
				sum12 += (Y_t(n, 0) - omp_1) * (Y_t(n, 1) - omp_2);
				sum13 += (Y_t(n, 0) - omp_1) * (Y_t(n, 2) - omp_3);
				sum23 += (Y_t(n, 1) - omp_2) * (Y_t(n, 2) - omp_3);

			}
			int N_SUBTRACT_ONE = N - 1;
			ocov_12 = sum12 / N_SUBTRACT_ONE;
			ocov_13 = sum13 / N_SUBTRACT_ONE;
			ocov_23 = sum23 / N_SUBTRACT_ONE;
			
			// SIMULATE X(0) ~ F(theta)
			for (int i = 0; i < N; i++) {
				x(i) = (xNorm(generator));
				pa_x(i) = (exp(x(i)));
			}

			for (int i = 0; i < N; i++) {
				std::normal_distribution<double> yNorm(mu_y + sigma_y * rho_xy * (x(i) - mu_x) / sigma_x, sqrt(cvar_ygx));
				y(i) = (yNorm(generator));
				pa_y(i) = (exp(y(i)));
			}
			cout << "line 431" << endl;
			/* matrix math for the z random vals. */
			MatrixXd r1bind(2, N); // first calculate a 2xN rbind matrix
			for (int i = 0; i < N; i++) {
				r1bind(0, i) = x(i) - mu_x;
				r1bind(1, i) = y(i) - mu_y;
			}
			MatrixXd z1Mean(1, N); // calculate the vector of means
			z1Mean = sigma_12 * sigma_22.inverse() * r1bind;
			for (int i = 0; i < z1Mean.size(); i++) {
				z1Mean(0, i) = z1Mean(0, i) + mu_z;
			}
			// finally actually calculate z and pa_z vectors
			for (int i = 0; i < N; i++) {
				std::normal_distribution<double> zNorm(z1Mean(0, i), sqrt(cvar_zgxy));
				z(i) = (zNorm(generator));
				pa_z(i) = (exp(z(i)));
			}
			X_0.col(0) = pa_x;
			X_0.col(1) = pa_y;
			X_0.col(2) = pa_z;

			cout << "ln 482" << endl;
			if (bsi == 1 && q == 1) {// save the simulated CYTOF data time 0
				X_0_obs = X_0;
			}
		}
		
		if (bsi == 1 && q > 1) {

			/* create shuffled indices based on uniform rand dist */
			vector<int> bindices;
			for (int i = 0; i < N; i++) { bindices.push_back(i);}
			shuffle(bindices.begin(), bindices.end(), generator); // shuffle indices as well as possible. 
			/* shuffle all of the values in the observed matrices*/
			for (int i = 0; i < N; i++) {
				X_0(i , 0) = X_0_obs(bindices.at(i), 0);
				X_0(i, 1) = X_0_obs(bindices.at(i), 1);
				X_0(i, 2) = X_0_obs(bindices.at(i), 2);
				Y_t(i, 0) = Y_t_obs(bindices.at(i), 0);
				Y_t(i, 1) = Y_t_obs(bindices.at(i), 1);
				Y_t(i, 2) = Y_t_obs(bindices.at(i), 2);
			}
			
			/* re-calc new omp, ovp, and ocovs, which should be the same???*/
			VectorXd ompV = Y_t.colwise().mean();
			omp_1 = ompV(0);
			omp_2 = ompV(1);
			omp_3 = ompV(2);
			/* variances - actually have to manually calculate it, no easy library  */
			ovp_1 = (Y_t.col(0).array() - Y_t.col(0).array().mean()).square().sum() / ((double)Y_t.col(0).array().size() - 1);
			ovp_2 = (Y_t.col(1).array() - Y_t.col(1).array().mean()).square().sum() / ((double)Y_t.col(1).array().size() - 1);
			ovp_3 = (Y_t.col(2).array() - Y_t.col(2).array().mean()).square().sum() / ((double)Y_t.col(2).array().size() - 1);

			/* covariances - also requires manual calculation*/
			double sum12 = 0, sum13 = 0, sum23 = 0;
			for (int n = 0; n < N; n++)
			{
				sum12 += (Y_t(n, 0) - omp_1) * (Y_t(n, 1) - omp_2);
				sum13 += (Y_t(n, 0) - omp_1) * (Y_t(n, 2) - omp_3);
				sum23 += (Y_t(n, 1) - omp_2) * (Y_t(n, 2) - omp_3);
			}
			int N_SUBTRACT_ONE = N - 1;
			ocov_12 = sum12 / N_SUBTRACT_ONE;
			ocov_13 = sum13 / N_SUBTRACT_ONE;
			ocov_23 = sum23 / N_SUBTRACT_ONE;
		}
		// Initialize variables to start the layered particle swarms
		int Nparts = Nparts_1;
		int Nsteps = Nsteps_1;

		w_mat = MatrixXd::Identity(9,9); //initialize weight matrix as identity matrix.
		
		VectorXd seedk(Npars); //initialize global best
		for (int i = 0; i < Npars; i++) { seedk(i) = unifDist(generator) /sf2; }

		/*Compute cost of seedk */
		trueK.k = seedk;

		// MatrixXd HM(3, 3);
		// HM << -k.at(2), k.at(2), 0,
		// 	k.at(1), -k.at(1) - k.at(4), k.at(4),
		// 	k.at(3), k.at(0), -k.at(0) - k.at(3);

		// MatrixXd HMT(3, 3);
		// HMT = t * HM.transpose();
		cout << "ln 547" << endl;
		/**** EXP() was here! ****/
		MatrixXd Q(N,3); // Q = X_t
		Data_Components dCom;
		Data_ODE_Observer obs(dCom);
		dCom.mat = MatrixXd::Zero(N, 3);
		dCom.timeToRecord = tf;
		State_N c0;
		Controlled_RK_Stepper_N controlledStepper;
		Linear_ODE3 ode3LinSys(trueK);
		//Nonlinear_ODE6 nonlinODE6(trueK);
		cout << "ln 559" << endl;
		for(int i = 0; i < N; i++){
			dCom.index = i;
			int k = 0;
			for(int j = 0; j < N_SPECIES; j++){ 
				c0[j] = X_0(i,j);
			}
			integrate_adaptive(controlledStepper, ode3LinSys, c0, t0, tf, dt, obs); 
			Q.row(i) = dCom.mat.row(i);
		}
		cout << "ln 576" << endl;
		//re-calc new omp, ovp, and ocovs, which should be the same???
	    pmpV = Q.colwise().mean();

		pmp_1 = pmpV(0);
		pmp_2 = pmpV(1);
		pmp_3 = pmpV(2);
	
		// variances - actually have to manually calculate it, no easy library
		pvp_1 = (Q.col(0).array() - Q.col(0).array().mean()).square().sum() / ((double) Q.col(0).array().size() - 1);
		pvp_2 = (Q.col(1).array() - Q.col(1).array().mean()).square().sum() / ((double) Q.col(1).array().size() - 1);
		pvp_3 = (Q.col(2).array() - Q.col(2).array().mean()).square().sum() / ((double) Q.col(2).array().size() - 1);

		// covariances - also requires manual calculation 
		double sum12 = 0, sum13 = 0, sum23 = 0;
		
		for (int n = 0; n < Q.rows(); n++)
		{
			sum12 += (Q(n, 0) - pmp_1) * (Q(n, 1) - pmp_2);	
			sum13 += (Q(n, 0) - pmp_1) * (Q(n, 2) - pmp_3);
			sum23 += (Q(n, 1) - pmp_2) * (Q(n, 2) - pmp_3);
		}
		double N_SUBTRACT_ONE = Q.rows() - 1.0;
		
		pcov_12 = sum12 / N_SUBTRACT_ONE;
		pcov_13 = sum13 / N_SUBTRACT_ONE;
		pcov_23 = sum23 / N_SUBTRACT_ONE;

		double term_1 = pmp_1 - omp_1, 
			term_2 = pmp_2 - omp_2, 
			term_3 = pmp_3 - omp_3, 
			term_4 = pvp_1 - ovp_1, 
			term_5 = pvp_2 - ovp_2,
			term_6 = pvp_3 - ovp_3,
			term_7 = pcov_12 - ocov_12, 
			term_8 = pcov_13 - ocov_13,
			term_9 = pcov_23 - ocov_23;
		// note to self: I'm using vectorXd from now on b/c it plays way better than the vectors built into C++ unless ofc there are strings that we need to input.
	
		all_terms << term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8, term_9;
	
		term_vec = all_terms;
		
		cost_seedk = term_vec.transpose() * w_mat * (term_vec.transpose()).transpose(); // CF2!

		// instantiate values 
		gbest = seedk;
		best_sofar = seedk;
		cost_gbest = cost_seedk;
		cost_sofar = cost_seedk;
		cout << "ln 626" << endl;
		GBMAT.conservativeResize(1, Npars + 1);
		cout << "ln 628" << endl;
		// will probably find a better method later, but will for now just create temp vec to assign values.
		VectorXd cbind(gbest.size() + 1);
		cbind << gbest, cost_gbest;
		GBMAT.row(GBMAT.rows() - 1) = cbind;
		cout << "ln 633" << endl;
		double nearby = sdbeta;
		MatrixXd POSMAT(Nparts, Npars);
		cout << "ln 635" << endl;
		for (int pso = 1; pso <= Biter + 1 ; pso++) {
			cout << "PSO:" << pso << endl;

			if (pso < Biter + 1) {
				for (int i = 0; i < Nparts; i++) {
					// row by row in matrix using uniform dist.
					for (int n = 0; n < Npars; n++) {
						POSMAT(i, n) = unifDist(generator) / sf2; // for each particle
					}
				}
				
			}
			
			if (pso == Biter + 1) {
				Nparts = Nparts_2;
				Nsteps = Nsteps_2;
				cout << "ln 652" << endl;
				GBMAT.conservativeResize(GBMAT.rows() + 1, Nparts + 1);
				cbind << best_sofar, cost_sofar;
				GBMAT.row(GBMAT.rows() - 1) = cbind;

				gbest = best_sofar;
				cost_gbest = cost_sofar;

				// reset POSMAT? 
				POSMAT.resize(Nparts, Npars);
				POSMAT.setZero();
	
				for (int init = 0; init < Nparts; init++) {
					for (int edim = 0; edim < Npars; edim++) {
						double tmean = gbest(edim);
						if (gbest(edim) > 0.5) {
							tmean = 1 - gbest(edim);
							wasflipped = 1;
						}
						double myc = (1 - tmean) / tmean;
						double alpha = myc / ((1 + myc) * (1 + myc) * (1 + myc)*nearby*nearby);
						double beta = myc * alpha;

						std::gamma_distribution<double> aDist(alpha, 1);
						std::gamma_distribution<double> bDist(beta, 1);

						double x = aDist(generator);
						double y = bDist(generator);
						double myg = x / (x + y);
						// sample from beta dist - this can be quite inefficient and taxing, there is another way with a gamma dist (THAT NEEDS TO BE REINVESTIGATED), but works so far. 
						//beta_distribution<double> betaDist(alpha, beta);
						//double randFromUnif = unifDist(generator);
						//double myg = quantile(betaDist, randFromUnif);

						if (wasflipped == 1) {
							wasflipped = 0;
							myg = 1 - myg;
						}
						POSMAT(init, edim) = myg;
					}
				}
				
			} 
			cout << "ln 695" << endl;
			// initialize PBMAT 
			MatrixXd PBMAT = POSMAT; // keep track of ea.particle's best, and it's corresponding cost
			
			PBMAT.conservativeResize(POSMAT.rows(), POSMAT.cols() + 1);
			for (int i = 0; i < PBMAT.rows(); i++) { PBMAT(i, PBMAT.cols() - 1) = 0; } // add the 0's on far right column
			cout << "ln 701" << endl;
			for (int h = 0; h < Nparts; h++) {
				for (int init = 0; init < Npars; init++) { trueK.k(init) = PBMAT(h, init); }
				if(h % 500 == 0){
					cout << "ln 705" << endl;
					cout << "h:" << h << endl;
					cout << "Nparts:" << Nparts << endl;
				}
				/* EXP() WAS HERE! -------------------------- SOLVE ODES AGAIN FOR X_t!*/
				Data_Components dCom;
				dCom.mat = MatrixXd::Zero(N, 3);
				dCom.timeToRecord = tf;
				State_N c0;
				Controlled_RK_Stepper_N controlledStepper;
				Linear_ODE3 ode3LinSys(trueK);
				Data_ODE_Observer obs(dCom);
				//Nonlinear_ODE6 nonlinODE6(trueK);
				for(int i = 0; i < N; i++){
					dCom.index = i;
					int k = 0;
					for(int j = 0; j < N_SPECIES; j++){ 
						c0[j] = X_0(i,j);
					}
					integrate_adaptive(controlledStepper, ode3LinSys, c0, t0, tf, dt, obs); 
					Q.row(i) = dCom.mat.row(i);
				}
				 
				pmpV = Q.colwise().mean();
				
				pmp_1 = pmpV(0);
				pmp_2 = pmpV(1);
				pmp_3 = pmpV(2);

				// variances - below is manual calculation  
				pvp_1 = (Q.col(0).array() - Q.col(0).array().mean()).square().sum() / ((double)Q.col(0).array().size() - 1);
				pvp_2 = (Q.col(1).array() - Q.col(1).array().mean()).square().sum() / ((double)Q.col(1).array().size() - 1);
				pvp_3 = (Q.col(2).array() - Q.col(2).array().mean()).square().sum() / ((double)Q.col(2).array().size() - 1);
				
				// covariances - also requires manual calculation 
				double sum12 = 0, sum13 = 0, sum23 = 0;

				for (int n = 0; n < Q.rows(); n++)
				{
					sum12 += (Q(n, 0) - pmp_1) * (Q(n, 1) - pmp_2);
					sum13 += (Q(n, 0) - pmp_1) * (Q(n, 2) - pmp_3);
					sum23 += (Q(n, 1) - pmp_2) * (Q(n, 2) - pmp_3);
				}
			    N_SUBTRACT_ONE = Q.rows() - 1.0;

				pcov_12 = sum12 / N_SUBTRACT_ONE;
				pcov_13 = sum13 / N_SUBTRACT_ONE;
				pcov_23 = sum23 / N_SUBTRACT_ONE;
				
				 term_1 = pmp_1 - omp_1,
					term_2 = pmp_2 - omp_2,
					term_3 = pmp_3 - omp_3,
					term_4 = pvp_1 - ovp_1,
					term_5 = pvp_2 - ovp_2,
					term_6 = pvp_3 - ovp_3,
					term_7 = pcov_12 - ocov_12,
					term_8 = pcov_13 - ocov_13,
					term_9 = pcov_23 - ocov_23;
				// note to self: I'm using vectorXd from now on b/c it plays way better than the vectors built into C++ unless ofc there are strings that we need to input.
				
				all_terms << term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8, term_9;
				term_vec = all_terms; 

				PBMAT(h, Npars) = term_vec.transpose() * w_mat * (term_vec.transpose()).transpose();
			}
			cout << "ln 692" << endl;
///////////////////////////////////////////////////////////   PSO PART 2 of Particle Module    ////////////////////////////////////////////////////////////////////////////////////////////////////			
			// ALL SWARMS BEGIN TO MOVE HERE 
			double sfi = sfe;
			double sfc = sfp;
			double sfs = sfg;

			for (int iii = 0; iii < Nsteps; iii++) { //REMEMBER IF THERE IS ITERATION WITH iii MAKE SURE TO SUBTRACT ONE

				
				if (pso == (Biter + 1)) {
					vector<int> chkpts;
					
					for (unsigned int i = 0; i < wmatup.size(); i++) {	
						chkpts.push_back(wmatup.at(i)* Nsteps);
					}
					
					if (iii == chkpts.at(0) || iii == chkpts.at(1) || iii == chkpts.at(2) || iii == chkpts.at(3)) {
						nearby = squeeze * nearby;

						trueK.k = gbest; 
						Data_Components dCom;
						dCom.mat = MatrixXd::Zero(N, 3);
						dCom.timeToRecord = tf;
						State_N c0 = {};
						Controlled_RK_Stepper_N controlledStepper;
						Linear_ODE3 ode3LinSys(trueK);
						Data_ODE_Observer obs(dCom);
						//Nonlinear_ODE6 nonlinODE6(trueK);
						cout << "line 804" << endl;
						for(int i = 0; i < N; i++){
							dCom.index = i;
							int k = 0;
							for(int j = 0; j < N_SPECIES; j++){ 
								c0[j] = X_0(i,j);
							}
							integrate_adaptive(controlledStepper, ode3LinSys, c0, t0, tf, dt, obs); 
							Q.row(i) = dCom.mat.row(i);
						}

						MatrixXd fmdiffs(Q.rows(), 3);
						fmdiffs = Y_t - Q;
						
						VectorXd mxt(3);
						mxt = Q.colwise().mean();
				
						VectorXd myt(3); 
						myt = Y_t.colwise().mean();
						/* cost function computations! */
						MatrixXd residxt(Q.rows(), Q.cols());
						residxt.col(0) = mxt.row(0).replicate(N, 1);
						residxt.col(1) = mxt.row(1).replicate(N, 1);
						residxt.col(2) = mxt.row(2).replicate(N, 1);
						residxt = Q - residxt;

						MatrixXd residyt(Y_t.rows(), Y_t.cols());
						residyt.col(0) = myt.row(0).replicate(N, 1);
						residyt.col(1) = myt.row(1).replicate(N, 1);
						residyt.col(2) = myt.row(2).replicate(N, 1);
						residyt = Y_t - residyt;

						MatrixXd smdiffs(N, 3);
						smdiffs = (residyt.array() * residyt.array()) - (residxt.array()* residxt.array());

						MatrixXd cprxt(N, 3);
						cprxt.col(0) = residxt.col(0).array() * residxt.col(1).array();
						cprxt.col(1) = residxt.col(0).array() * residxt.col(2).array();
						cprxt.col(2) = residxt.col(1).array() * residxt.col(2).array();

						MatrixXd cpryt(N, 3);
						cpryt.col(0) = residyt.col(0).array() * residyt.col(1).array();
						cpryt.col(1) = residyt.col(0).array() * residyt.col(2).array();
						cpryt.col(2) = residyt.col(1).array() * residyt.col(2).array();

						MatrixXd cpdiffs(N, 3);
						cpdiffs = cpryt - cprxt;

						MatrixXd Adiffs(N, 9);
						Adiffs << fmdiffs, smdiffs, cpdiffs; // concatenate

						MatrixXd g_mat(N, Nterms);
						g_mat = Adiffs;
						/* RECOMPUTE WEIGHT FUNCTION! */
						for (int m = 0; m < N; m++) { w_mat = w_mat + g_mat.row(m).transpose() * g_mat.row(m); }
						w_mat = w_mat / N;
						w_mat = w_mat.inverse();

						if (useDiag == 1) { w_mat = w_mat.diagonal().diagonal(); }

						// CALCULATE MEANS, VARIANCES, AND COVARIANCES
						pmpV = Q.colwise().mean();

						pmp_1 = pmpV(0);
						pmp_2 = pmpV(1);
						pmp_3 = pmpV(2);

						// variances - manually calculate it, no easy library 
						pvp_1 = (Q.col(0).array() - Q.col(0).array().mean()).square().sum() / ((double)Q.col(0).array().size() - 1);
						pvp_2 = (Q.col(1).array() - Q.col(1).array().mean()).square().sum() / ((double)Q.col(1).array().size() - 1);
						pvp_3 = (Q.col(2).array() - Q.col(2).array().mean()).square().sum() / ((double)Q.col(2).array().size() - 1);

						// covariances - manual calculation 
						double sum12 = 0, sum13 = 0, sum23 = 0;

						for (int n = 0; n < Q.rows(); n++)
						{
							sum12 += (Q(n, 0) - pmp_1) * (Q(n, 1) - pmp_2);
							sum13 += (Q(n, 0) - pmp_1) * (Q(n, 2) - pmp_3);
							sum23 += (Q(n, 1) - pmp_2) * (Q(n, 2) - pmp_3);

						}
						N_SUBTRACT_ONE = Q.rows() - 1.0;

						pcov_12 = sum12 / N_SUBTRACT_ONE;
						pcov_13 = sum13 / N_SUBTRACT_ONE;
						pcov_23 = sum23 / N_SUBTRACT_ONE;

						term_1 = pmp_1 - omp_1,
							term_2 = pmp_2 - omp_2,
							term_3 = pmp_3 - omp_3,
							term_4 = pvp_1 - ovp_1,
							term_5 = pvp_2 - ovp_2,
							term_6 = pvp_3 - ovp_3,
							term_7 = pcov_12 - ocov_12,
							term_8 = pcov_13 - ocov_13,
							term_9 = pcov_23 - ocov_23;
						// note to self: I'm using vectorXd from now on b/c it plays way better than the vectors built into C++ unless ofc there are strings that we need to input.
						
						all_terms << term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8, term_9;
						term_vec = all_terms;
						
						cost_gbest = term_vec.transpose() * w_mat * term_vec.transpose().transpose();
						
						GBMAT.conservativeResize(GBMAT.rows() + 1, GBMAT.cols());
						VectorXd cbind1(GBMAT.cols());
						cbind1 << gbest, cost_gbest;
						GBMAT.row(GBMAT.rows() - 1) = cbind1;
						
						POSMAT.resize(Nparts,Npars); //reset to 0???
						POSMAT.setZero();
						
						for (int init = 0; init < Nparts; init++) {
							for (int edim = 0; edim < Npars; edim++) {
								double tmean = gbest(edim);
								if (gbest(edim) > 0.5) {
									tmean = 1 - gbest(edim);
									wasflipped = 1;
								}
								double myc = (1 - tmean) / tmean;
								double alpha = myc / ((1 + myc) * (1 + myc) * (1 + myc) * nearby * nearby);
								double beta = myc * alpha;

								// sample from beta dist - this can be quite inefficient and taxing, there is another way with a gamma dist (THAT NEEDS TO BE REINVESTIGATED), but works so far.
								std::gamma_distribution<double> aDist(alpha, 1);
								std::gamma_distribution<double> bDist(beta, 1);

								double x = aDist(generator);
								double y = bDist(generator);
								double myg = x/(x+y);
								if (wasflipped == 1) {
									wasflipped = 0;
									myg = 1 - myg;
								}
								POSMAT(init, edim) = myg;
							}
						}
						
						MatrixXd cbindMat(POSMAT.rows(), POSMAT.cols() + 1); // keep track of each particle's best and it's corresponding cost
						cbindMat << POSMAT, VectorXd::Zero(POSMAT.rows());

						for (int h = 0; h < Nparts; h++) {
							//for (int init = 0; init < Npars; init++) { k.at(init) = PBMAT(h, init); } 
							trueK.k = PBMAT.row(h);
							
							dCom.mat = MatrixXd::Zero(N, 3);
							dCom.timeToRecord = tf;
							State_N c0 = {};
							Controlled_RK_Stepper_N controlledStepper;
							Linear_ODE3 ode3LinSys(trueK);
							cout << "ln 961" << endl;
							Data_ODE_Observer obs(dCom);
							for(int i = 0; i < N; i++){
								dCom.index = i;
								int k = 0;
								for(int j = 0; j < N_SPECIES; j++){ 
									c0[j] = X_0(i,j);
								}
								integrate_adaptive(controlledStepper, ode3LinSys, c0, t0, tf, dt, obs); 
								Q.row(i) = dCom.mat.row(i);
							}

							// CALCULATE MEANS, VARIANCES, AND COVARIANCES
							pmpV = Q.colwise().mean();

							pmp_1 = pmpV(0);
							pmp_2 = pmpV(1);
							pmp_3 = pmpV(2);

							// variances - manually calculate it, no easy library 
							pvp_1 = (Q.col(0).array() - Q.col(0).array().mean()).square().sum() / ((double)Q.col(0).array().size() - 1);
							pvp_2 = (Q.col(1).array() - Q.col(1).array().mean()).square().sum() / ((double)Q.col(1).array().size() - 1);
							pvp_3 = (Q.col(2).array() - Q.col(2).array().mean()).square().sum() / ((double)Q.col(2).array().size() - 1);
							// covariances - manual calculation 
							double sum12 = 0, sum13 = 0, sum23 = 0;

							for (int n = 0; n < Q.rows(); n++)
							{
								sum12 += (Q(n, 0) - pmp_1) * (Q(n, 1) - pmp_2);
								sum13 += (Q(n, 0) - pmp_1) * (Q(n, 2) - pmp_3);
								sum23 += (Q(n, 1) - pmp_2) * (Q(n, 2) - pmp_3);

							}
							N_SUBTRACT_ONE = Q.rows() - 1.0;

							pcov_12 = sum12 / N_SUBTRACT_ONE;
							pcov_13 = sum13 / N_SUBTRACT_ONE;
							pcov_23 = sum23 / N_SUBTRACT_ONE;

							term_1 = pmp_1 - omp_1,
								term_2 = pmp_2 - omp_2,
								term_3 = pmp_3 - omp_3,
								term_4 = pvp_1 - ovp_1,
								term_5 = pvp_2 - ovp_2,
								term_6 = pvp_3 - ovp_3,
								term_7 = pcov_12 - ocov_12,
								term_8 = pcov_13 - ocov_13,
								term_9 = pcov_23 - ocov_23;
							// note to self: I'm using vectorXd from now on b/c it plays way better than the vectors built into C++ unless ofc there are strings that we need to input.

							all_terms << term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8, term_9;
							term_vec = all_terms;
							PBMAT(h, Npars) = term_vec.transpose() * w_mat * term_vec.transpose().transpose();
						}
					}
				}

				//cout << "line 802" << endl;
				for (int jjj = 0; jjj < Nparts; jjj++) {

					double w1 = sfi * unifDist(generator) /sf2, w2 = sfc*  unifDist(generator) / sf2, w3 = sfs * unifDist(generator) / sf2;
					double sumw = w1 + w2 + w3;

					w1 = w1 / sumw;
					w2 = w2 / sumw;
					w3 = w3 / sumw;

					// R -sample ~ shuffle
					
					vector<int> seqOneToFive;
					seqOneToFive.clear();
					for (int i = 0; i < Npars; i++) {
						seqOneToFive.push_back(i);
					}
					shuffle(seqOneToFive.begin(), seqOneToFive.end(), generator); // shuffle indices as well as possible. 
					int ncomp = seqOneToFive.at(0);
					VectorXd wcomp(ncomp);
					shuffle(seqOneToFive.begin(), seqOneToFive.end(), generator);
					for (int i = 0; i < ncomp; i++) {
						wcomp(i) = seqOneToFive.at(i);
					}
				
					VectorXd rpoint = POSMAT.row(jjj);

					for (int smart = 0; smart < ncomp; smart++) {
						int px = wcomp(smart);
						double pos = rpoint(px);
						double alpha = 4 * pos;
						double beta = 4 - alpha;

						std::gamma_distribution<double> aDist(alpha, 1);
						std::gamma_distribution<double> bDist(beta, 1);
						
						double x = aDist(generator);
						double y = bDist(generator);
						
						rpoint(px) = (x/(x+y)) / sf2;
					}
					
					VectorXd PBMATV(5);
					PBMATV << PBMAT(jjj, 0), PBMAT(jjj, 1), PBMAT(jjj, 2), PBMAT(jjj, 3), PBMAT(jjj, 4);
					POSMAT.row(jjj) = w1 * rpoint + w2 * PBMATV + w3 * gbest;


					/* set k equal to next position of particle */
					//for (int i = 0; i < Npars; i++) { k.at(i) = POSMAT(jjj, i); }
					trueK.k = POSMAT.row(jjj);
					if(jjj % 1000 == 0){
						cout << "jjj:" << jjj << endl; 
					}

					dCom.mat = MatrixXd::Zero(N, 3);
					dCom.timeToRecord = tf;
					State_N c0 = {};
					// Linear_ODE3 ode3LinSys(trueK);
					Data_ODE_Observer obs(dCom);
					Linear_ODE3 linSys3(trueK);
					for(int i = 0; i < N; i++){
						dCom.index = i;
						int k = 0;
						for(int j = 0; j < N_SPECIES; j++){ 
							c0[j] = X_0(i,j);
						}
						integrate_adaptive(controlledStepper, linSys3, c0, t0, tf, dt, obs); 
						Q.row(i) = dCom.mat.row(i);
					}

					// CALCULATE MEANS, VARIANCES, AND COVARIANCES
					pmpV = Q.colwise().mean();
				
					pmp_1 = pmpV(0);
					pmp_2 = pmpV(1);
					pmp_3 = pmpV(2);
 
					// variances - below is best way to calculate column wise
					pvp_1 = (Q.col(0).array() - Q.col(0).array().mean()).square().sum() / ((double)Q.col(0).array().size() - 1);
					pvp_2 = (Q.col(1).array() - Q.col(1).array().mean()).square().sum() / ((double)Q.col(1).array().size() - 1);
					pvp_3 = (Q.col(2).array() - Q.col(2).array().mean()).square().sum() / ((double)Q.col(2).array().size() - 1);

					// covariances - manual calculation 
					double sum12 = 0, sum13 = 0, sum23 = 0;

					for (int n = 0; n < Q.rows(); n++)
					{
						sum12 += (Q(n, 0) - pmp_1) * (Q(n, 1) - pmp_2);
						sum13 += (Q(n, 0) - pmp_1) * (Q(n, 2) - pmp_3);
						sum23 += (Q(n, 1) - pmp_2) * (Q(n, 2) - pmp_3);

					}
					N_SUBTRACT_ONE = Q.rows() - 1.0;

					pcov_12 = sum12 / N_SUBTRACT_ONE;
					pcov_13 = sum13 / N_SUBTRACT_ONE;
					pcov_23 = sum23 / N_SUBTRACT_ONE;

					term_1 = pmp_1 - omp_1,
						term_2 = pmp_2 - omp_2,
						term_3 = pmp_3 - omp_3,
						term_4 = pvp_1 - ovp_1,
						term_5 = pvp_2 - ovp_2,
						term_6 = pvp_3 - ovp_3,
						term_7 = pcov_12 - ocov_12,
						term_8 = pcov_13 - ocov_13,
						term_9 = pcov_23 - ocov_23;
					// note to self: I'm using vectorXd from now on b/c it plays way better than the vectors built into C++ unless ofc there are strings that we need to input.

					all_terms << term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8, term_9;
					term_vec = all_terms;

					// USE THE MOST RECENT ESTIMATE OF WMAT UNLESS USEEQUAL == 1
					double cost_newpos;
					if (useEqual == 0) { cost_newpos = term_vec.transpose() * w_mat * term_vec.transpose().transpose(); }
					if (useEqual == 1) { cost_newpos = term_vec.transpose() * MatrixXd::Identity(9,9) * term_vec.transpose().transpose(); }
					if (cost_newpos < PBMAT(jjj, Npars)) {
						//cout << "line 913" << endl;
						VectorXd POSMAT_cost_newpos(POSMAT.cols());
						POSMAT_cost_newpos = POSMAT.row(jjj);
						POSMAT_cost_newpos.conservativeResize(PBMAT.cols());
						POSMAT_cost_newpos(PBMAT.cols() - 1) = cost_newpos;
						PBMAT.row(jjj) = POSMAT_cost_newpos;

						if (cost_newpos < cost_gbest) {
							gbest = POSMAT.row(jjj);
							cost_gbest = cost_newpos;
						}
					}
					
				}


				sfi = sfi - (sfe - sfg) / Nsteps; // reduce the inertial weight after each step
				sfs = sfs + (sfe - sfg) / Nsteps; // increase social weight after each step


				// CHECK IF NEW GLOBAL BEST HAS BEEN FOUND
				int neflag = 0;
				int lastrow = GBMAT.rows() - 1;
				
				for (int ne = 0; ne < Npars; ne++) {
					if (GBMAT(lastrow, ne) != gbest(ne)) {
						neflag = 1;
					}
				}

				if (GBMAT(lastrow, Npars) != cost_gbest) {
					neflag = 1;
				}

				//IF NEW GLOBAL BEST HAS BEEN FOUND, THEN UPDATE GBMAT
				if (neflag == 1) {
					//cout << "line 943" << endl;
					GBMAT.conservativeResize(GBMAT.rows() + 1, GBMAT.cols()); //rbind method currently.... where cbind is the "column bind vector"
					cbind << gbest, cost_gbest;
					GBMAT.row(GBMAT.rows() - 1) = cbind;
				}
				
			}

			if (pso < (Biter + 1)) {
				trueK.k = gbest;  // best estimate of k to compute w.mat
				/* RECOMPUTE X_t */
				cout << "ln 1194" << endl;
				dCom.mat = MatrixXd::Zero(N, 3);
				dCom.timeToRecord = tf;
				// Linear_ODE3 ode3LinSys(trueK);
				Nonlinear_ODE6 nonlinODE6(trueK);
				for(int i = 0; i < N; i++){
					dCom.index = i;
					int k = 0;
					for(int j = 0; j < N_SPECIES; j++){ 
						// if(j == sub(j)){
						// 	c0[j] = X_0(i, k);
						// 	k++; 
						// }else{
						// 	c0[j] = 0;
						// }
						c0[j] = X_0(i,j);
					}
					integrate_adaptive(controlledStepper, nonlinODE6, c0, t0, tf, dt, obs); 
					Q.row(i) = dCom.mat.row(i);
				}

				MatrixXd fmdiffs(N, 3);
				fmdiffs = Y_t - Q;

				VectorXd mxt(3);
				mxt = Q.colwise().mean();

				VectorXd myt(3);
				myt = Y_t.colwise().mean();

				MatrixXd residxt(Q.rows(), Q.cols());
				residxt.col(0) = mxt.row(0).replicate(N, 1);
				residxt.col(1) = mxt.row(1).replicate(N, 1);
				residxt.col(2) = mxt.row(2).replicate(N, 1);
				residxt = Q - residxt;

				MatrixXd residyt(Y_t.rows(), Y_t.cols());
				residyt.col(0) = myt.row(0).replicate(N, 1);
				residyt.col(1) = myt.row(1).replicate(N, 1);
				residyt.col(2) = myt.row(2).replicate(N, 1);
				residyt = Y_t - residyt;

				MatrixXd smdiffs(N, 3);
				smdiffs = (residyt.array() * residyt.array()) - (residxt.array() * residxt.array());
				
				MatrixXd cprxt(N, 3);
				cprxt.col(0) = residxt.col(0).array() * residxt.col(1).array();
				cprxt.col(1) = residxt.col(0).array() * residxt.col(2).array();
				cprxt.col(2) = residxt.col(1).array() * residxt.col(2).array();

				MatrixXd cpryt(N, 3);
				cpryt.col(0) = residyt.col(0).array() * residyt.col(1).array();
				cpryt.col(1) = residyt.col(0).array() * residyt.col(2).array();
				cpryt.col(2) = residyt.col(1).array() * residyt.col(2).array();

				MatrixXd cpdiffs(N, 3);
				cpdiffs = cpryt - cprxt;

				MatrixXd Adiffs(N, 9);

				Adiffs << fmdiffs, smdiffs, cpdiffs; // concatenate

				MatrixXd g_mat(N, Nterms);
				g_mat = Adiffs;

				w_mat.setZero();
				for (int m = 0; m < N; m++) { w_mat = w_mat + (g_mat.row(m).transpose()) * g_mat.row(m); }
				w_mat = (w_mat/N).inverse();
			
				if (useDiag == 1) { w_mat = w_mat.diagonal().diagonal(); }

				// Update cost_gbest with w_mat

				trueK.k = gbest; // recompute the cost for seedk using this w.mat

				cout << "ln 1270" << endl;
				dCom.mat = MatrixXd::Zero(N, 3);
				dCom.timeToRecord = tf;
				// Linear_ODE3 ode3LinSys(trueK);
				Linear_ODE3 lin3(trueK);
				for(int i = 0; i < N; i++){
					dCom.index = i;
					int k = 0;
					for(int j = 0; j < N_SPECIES; j++){ 
						c0[j] = X_0(i,j);
					}
					integrate_adaptive(controlledStepper, lin3, c0, t0, tf, dt, obs); 
					Q.row(i) = dCom.mat.row(i);
				}

				// CALCULATE MEANS, VARIANCES, AND COVARIANCES
				pmpV = Q.colwise().mean();

				pmp_1 = pmpV(0);
				pmp_2 = pmpV(1);
				pmp_3 = pmpV(2);

				pvp_1 = 0;
				pvp_2 = 0;
				pvp_3 = 0;
				// variances - manually calculate it, no easy library 
				pvp_1 = (Q.col(0).array() - Q.col(0).array().mean()).square().sum() / ((double)Q.col(0).array().size() - 1);
				pvp_2 = (Q.col(1).array() - Q.col(1).array().mean()).square().sum() / ((double)Q.col(1).array().size() - 1);
				pvp_3 = (Q.col(2).array() - Q.col(2).array().mean()).square().sum() / ((double)Q.col(2).array().size() - 1);
				// covariances - manual calculation 
				double sum12 = 0, sum13 = 0, sum23 = 0;

				for (int n = 0; n < Q.rows(); n++)
				{
					sum12 += (Q(n, 0) - pmp_1) * (Q(n, 1) - pmp_2);
					sum13 += (Q(n, 0) - pmp_1) * (Q(n, 2) - pmp_3);
					sum23 += (Q(n, 1) - pmp_2) * (Q(n, 2) - pmp_3);

				}
				double N_SUBTRACT_ONE = Q.rows() - 1.0;

				pcov_12 = sum12 / N_SUBTRACT_ONE;
				pcov_13 = sum13 / N_SUBTRACT_ONE;
				pcov_23 = sum23 / N_SUBTRACT_ONE;

				term_1 = pmp_1 - omp_1,
					term_2 = pmp_2 - omp_2,
					term_3 = pmp_3 - omp_3,
					term_4 = pvp_1 - ovp_1,
					term_5 = pvp_2 - ovp_2,
					term_6 = pvp_3 - ovp_3,
					term_7 = pcov_12 - ocov_12,
					term_8 = pcov_13 - ocov_13,
					term_9 = pcov_23 - ocov_23;
				// note to self: I'm using vectorXd from now on b/c it plays way better than the vectors built into C++ unless ofc there are strings that we need to input.

				all_terms << term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8, term_9;
				term_vec = all_terms;

				cost_gbest = term_vec.transpose() * w_mat * (term_vec.transpose()).transpose();

				GBMAT.conservativeResize(GBMAT.rows() + 1, GBMAT.cols());
				VectorXd cbind(gbest.size() + 1);
				cbind << gbest, cost_gbest;
				GBMAT.row(GBMAT.rows() - 1) = cbind;

				if (cost_gbest < cost_sofar) {
					best_sofar = gbest;
					cost_sofar = cost_gbest;
				}
			}

			if (bsi == 0 || q == 1) {
				if (pso < (Biter + 1)) {
					cout << "blindpso+cost.est:" << endl << best_sofar << endl << cost_sofar << endl << endl;
				}
				if (pso == (Biter + 1)) {
					cout << "igmme + cost.est:"<< endl << gbest << endl << cost_gbest  << endl << endl;
					cout << "w.mat" << w_mat.transpose()  << endl << endl;
				}
			}
			if (bsi == 1 && q > 1) {
				if (pso == 1) {
					cout << gbest << cost_gbest << "ubsreps+cost.mat" << endl << endl;
				}
				if (pso > 1) {
					cout << gbest << cost_gbest << "wbsreps+cost.mat" << endl << endl;
				} 
			} 
		}  // end loop over PSO layers */
		

	} // end loop over NIter simulations
	cout << "GBMAT: " << endl;
	cout << GBMAT << endl;
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
	cout << "CODE FINISHED RUNNING IN "<< duration<< " s TIME!" << endl;


	return 0; // just to close the program at the end.
}







/* generate random individual values */

/*double uniformRandDouble(double range_from, double range_to) {
	std::random_device rand_dev;
	std::mt19937 generator(rand_dev());
	std::uniform_int_distribution<double>    distr(range_from, range_to);
	return distr(generator);
}*/




