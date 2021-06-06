#ifndef _ODE_HPP_
#define _ODE_HPP_

#include "main.hpp"
/* ODE Systems Functions */
void nonlinearODE3( const state_N_type &c , state_N_type &dcdt , double t );
void linearODE3_true( const state_N_type &c , state_N_type &dcdt , double t );
void linearODEn_1( const state_N_type &c , state_N_type &dcdt , double t );
void nonlinearODE6( const state_N_type &c , state_N_type &dcdt , double t);


struct K
{
    VectorXd k;
};

/* /* 3-var linear ODE system - need to rename! @TODO */
class Particle_Linear
{
    struct K bill;

public:
    Particle_Linear(struct K G) : bill(G) {}

    void operator() (  const state_N_type &c , state_N_type &dcdt , double t)
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

/* 6-variable nonlinear ODE system - G has no meaning, just a simple placeholder var */

class Nonlinear6ODE
{
    struct K jay;

public:
    Nonlinear6ODE(struct K G) : jay(G) {}

    void operator() (  const state_6_type &c , state_6_type &dcdt , double t)
    {
        
        dcdt[0] = - (jay.k(0) * c[0] * c[1]) 
                  + jay.k(1) * c[2] 
                  + jay.k(2) * c[2];
           
        dcdt[1] = - (jay.k(0) * c[0] * c[1]) 
                + jay.k(1) * c[2] 
                + jay.k(5) * c[5];

        dcdt[2] = jay.k(0) * c[0] * c[1] 
                - jay.k(1) * c[2]
                - jay.k(2) * c[2];

        dcdt[3] = jay.k(2) * c[2]
                - jay.k(3) * c[3] * c[4] 
                + jay.k(4) * c[5];

        dcdt[4] = -(jay.k(3) * c[3] * c[4]) 
                + jay.k(4) * c[5] 
                + jay.k(5) * c[5];

        dcdt[5] = jay.k(3) * c[3] * c[4] 
                - jay.k(4) * c[5] 
                - jay.k(5) * c[5];
    }
};

/* Observer Functions */
struct Data_Components{
    VectorXd subset;
    VectorXd mVec;
    MatrixXd m2Mat;
};
struct Data_ODE_Observer 
{
    struct Data_Components &dComp;
    Data_ODE_Observer( struct Data_Components &dCom) : dComp( dCom ) {}
    void operator()( state_N_type const& c, const double t ) const 
    {
        if(t == tf){
            for(int row = 0; row < dComp.subset.size(); row++){ // first moments of specified subset
                int i = dComp.subset(row) - 1; // i.e subset = {1,2,3} = index = {0,1,2}
                if(i >= 0){ dComp.mVec(i) +=  c[i]; }
                for(int col = row; col < dComp.subset.size(); col++){
                    int j = dComp.subset(col) - 1;
                    if (j >= 0){
                        if( i == j ){
                            dComp.mVec(N_SPECIES + i) += c[i] * c[j];
                        }else{
                            dComp.mVec(2*N_SPECIES + (i + j - 1)) += c[i] *c[j];
                        }
                        dComp.m2Mat(i,j) += (c[i] * c[j]);   // store in a 2nd moment matrix
                        dComp.m2Mat(j,i) = dComp.m2Mat(i,j);   // store in a 2nd moment matrix
                    }
                }
            }
        }
    }
};


/* Observer Function for filling up respective particle moment vectors and sample matrices */
struct Particle_Components
{   
    VectorXd subset;
    VectorXd momVec; // moment vector
    MatrixXd sampleMat; 
};
struct Particle_Observer
{
    struct Particle_Components &pComp;
    Particle_Observer( struct Particle_Components &pCom) : pComp( pCom ){}
    void operator()( const state_N_type &c , const double t ) 
    {
        if(t == tf){
           // cout << "confirmed" << endl;
            for(int col = 0; col < N_SPECIES; col++){    
                int i = pComp.subset(col) - 1;
                if(i >= 0){ pComp.sampleMat(pComp.sampleMat.rows() - 1, i) = c[i]; }   
            }
            pComp.sampleMat.conservativeResize(pComp.sampleMat.rows() + 1 , pComp.sampleMat.cols());

            for(int row = 0; row < pComp.subset.size(); row++){ // first moments of specified subset
                int i = pComp.subset(row) - 1; // i.e subset = {1,2,3} = index = {0,1,2}
                if(i >= 0){ pComp.momVec(i) +=  c[i]; }

                for(int col = row; col < pComp.subset.size(); col++){
                    int j = pComp.subset(col) - 1;
                    if(j >= 0){
                        if(i == j){
                            pComp.momVec(N_SPECIES + i) += c[i] * c[j];
                        }else{
                            pComp.momVec(2*N_SPECIES + (i + j - 1)) += c[i] *c[j];
                        }
                    }
                }
            }
        }
    }
}; 
#endif