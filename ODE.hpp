#ifndef _ODE_HPP_
#define _ODE_HPP_

#include "main.hpp"
/* ODE Systems Functions */
void nonlinearODE3( const state_type &c , state_type &dcdt , double t );
void linearODE3_true( const state_type &c , state_type &dcdt , double t );
void linearODEn_1( const state_type &c , state_type &dcdt , double t );
void nonlinearODE6( const state_type &c , state_type &dcdt , double t);

/* 3-var linear ODE system */
struct K
{
    VectorXd k;
};

/* ODE- System to be used for parallel computing for particles */
class Particle_Linear
{
    struct K T1;

public:
    Particle_Linear(struct K G) : T1(G) {}

    void operator() (  const state_type &c , state_type &dcdt , double t)
    {
        MatrixXd kr(N_SPECIES, N_SPECIES); 
        kr << 0, T1.k(1), T1.k(3),
            T1.k(2), 0, T1.k(0),
            0, T1.k(4), 0;
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
    void operator()( state_type const& c, const double t ) const 
    {
        if(t == tf){
            for(int row = 0; row < dComp.subset.size(); row++){ // first moments of specified subset
                int i = dComp.subset(row) - 1; // i.e subset = {1,2,3} = index = {0,1,2}
                dComp.mVec(i) +=  c[i];
                for(int col = 0; col < dComp.subset.size(); col++){
                    int j = dComp.subset(col) - 1;
                    if( i == j){
                        dComp.mVec(N_SPECIES + i) += c[i] * c[j];
                    }else{
                        dComp.mVec(2*N_SPECIES + (i + j - 1)) += c[i] *c[j];
                    }
                }
            }
        }
    }
};


/* Observer Function for filling up respective particle moment vectors and sample matrices */
struct Particle_Components
{
    VectorXd momVec; // moment vector
    MatrixXd sampleMat; 
};

struct Particle_Observer
{
    struct Particle_Components &pComp;
    Particle_Observer( struct Particle_Components &pCom) : pComp( pCom ){}
    void operator()( const state_type &c , const double t ) 
    {
        if(t == tf){
           // cout << "confirmed" << endl;
            for(int i = 0; i < N_SPECIES; i++){
                 pComp.sampleMat(pComp.sampleMat.rows() - 1, i) = c[i];
            }
            pComp.sampleMat.conservativeResize(pComp.sampleMat.rows() + 1 , pComp.sampleMat.cols());

            for(int row = 0; row < N_SPECIES; row++){
                pComp.momVec(row) += c[row]; 
                for(int col = row; col < N_SPECIES; col++){
                    if( row == col){
                        pComp.momVec(N_SPECIES + row) += c[row] * c[col];
                    }else{
                        pComp.momVec(2*N_SPECIES + (row + col - 1)) += c[row] *c[col];
                    }
                }
            }
        }
    }
}; 
#endif