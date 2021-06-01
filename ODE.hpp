#ifndef _ODE_HPP_
#define _ODE_HPP_

#include "main.hpp"
/* ODE Systems Functions */
void nonlinearODE3( const state_type &c , state_type &dcdt , double t );
void linearODE3_true( const state_type &c , state_type &dcdt , double t );
void linearODEn_1( const state_type &c , state_type &dcdt , double t );
void nonlinearODE6( const state_type &c , state_type &dcdt , double t);

/* The rhs of x' = f(x) defined as a class */
// define structure
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

/* Example Streaming Observer Format 
struct streaming_observer
{
    std::ostream& m_out;

    streaming_observer( std::ostream &out ) : m_out( out ) { }

    template< class State >
    void operator()( const State &x , double t ) const
    {
        container_type &q = x.first;
        m_out << t;
        for( size_t i=0 ; i<q.size() ; ++i ) m_out << "\t" << q[i];
        m_out << "\n";
    }
}; */

struct Particle_Components
{
    VectorXd momentVector; // note: Unfortunately, VectorXd from Eigen is far more complicated?
    MatrixXd sampleMat; 
};

/* Example Streaming Observer Format */
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
            pComp.sampleMat.conservativeResize(pComp.sampleMat.rows() + 1 ,pComp.sampleMat.cols());

            for(int row = 0; row < N_SPECIES; row++){
                pComp.momentVector(row) += c[row]; 
                for(int col = row; col < N_SPECIES; col++){
                    if( row == col){
                        pComp.momentVector(N_SPECIES + row) += c[row] * c[col];
                    }else{
                        pComp.momentVector(2*N_SPECIES + (row + col - 1)) += c[row] *c[col];
                    }
                }
            }
        }
    }
}; 



#endif