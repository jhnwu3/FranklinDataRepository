#ifndef _FILEIO_HPP_
#define _FILEIO_HPP_
#include "main.hpp"

/* FILE IO */
void open_files(ofstream& file0, ofstream& file1, ofstream& file2);
void close_files(ofstream& file0, ofstream& file1, ofstream& file2);
void write_particle_data( const VectorXd& k , const VectorXd &initCon, const VectorXd& mom, const VectorXd& mu, double cost);

struct Write_File_CSV // csv storing for 3 var only
{
    ostream& fOut;
    Write_File_CSV (ostream& out) : fOut( out ) {} 
    void operator()(const state_type &c, const double t){
        fOut << t << endl;
        for(int i = 0; i < N_SPECIES; i++){
        fOut << "," << c[i];
        }
    }
};

struct Write_File_Plot // for gnu plot file stream write out solved values for all of them
{
    ostream& fOut;
    Write_File_Plot (ostream& out) : fOut( out ) {} 
    void operator()(const state_type &c, const double t){
        fOut << t;
        for(int i = 0; i < N_SPECIES; i++){
        fOut << " " << c[i];
        }
        fOut << endl;
    }
};

struct Data_Components_IO{
    VectorXd subset;
    VectorXd mVec;
    MatrixXd m2Mat;
    ofstream out;
};
struct Data_ODE_Observer_IO 
{
    struct Data_Components_IO &dComp;
    Data_ODE_Observer_IO( struct Data_Components_IO &dCom) : dComp( dCom ) {}
    void operator()( state_type const& c, const double t ) const 
    {   
        dComp.out << t;
        for(int i = 0; i < N_SPECIES; i++){
            dComp.out << " " << c[i];
        }
        dComp.out << endl;

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

#endif
/* more FILE IO @TODO write classes for ease of file IO */
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