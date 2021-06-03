#ifndef _FILEIO_HPP_
#define _FILEIO_HPP_
#include "main.hpp"

/* FILE IO */
void open_files(ofstream& file0, ofstream& file1, ofstream& file2);
void close_files(ofstream& file0, ofstream& file1, ofstream& file2);
void write_particle_data( ofstream& file, const VectorXd& k , const VectorXd &initCon, const VectorXd& mom, const VectorXd& mu, double cost);

struct Write_File
{
    ostream& fOut;
    Write_File (ostream& out) : fOut( out ) {} 
    void operator()(const state_type &c, const double t){
         fOut << t << ',' << c[0] << ',' << c[1] << ',' << c[2] << endl;
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