#include "main.hpp"
#include "fileIO.hpp"
/* @TODO LATER */

/* open files for writing */
void open_files(ofstream& file0, ofstream& file1, ofstream& file2){
    file0.open("ODE_Soln.csv");
    file1.open("ODE_Const_Soln.csv"); 
    file2.open("Moment.csv");
}

/* close files */
void close_files(ofstream& file0, ofstream& file1, ofstream& file2){
    file0.close();
    file1.close();
    file2.close();
}


void write_particle_data( ofstream &file, const VectorXd& k , const VectorXd &initCon, const VectorXd& mom, const VectorXd& mu, double cost){
    file <<"k const:" << k.transpose() << endl 
    << "cond:" << initCon.transpose() << endl
    << "mu:" << mu.transpose() << endl
    << "moments:" << mom.transpose() << endl
    << "cost:" << cost << endl;
    file.close();
}