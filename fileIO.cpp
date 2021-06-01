#include "main.hpp"
#include "fileIO.hpp"
/* @TODO LATER */

/* open files for writing */
void open_files(ofstream& file0, ofstream& file1, ofstream& file2){
    file0.open("ODE_Soln.csv");
    file1.open("ODE_Const_Soln.csv"); 
    file2.open("mAv.csv");
}

/* close files */
void close_files(ofstream& file0, ofstream& file1, ofstream& file2){
    file0.close();
    file1.close();
    file2.close();
}