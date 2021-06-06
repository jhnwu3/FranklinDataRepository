#include "ODE.hpp"
#include "calc.hpp"

/* abbreviation: cf1 = cost function 1 */
double calculate_cf1(const VectorXd& trueVec, const VectorXd& estVec, int n){
    double cost = 0;
    for(int i = 0; i < n; i++){
        cost += (estVec(i) - trueVec(i)) * (estVec(i) - trueVec(i));
    }
    return cost;
}

double calculate_cf2(const VectorXd& trueVec, const  VectorXd& estVec, const MatrixXd& w, int n){
    double cost = 0;
    VectorXd diff(n);
    /*for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
           cost += (estVec(i) - trueVec(i)) * w(i,j) *(estVec(j) - trueVec(j));
        }
    }*/
    diff = trueVec - estVec;
    cost = diff.transpose() * w * (diff.transpose()).transpose();
    return cost;
}

/* mVec = first moment vector (may or may not have other vector components
   m2 = second moment vector
   cov = value returned after calculating cov matrix
*/
MatrixXd calculate_covariance_matrix(const MatrixXd& m2, const VectorXd& mVec, int nProt){
    MatrixXd cov(nProt, nProt);
     /* calculate covariance matrix */
    for(int row = 0; row < nProt; row++){
        for(int col = 0; col < nProt; col++){
            cov(row, col) = m2(row,col) - mVec(row)*mVec(col);
        }
    }
    return cov;
}


/* Creation Functions for testing of ODEs and multinormal distributions! */
MatrixXd create_covariance_matrix(const MatrixXd& sampleSpace, const VectorXd& mu, int nProt){
    MatrixXd cov = MatrixXd::Zero(nProt, nProt);
    /* Calculate covar matrix labeled sigma */
    for (int i = 0; i < N_SPECIES; i++) {
        for (int j = 0; j < N_SPECIES; j++) {
            for (int a = 0; a < N; a++) {
                cov(i, j) += (sampleSpace(a, i) - mu(i)) * (sampleSpace(a, j) - mu(j));
            }
        }
    }
    cov /= N;
    return cov;
}
MatrixXd generate_sample_space(int nProt, int n){
     /* Random Number Generator */
    random_device ranDev;
    mt19937 generator(ranDev());
    MatrixXd sampleSpace(n, nProt);
    normal_distribution<double> xNorm(mu_x, sigma_x);
    normal_distribution<double> yNorm(mu_y, sigma_y);
    normal_distribution<double> zNorm(mu_z, sigma_z);
    /* Generate NPRotein mu vector and also NProtein Cov Matrix using the three MVN values */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N_SPECIES; j++) {
            if (i % 3 == 0 ) {
                sampleSpace(i, j) = xNorm(generator);
            }else if (i % 3 == 1) {
                sampleSpace(i, j) = yNorm(generator);
            }else {
                sampleSpace(i,j) = zNorm(generator);
            }
        }
    }
    return sampleSpace;
}
/* calculates weight matrix */
MatrixXd calculate_weight_matrix(const MatrixXd &sample, const VectorXd &mu, int nMom, int n){
    MatrixXd inv = MatrixXd::Zero(nMom, nMom);
    VectorXd X = VectorXd::Zero(nMom);
    for(int s = 0; s < n; s++){
        for(int row = 0; row < N_SPECIES; row++){
            X(row) = sample(s, row); 
            for(int col = row; col < N_SPECIES; col++){
                if( row == col){
                    X(N_SPECIES + row) = sample(s, row) * sample(s, col);
                }else{
                    X(2*N_SPECIES + (row + col - 1)) = sample(s,row) * sample(s,col);
                }
            }
        }
        for(int i = 0; i < nMom; i++){
            for(int j = 0; j < nMom; j++){
                inv(i,j) += (X(i) - mu(i)) * (X(j) - mu(j));
            }
        }
    }
    inv /= n;
    return inv.inverse();
}