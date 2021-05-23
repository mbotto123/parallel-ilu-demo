//This code is an implementation of the Parallel ILU algorithm described in Chow and Patel (2015)
//"Fine-Grained Parallel Incomplete LU Factorization," SIAM J. Sci Comput., Vol. 37, No.2, C169-C193

//C++ and the Eigen library are used for sparse linear algebra, OpenMP is used for parallelization
//This implementation assumes the input of a square, symmetric, positive definite matrix
//Terminology used from paper: "ILU residual norm", "nonlinear residual norm"

//Developed by Marcos Botto Tornielli, May 2021

//####################################################################################################//
//######################## Parallel Incomplete LU Factorization ######################################//
//####################################################################################################//
#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iomanip>

int main()
{
    //dummy parallel region to use up the overhead from starting first region
#pragma omp parallel
    {
#pragma omp single
        std::cout << "This is thread #" << omp_get_thread_num() << std::endl;  
    }
    
    //################################################################################################//
    //##################################### Matrix I/O ###############################################//
    //################################################################################################//

    //Define common Eigen objects to be used throughout the code
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SpMatRow;
    typedef Eigen::Triplet<double> T;

    //Define timing variables
    double t1,t2;

    //Load input matrix
    SpMat A;
    Eigen::loadMarket(A, "../input/Pres_Poisson/Pres_Poisson.mtx");

    //Matrix information as read from MatrixMarket file
    //Unknown loadMarket bug causes only lower triangular part of symmetric matrix to be read
    int rows = A.rows();
    int cols = A.cols();
    int nonzs = A.nonZeros();
    double norm = A.norm();
    bool compr = A.isCompressed();
    std::cout << "Matrix is compressed: " << compr << std::endl;
    std::cout << "Number of rows    : " << rows << std::endl;
    std::cout << "Number of columns : " << cols << std::endl;
    std::cout << "Number of nonzeros: " << nonzs << std::endl;
    std::cout << "Frobenius norm    : " << norm << std::endl << std::endl;
    
    //Workaround for loadMarket symmetry issue
    //Add A to its transpose; diagonal will be divided by 2 to compensate later
    SpMat A_t = A.transpose();
    A = A + A_t;
    //Matrix information for true A matrix
    rows = A.rows();
    cols = A.cols();
    nonzs = A.nonZeros();
    norm = A.norm();
    compr = A.isCompressed();
    std::cout << "Matrix is compressed: " << compr << std::endl;
    std::cout << "Number of rows    : " << rows << std::endl;
    std::cout << "Number of columns : " << cols << std::endl;
    std::cout << "Number of nonzeros: " << nonzs << std::endl;
    std::cout << "Frobenius norm    : " << norm << std::endl << std::endl;

    //################################################################################################//
    //################################ Diagonal Scaling of A #########################################//
    //################################################################################################//

    //Scale the diagonal down by 2 for loadMarket issue workaround
    //Compute values for diagonal scaling matrix D
    //Chow and Patel assume that the matrix used in the algorithm has been scaled to have unit diagonal

    std::vector<T> dTriplets;
    dTriplets.reserve(rows);
    for (int i=0; i<A.outerSize(); ++i)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A,i); it; ++it)
        {
            if (it.row() == it.col())
            {
                it.valueRef() = it.value()/2.0;
                double diag_entry = it.value();
                double scale_diag_entry = 1.0/sqrt(diag_entry);
                dTriplets.push_back(T(it.row(),it.row(),scale_diag_entry));
            }

        }
    }

    SpMat D(A.rows(),A.cols());
    D.setFromTriplets(dTriplets.begin(),dTriplets.end());

    //Scale A matrix for unit diagonal
    A = D*A*D;

    //################################################################################################//
    //##################### Define Sparsity Structure (not supported) ################################//
    //################################################################################################//

    //This section is included for potential future development
    //But the higher ILU Levels are not currently supported by the code (5/8/21)
    //Unless this support is added, the rest of the code assumes ILU(0) sparsity structure
    
    int ilu_level = 0;
    SpMat A_sparsity = A;

    for (int i=0; i<ilu_level; i++)
    {
        A_sparsity = A_sparsity*A;
    }

    //################################################################################################//
    //################### Construct Initial Guesses for L and U ######################################//
    //################################################################################################//

    //"Standard" initial guess used:
    //L guess is the strictly lower triangular part of A with an enforced unit diagonal
    //U guess is the upper triangular part of A (also has a unit diagonal because A does)

    std::vector<T> lTriplets;
    std::vector<T> uTriplets;

    lTriplets.reserve(((A.nonZeros() - A.rows())/2.0)+A.rows());
    uTriplets.reserve(((A.nonZeros() - A.rows())/2.0)+A.rows());


    for (int i=0; i<A.outerSize(); ++i)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A,i); it; ++it)
        {
            if (it.row() == it.col())
            {
                lTriplets.push_back(T(it.row(),it.col(),1.0));
            }

            if (it.row() > it.col())
            {
                lTriplets.push_back(T(it.row(),it.col(),it.value()));
            }else{
                uTriplets.push_back(T(it.row(),it.col(),it.value()));
            }

        }
    }

    SpMatRow L(A.rows(),A.cols());
    SpMat U(A.rows(),A.cols());
    L.setFromTriplets(lTriplets.begin(),lTriplets.end());
    U.setFromTriplets(uTriplets.begin(),uTriplets.end());

    //Calculate ILU residual norm for 0 sweeps
    SpMat resM = A - L*U;
    double ilu_res_norm = resM.norm();
    std::cout << "ILU residual norm,       0 sweeps: " << ilu_res_norm << std::endl;

    //Calculate nonlinear residual norm for 0 sweeps

    double nonl_res_norm = 0.0;

    t1 = omp_get_wtime();

#pragma omp parallel for reduction(+:nonl_res_norm) schedule(dynamic,8)
    for (int i=0; i<A.outerSize(); ++i)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A,i); it; ++it)
        {
            if (it.row() > it.col())
            {
                SpMat in_prod_mat = L.block(it.row(),0,1,it.col()+1) * U.block(0,it.col(),it.col()+1,1);
                double in_prod = in_prod_mat.coeffRef(0,0);
                nonl_res_norm = nonl_res_norm + abs(it.value() - in_prod);
            }else{
                SpMat in_prod_mat = L.block(it.row(),0,1,it.row()+1) * U.block(0,it.col(),it.row()+1,1);
                double in_prod = in_prod_mat.coeffRef(0,0);
                nonl_res_norm = nonl_res_norm + abs(it.value() - in_prod); 
            }

        }
    }

    t2 = omp_get_wtime();

    std::cout << "Nonlinear residual norm, 0 sweeps: " << nonl_res_norm << std::endl;
    std::cout << "Time to compute (s)              : " << t2-t1 << std::endl << std::endl;

    //################################################################################################//
    //################################# MAIN ALGORITHM LOOP ##########################################//
    //################################################################################################//

    int n_sweeps = 5;

    for (int sweep=0; sweep<n_sweeps; sweep++)
    {
        t1 = omp_get_wtime();

#pragma omp parallel for schedule(dynamic,8)
        for (int i=0; i<A.outerSize(); ++i)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A,i); it; ++it)
            {
                if (it.row() > it.col())
                {
                    double div = 1.0/U.coeffRef(it.col(),it.col());
                    SpMat in_prod_mat = L.block(it.row(),0,1,it.col()) * U.block(0,it.col(),it.col(),1);
                    double in_prod = in_prod_mat.coeffRef(0,0);
                    L.coeffRef(it.row(),it.col()) = (it.value() - in_prod) * div;
                }else{
                    SpMat in_prod_mat = L.block(it.row(),0,1,it.row()) * U.block(0,it.col(),it.row(),1);
                    double in_prod = in_prod_mat.coeffRef(0,0);
                    U.coeffRef(it.row(),it.col()) = it.value() - in_prod; 
                }

            }
        }

        t2 = omp_get_wtime();
        std::cout << "Time to compute sweep " << sweep+1 << " (s)      : " << t2-t1 << std::endl << std::endl;


        //Calculate ILU residual norm for current number of sweeps
        resM = A - L*U;
        ilu_res_norm = resM.norm();
        std::cout << "ILU residual norm,       " << sweep+1 << " sweeps: " << ilu_res_norm << std::endl;

        //Calculate nonlinear residual norm for current number of sweeps
        nonl_res_norm = 0.0;
    
        t1 = omp_get_wtime();

#pragma omp parallel for reduction(+:nonl_res_norm) schedule(dynamic,8)
        for (int i=0; i<A.outerSize(); ++i)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A,i); it; ++it)
            {
                if (it.row() > it.col())
                {
                    SpMat in_prod_mat = L.block(it.row(),0,1,it.col()+1) * U.block(0,it.col(),it.col()+1,1);
                    double in_prod = in_prod_mat.coeffRef(0,0);
                    nonl_res_norm = nonl_res_norm + abs(it.value() - in_prod);
                }else{
                    SpMat in_prod_mat = L.block(it.row(),0,1,it.row()+1) * U.block(0,it.col(),it.row()+1,1);
                    double in_prod = in_prod_mat.coeffRef(0,0);
                    nonl_res_norm = nonl_res_norm + abs(it.value() - in_prod); 
                }

            }
        }

        t2 = omp_get_wtime();

        std::cout << "Nonlinear residual norm, " << sweep+1 << " sweeps: " << nonl_res_norm << std::endl;
        std::cout << "Time to compute (s)              : " << t2-t1 << std::endl << std::endl <<std::endl;


    }

    return 0;
}
//####################################################################################################//
//####################################################################################################//
//####################################################################################################//

