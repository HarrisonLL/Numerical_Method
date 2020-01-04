import numpy as np
import scipy.linalg as la




def reorder_rows_columns(K,equation_numbers):
    # construct the matrix Khat
    P = np.zeros(K.shape)
    for i, v in enumerate(equation_numbers) :
        P[i,v-1] = 1
    Khat = P@K@(P.T)
    return Khat

def partition_stiffness_matrix(K,equation_numbers,nk):
    # construct the smaller matrices
    Khat = reorder_rows_columns(K,equation_numbers)
    Kpp = Khat[:nk, :nk]
    Kpf = Khat[:nk, nk:]
    Kfp = Khat[nk:, :nk]
    Kff = Khat[nk:, nk:]
    return  Kpp,Kpf,Kfp,Kff


def my_lu(A):
    # The upper triangular matrix U is saved in the upper part of the matrix M (including the diagonal)
    # The lower triangular matrix L is saved in the lower part of the matrix M (not including the diagonal)
    # Do NOT use `scipy.linalg.lu`!
    # You should not use pivoting
    n = A.shape[0]
    A = np.asarray(A)
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    M = A.copy()
    for i in range(n) :
        U[i, i:] = M[i, i:]
        L[i:,i] = M[i:,i]/U[i,i]
        M[i+1:,i+1:] -= np.outer(L[i+1:,i],U[i,i+1:])
        
    for i in range(n) :
        for j in range(n) :
            if i!=j and L[i,j] != 0:
                M[i,j] = L[i,j]
    return M

def my_triangular_solve(M, b):

    n = M.shape[0]
    y = []
    x = [None]*n
    L = np.zeros(M.shape, dtype = float)
    U = np.zeros(M.shape, dtype = float)
    for i in range(n) :
        for j in range(n) :
            if i == j : L[i,j] = 1.0
            if i > j :L[i,j] = M[i,j]
            if i <= j : U[i,j] = M[i,j]

    ## Ly = b solve for y (Forward-substitution)
    y.append(b[0]/L[0,0])
    for i in range(1,n) :
        sum_ = 0.0
        for j in range(0,i) :
            sum_ += L[i,j]*y[j]
        value = (b[i] - sum_)/L[i,i]
        y.append(value)
    
    ## Ux = y solve for x (Backward-substitution)
    x[n-1] = (y[n-1]/U[n-1, n-1])
    for i in range(n-2, -1, -1) :
        sum_ = 0.0
        for j in range(i+1, n) :
            sum_ += U[i,j]*x[j]
        value = (y[i] - sum_)/U[i,i]
        x[i] = value
    
    x = np.array(x)
    return x


def fea_solve(Kpp,Kpf,Kfp,Kff,xp,Ff):
    # do stuff here
    temp = Ff - Kfp@xp
    M = my_lu(Kff)
    xf = my_triangular_solve(M, temp)
    Fp = Kpp@xp + Kpf@xf
    return xf,Fp

