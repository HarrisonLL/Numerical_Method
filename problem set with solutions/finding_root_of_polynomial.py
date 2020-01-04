import numpy as np 
from numpy import linalg as la

## reference: 
## http://cstl-csm.semo.edu/jwojdylo/MA345/Chapter6/polyroot/polyroot.pdf
## a polynomial function takes a form as : a0 + a1*t + ... + an-1*tn
## construct a companion matrix A 
## then run following code 

tol = 10**-6
def find_root(A) :
    for i in range(100) :
        A_prev = A
        q,r = la.qr(A)
        A = r@q
    ## check if needed:
        if abs(la.norm(A-A_prev, 2)) <= tol: break
    return np.diagonal(A)
A  = np.array([[0,1,0],[0,0,1],[-6,5,2]])
print(find_root(A))
