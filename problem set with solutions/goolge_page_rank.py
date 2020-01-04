import numpy as np 
from numpy import linalg as la

def google_matrix(M, alpha) :
    n = M.shape[1]
    col_sum = M.sum(axis=0)
    G = np.zeros(shape=(n,n))
    for i,v in enumerate(col_sum):
        if v != 0:
            G[:,i] = M[:,i]/v
        else :
            G[:,i] = 1/n 
    G = alpha*G + (1-alpha)/n
    print(G)
    return G

def page_rank(G,alpha,tol):
    n = G.shape[1]
    probs = np.random.rand(n)
    probs = probs / probs.sum(axis=0)
    # also can be written as:  probs = probs/la.norm(probs,1)
    while True:
        probs_prev = probs
        probs = G@probs
        if (abs(la.norm(probs-probs_prev,2)) <= tol) : break
    return probs


string_list = ['a','b','c','d']
G = google_matrix(M, alpha=0.85)
probs = page_rank(G,alpha=0.85,tol=10**-6)
string_dict = {}
for i,v in enumerate(probs):
    string_dict[string_list[i]] = v
string_sorted_list = sorted(string_dict.items(), key=lambda x:(-x[1],x[0]))
string_sorted_list = [item[0] for item in string_sorted_list]


