import numpy as np
import scipy 
from scipy.signal import convolve2d
import math
import matplotlib.pyplot as plt
from matplotlib import colors


def get_change(xvec1, xvec2) :
    diff = abs(xvec1 - xvec2)
    return max(diff)

def func(xvec):
    # xvec: 1d numpy array
    # f: float
    # code to obtain f
    # return f
    pass 
    
def dfunc(x, dx):
    # ADD CODE HERE
    df = []
    for i,v in enumerate(x):
        new_x = x.copy()
        new_x[i] += dx
        df_ = (func(new_x)-func(x))/dx
        df.append(df_)
    df = np.array(df)
    return df

def create_kernel(rmin):
     N = 2*math.floor(rmin) + 1 
     H = np.zeros((N,N))
     center = float(N-1)/2
     for i in range(N) :
         for j in range(N) :
             dist = math.sqrt((i-center)**2 + (j-center)**2)
             H[i,j] = max(0, rmin-dist)
     return H

def filter_design_variable(xvec,H,H1):
    # add code here to filter xvec and return xf
    X = xvec.reshape(H1.shape[0],H1.shape[1])
    xh = convolve2d(X, H, mode = 'same')
    xf = xh / H1
    xf_reshape = xf.reshape(H1.shape[0]*H1.shape[1],)
    return xf_reshape

def func2(xvec):
    # xvec: 1d numpy array
    # f: float
    # return f
    pass

def dfunc2(xvec):
    # xvec: 1d numpy array
    # df: 1d numpy array
    # return df
    pass 

def filter_design_variable2(xvec):
    # xvec: 1d numpy array
    # xfilt: 1d numpy array
    # return xfilt
    pass

def optimizer2(xvec,f,df):
    # xvec: 1d numpy array
    # f: float
    # df: 1d numpy array
    # xnew: 1d numpy array
    # return xnew
    pass

def Optimization(xvec,tol,maxiter):
    itercount = 0
    while True:
        f = func2(xvec)
        df = dfunc2(xvec)
        xnew = optimizer2(xvec, f, df)
        xnew = filter_design_variable2(xnew)
        change = get_change(xnew, xvec)
        xvec = xnew
        itercount += 1
        if (change <= tol or itercount > maxiter) : break
    return xnew