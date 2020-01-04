import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import math
import matplotlib

##question1
def get_change(xvec1, xvec2) :
    diff = abs(xvec1 - xvec2)
    return max(diff)
xvec1 = np.array([0.0,0.7,1.0,0.0,0.7,0.2])
xvec2 = np.array([0.7,0.2,0.2,1.0,0.2,1.0])

##question2
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

##question3

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import convolve2d

image = np.random.random((80,80))

identity_filter = np.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
image0 = image.copy()
identity_image = convolve2d(image0, identity_filter, mode = 'same')
plt.imshow(identity_image, cmap = 'gray', vmin = 0, vmax = 250)

blur_filter = (1/9) * np.array([[1.0,1.0,1.0], [1.0, 1.0, 1.0], [1.0,1.0,1.0]])
image1 = image.copy()
blurred_image = convolve2d(image1, blur_filter, mode = 'same')
plt.imshow(blurred_image, cmap = 'gray', vmin = 0, vmax = 250)

sharpen_filter = 2*identity_filter-blur_filter
image2 = image.copy()
sharpened_image = convolve2d(image2, sharpen_filter, mode = 'same')
plt.imshow(sharpened_image, cmap = 'gray', vmin = 0, vmax = 250)


##question4

def create_kernel(rmin):
    # rmin: float
    # H: 2d numpy array
    # code to obtain kernel H
    N = 2*math.floor(rmin) + 1 
    H = np.zeros([N,N])
    center = float((N-1)/2)
    
    for i in range(H.shape[0]) :
        for j in range(H.shape[1]) :     
            dist = math.sqrt((float(i-center))**2 + (float(j-center))**2)
            H[i][j] = max(0, rmin-dist)
    return H

##question5

def create_kernel(rmin,nelx,nely):
  # code to obtain kernel H
  # code to obtain H1
    N = 2*math.floor(rmin) + 1 
    H = np.zeros([N,N])
    center = float((N-1)/2)
    ones = np.ones([nelx,nely])
    for i in range(N) :
      for j in range(N) :
         dist = math.sqrt((i-center)**2 + (j-center)**2)
         H[i,j] = max(0, rmin-dist)
    H1 = convolve2d(ones, H, mode = 'same')
    return H, H1

def filter_design_variable(xvec,H,H1):
    # add code here to filter xvec and return xf
    X = xvec.reshape(H1.shape[0],H1.shape[1])
    xh = convolve2d(X, H, mode = 'same')
    xf = xh / H1
    xf_reshape = xf.reshape(H1.shape[0]*H1.shape[1],)
    return xf_reshape
    
H,H1 = create_kernel(2.5,nelx,nely)
xfilt = filter_design_variable(xvec, H, H1)
X = xfilt.reshape(nelx, nely)
image_plot = plt.imshow(-X.T, cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))

##question6

# import numpy as np
# from matplotlib import colors
# import matplotlib
# import matplotlib.pyplot as plt

itercount = 0
while True:
    f = func(xvec)
    df = dfunc(xvec)
    xnew = optimizer(xvec, f, df)
    xnew = filter_design_variable(xnew)
    change = get_change(xnew, xvec)
    xvec = xnew
    itercount += 1
    if (change <= tol or itercount > maxiter) : break
X = xnew.reshape(nelx, nely)
image_plot = plt.imshow(-X.T, cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))

##question7

xnew1 = topopt(nelx=60,nely=30,volfrac=0.5,maxiter=60,tol=1e-3,rmin=1.5,ft=2)
xnew2 = topopt(nelx=60,nely=30,volfrac=0.5,maxiter=60,tol=1e-3,rmin=1.5,ft=1)
xnew3 = topopt(nelx=60,nely=30,volfrac=0.5,maxiter=60,tol=1e-3,rmin=2.5,ft=1)
xnew4 = topopt(nelx=60,nely=30,volfrac=0.5,maxiter=60,tol=1e-3,rmin=4.0,ft=1)
X1 = xnew1.reshape(60, 30)
X2 = xnew2.reshape(60, 30)
X3 = xnew3.reshape(60, 30)
X4 = xnew4.reshape(60, 30)


image_plot_1 = plt.imshow(-X1.T, cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
image_plot_2 = plt.imshow(-X2.T, cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
image_plot_3 = plt.imshow(-X3.T, cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
image_plot_4 = plt.imshow(-X4.T, cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))



##conclusion
# 这个mp主要手动完成的是filter, 为image processing/dl 的一部分。原理：将kernel matrix和data matrix移动相乘
# 链接：https://en.wikipedia.org/wiki/Kernel_(image_processing)
# 其次是optimization，略有涉及。remove noise by measuring structure stiffness
# 链接：https://en.wikipedia.org/wiki/Topology_optimization