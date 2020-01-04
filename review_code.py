import numpy as np
from numpy import linalg as la
from numpy import sin, cos, sqrt
import scipy
import scipy.linalg as sla
import sympy ## math


### eigen values and eigen vectors 
print("power iteration")

def power_iter(A):
    n = A.shape[1]
    x_k = np.random.randn(n)
    x_k = x_k/la.norm(x_k) 
    while True:
    # for i in range(40):
        x_prev = x_k
        x_k = A@x_k
        x_k = x_k/la.norm(x_k)
        if (la.norm(abs(x_k-x_prev)) <= 1e-15) : break
    return x_k

n = 5
X = np.random.rand(n,n)
U,_ = sla.qr(X)
D = np.diag([6,2,4,7,6])
A = U@D@U.T

print(power_iter(A))
print(la.eig(A))

## Rayleigh Quotatient iteration
## find eigen value close to sigma

def Rayleigh (A, sigma):
    sigmak = np.random.randn(1)[0]
    n = A.shape[0]
    I = np.ones(n)
    I = np.diag(I)
    x = np.random.randn(n)
    
    for i in range(40):
        B = A - sigmak*I
        P,L,U = sla.lu(B)
        y = sla.solve_triangular(L,np.dot(P.T, x), lower=True)
        x = sla.solve_triangular(U,y)
        sigmak = x.T@A@x/(x.T@x)

    return sigmak





### Nonlinear
## Secant
roots = []
xk = xks[0]
xkp1 = xks[1]
for i in range(5) :
    df = (f(xkp1) - f(xk))/(xkp1 - xk)
    xk = xkp1
    xkp1 = xkp1 - f(xkp1)/df
    roots.append(xkp1)
roots = np.array(roots)

## bisection

def bisection(a,b,n_iter,epsilon,function) :
    if a >= b or function(a)*function(b) > 0 : 
        return None
    m = float(a+b)/2
    count = 1
    while (abs(function(m)) > epsilon) :
        if function(m) * function(a) > 0: 
            a = m
        else : 
            b = m
        m = float(a+b)/2
        count += 1
        if count > n_iter: 
            return None
    return m

## newton in nd
def f(x,y):
    return np.array([x**3 - y**2, x+y*x**2 - 2])

def J(x,y):
    return np.array([[3*x**2, -2*y],[1+2*x*y, x**2]])

x = xi[0]
y = xi[1]
while (la.norm(f(x,y),2) >= tol) :
    b = -(f(x,y))
    Joc = J(x,y)
    s = la.solve(Joc,b)
    x = x + s[0]
    y = y + s[1]
root = np.array([x,y])
res = la.norm(f(x,y),2)


### Optimization 
## golden section search (1D)

brackets = []
gs = (np.sqrt(5) - 1) / 2
m1 = a + (1 - gs) * (b - a)
m2 = a + gs * (b - a)

f1 = f(m1)
f2 = f(m2)
while True:

    if (f1>f2) :
        a = m1
        m1 = m2
        m2 = a + gs*(b-a)
        f1 = f2
        f2 = f(m2)
        
    else:
        b = m2
        m2 = m1
        m1 = a + (1-gs)*(b-a)
        f2 = f1
        f1 = f(m1)
    
    brackets.append([a, m1, m2, b])
    if (b-a) < 10**-5: break

##  Newton
## gredient
def Gredient(r):
    x = sympy.var('x')
    y = sympy.var('y')
    f = 3+(x**2)/8+(y**2)/8-sympy.sin(x)*sympy.cos(sympy.sqrt(2)*y/2)
    gred = np.zeros(2)
    gred[0] = f.diff(x,1).subs(x,r[0])
    gred[1] = f.diff(y,1).subs(y,r[1])
    return gred
    
## Hessian
def Hessian(r):
    x = sympy.var('x')
    y = sympy.var('y')
    f = 3+(x**2)/8+(y**2)/8-sympy.sin(x)*sympy.cos(sympy.sqrt(2)*y/2)
    H = np.zeros(2,2)
    H[0,0] = f.diff(x,2).subs(x, r[0]).subs(y, r[1])
    H[0,1] = f.diff(x,1).diff(y,1).subs(x, r[0]).subs(y, r[1])
    H[1,0] = f.diff(y,1).diff(x,1).subs(x, r[0]).subs(y, r[1])
    H[1,1] = f.diff(x,2).subs(x, r[0]).subs(y, r[1])
    return H


## Newton in N-D and Steepest Descent
def f(r):
    x, y = r
    return 3 +((x**2)/8) + ((y**2)/8) - np.sin(x)*np.cos((2**-0.5)*y)
    
def obj_f(alpha, x, s):
    return f(x+alpha*s)

def gred(r):
    x, y = r
    ret1 = x/4-cos(x)*cos((sqrt(2)/2)*y)
    ret2 = y/4+sin(x)*sin((sqrt(2)/2)*y)*(sqrt(2)/2)
    return np.array([ret1, ret2])
    
def Hessian(r):
    x, y = r
    ret11 = 1/4 + sin(x)*cos((sqrt(2)/2)*y)
    ret12 = cos(x)*(sqrt(2)/2)*sin((sqrt(2)/2)*y)
    ret21 = cos(x)*sin((sqrt(2)/2)*y)*(sqrt(2)/2)
    ret22 = (1/4) + (1/2)*sin(x)*cos((sqrt(2)/2)*y)
    return np.array([[ret11,ret12],[ret21,ret22]])
    
r = r_init.copy()
iteration_count_newton = 0
list_newton = []
while True:
    gredf = gred(r)
    if la.norm(gredf,2) < stop: break
    H = Hessian(r)
    s = la.solve(H,-gredf)
    r += s
    iteration_count_newton += 1
    list_newton.append(r.tolist())
r_newton = r.copy() 

r2 = r_init.copy()
iteration_count_sd = 0
list_sd = []
while True:
    gredf2 = -gred(r2)
    if la.norm(gredf2,2) < stop: break
    alpha = scipy.optimize.minimize_scalar(obj_f, args = (r2, gredf2)).x0
    r2 += alpha*gredf2
    iteration_count_sd += 1
    list_sd.append(r2.tolist())
r_sd = r2.copy()

print(list_newton)
print(list_sd)    


## svd solve linear least square 
def lst_sq(A,b) :

    u,s,vt = la.svd(A,full_matrices=False)
    z = u.T@b
    y = np.zeros(len(s), dtype = 'float64')
    for i,v in enumerate(z):
        if s[i]> 1e-15:
            y[i] = float(v)/(s[i])
    x = la.solve(vt, y)

    return x

## pca

X = np.array([[0.4281198369, -0.9162867472, 2.1726314829], [-1.3305346185, 0.2509328100, 2.4048395132], [0.8927572934, 0.6473473661, -4.5512262718], [0.0782591433, -0.2328813346, 0.7843094947], [-0.7418688106, 1.7917188424, -4.9153901781], [0.6732671555, -1.5408309366, 4.1048359592]])

mean = X.sum(axis=0)/X.shape[0]
mean = np.array(mean)
X = X-mean
Y = (1/np.sqrt(X.shape[0]-1))*X
u, s, vt = la.svd(Y, full_matrices = False)

p1 = X@vt[0] / la.norm(vt[0])
p2 = X@vt[1] / la.norm(vt[1])

n = X.shape[0]
cor_matrix = (1/(n-1))*(X.T@X)

