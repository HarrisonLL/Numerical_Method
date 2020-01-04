import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

## code past test cases
## copyed and pasted for review purpose 

###Traveling by Train
print("HW8.18")
mat = np.array([[0,0,0,1/2,1],[1/4,0,1/3,1/2,0],[1/4,1/2,0,0,0],[1/4,0,1/3,0,0],[1/4,1/2,1/3,0,0]])

prob = np.array([1/5,1/5,1/5,1/5,1/5])
prob_prev = np.zeros(5)
while abs(la.norm(prob_prev) - la.norm(prob)) > 10**-17:
    prob_prev = prob
    prob = mat@prob

##Drug Metabolism Markov Model
print("HW8.19")
M = np.array([[0.7,0,0,0],[0.1,0.5,0.1,0],[0.1,0.4,0.8,0],[0.1,0.1,0.1,1]])
ini = np.array([1,0,0,0])

hours = 0
while ini[3] <= 0.85:
    ini = M@ini
    hours += 1
##Implementing Secant Method for 1D Problem
print("HW9.14")
xks = [2,3]
roots = []
xk = xks[0]
xkp1 = xks[1]
for i in range(5) :
    df = (f(xkp1) - f(xk))/(xkp1 - xk)
    xk = xkp1
    xkp1 = xkp1 - f(xkp1)/df
    roots.append(xkp1)
roots = np.array(roots)

##Bisection Method for Root-Finding in 1D
print("HW9.15")
def bisection(a,b,n_iter,epsilon) :
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
    
roots = []
for i in range(len(intervals)) :
    a = intervals[i][0]
    b = intervals[i][1]
    roots.append(bisection(a,b,n_iter,epsilon))
print(roots)

##Newton’s Method for Root-Finding in 2-d
print("HW9.16")
xi = np.array([0,1])
tol = 10**-6


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

##Local Convergence of Newton’s Method
print("HW9.17")

iterations_to_converge = []
Xs = np.zeros((len(X0), 10))
iteration_limit = 100

def f917(x):
    return 0.5 * x * np.sin(2 * x) - 0.8 * x + 1

def dx917(x):
    return 0.5 * np.sin(2*x) + x*np.cos(2*x) - 0.8

for i in range(len(X0)) :
    X0_new = X0[i]
    for j in range(100) :
        X0_prev = X0_new
        X0_new = X0_prev - f917(X0_prev)/dx917(X0_prev)
        if j<10:
            Xs[i,j] = X0_new
        if abs(X0_new-X0_prev) < tol:
            break
    iterations_to_converge.append(j+1)


# Use the following code to plot the result.
plt.figure(figsize = (8,6))
plot_x = np.linspace(-5, 5, 500)
plt.plot(plot_x, f917(plot_x), label = "fx = 0.5x * sin(2x) - 0.8 * x + 1")
plt.plot(plot_x, [0 for i in range(len(plot_x))], label="f(x) = 0")
plt.scatter(X0, f917(X0), c = [1 - t/100 for t in iterations_to_converge], label = "heat-map of times to converge")
for i in range(len(X0)):
    plt.text(X0[i], f917(X0[i])+ (-1)**(i+1) * 0.5, str(iterations_to_converge[i]), ha="center", va="center", color="black")
plt.legend()
plt.title("Local convergence of Newton's method")
plt.show()

