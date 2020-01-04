import numpy as np
import matplotlib.pyplot as plt

# import numpy as np
import scipy
# import matplotlib.pyplot as plt
from numpy import linalg as la
from numpy import sin,cos,sqrt
import sympy

# def gs(a,b,f) :
#     brackets = []
#     gs = (np.sqrt(5) - 1) / 2
#     m1 = a + (1 - gs) * (b - a)
#     m2 = a + gs * (b - a)

#     # Begin your modifications below here
#     f1 = f(m1)
#     f2 = f(m2)
#     while True:

#         if (f1>f2) :
#             a = m1
#             m1 = m2
#             m2 = a + gs*(b-a)
#             f1 = f2
#             f2 = f(m2)
            
#         else:
#             b = m2
#             m2 = m1
#             m1 = a + (1-gs)*(b-a)
#             f2 = f1
#             f1 = f(m1)
        
#         brackets.append([a, m1, m2, b])
#         if (b-a) < 10**-5: break
#     # End your modifications above here
#     return brackets




# def f(r):
#     x, y = r
#     return 3 +((x**2)/8) + ((y**2)/8) - np.sin(x)*np.cos((2**-0.5)*y)
    
# def obj_f(alpha, x, s):
#     return f(x+alpha*s)

# def gred(r):
#     x, y = r
#     ret1 = x/4-cos(x)*cos((sqrt(2)/2)*y)
#     ret2 = y/4+sin(x)*sin((sqrt(2)/2)*y)*(sqrt(2)/2)
#     return np.array([ret1, ret2])
    
# def Hessian(r):
#     x, y = r
#     ret11 = 1/4 + sin(x)*cos((sqrt(2)/2)*y)
#     ret12 = cos(x)*(sqrt(2)/2)*sin((sqrt(2)/2)*y)
#     ret21 = cos(x)*sin((sqrt(2)/2)*y)*(sqrt(2)/2)
#     ret22 = (1/4) + (1/2)*sin(x)*cos((sqrt(2)/2)*y)
#     return np.array([[ret11,ret12],[ret21,ret22]])
    
# r = r_init.copy()
# iteration_count_newton = 0
# list_newton = []
# while True:
#     gredf = gred(r)
#     if la.norm(gredf,2) < stop: break
#     H = Hessian(r)
#     s = la.solve(H,-gredf)
#     r += s
#     iteration_count_newton += 1
#     list_newton.append(r.tolist())
# r_newton = r.copy() 

# r2 = r_init.copy()
# iteration_count_sd = 0
# list_sd = []
# while True:
#     gredf2 = gred(r2)
#     if la.norm(gredf2,2) < stop: break
#     alpha = scipy.optimize.minimize_scalar(obj_f, args = (r2, gredf2)).x
#     r2 += alpha*gredf2
#     iteration_count_sd += 1
#     list_sd.append(r2.tolist())
# r_sd = r2.copy()

# print(list_newton)
# print(list_sd)

# error_n = [la.norm(r_init-r_newton)]
# error_s = [la.norm(r_init-r_sd)]
# for i in range(iteration_count_newton):
#     error_n.append(la.norm(list_newton[i]-r_newton))
# for j in range(iteration_count_sd) :
#     error_s.append(la.norm(list_sd[j]-r_sd))
# error_n = np.log(error_n)
# error_s = np.log(error_s)

# plt.figure
# plt.plot(range(iteration_count_newton + 1), error_n)
# plt.plot(range(iteration_count_sd + 1), error_s)
# plt.title('Newton vs SD')
# plt.legend()
# plt.xlabel('Numbers of iterations')
# plt.ylabel('Error')

### HW10.16

print(np.array([-36,-43])@ la.inv(np.array([[6,2],[2,11]])))


## HW10.18
# def f(r) :
#     x,y = r
#     return(3+(x**2)/8+(y**2)/8-sin(x)*cos(sqrt(2)*y/2)) 
def df(r):
    sympy.var('x')
    sympy.var('y')
    f = 3+(x**2)/8+(y**2)/8-sympy.sin(x)*sympy.cos(sympy.sqrt(2)*y/2)
    dfx = f.diff(x,1).subs(x,r[0]).subs(y,r[1])
    dfy = f.diff(y,1).subs(x,r[0]).subs(y,r[1])
    return np.array([dfx, dfy], dtype = float)

def H(r):
    x = sympy.var('x')
    y = sympy.var('y')
    f = 3+(x**2)/8+(y**2)/8-sympy.sin(x)*sympy.cos(sympy.sqrt(2)*y/2)
    H = np.zeros((2,2))
    H[0,0] = f.diff(x,2).subs(x,r[0]).subs(y,r[1])
    H[0,1] = f.diff(x,1).diff(y,1).subs(x,r[0]).subs(y,r[1])
    H[1,0] = f.diff(y,1).diff(x,1).subs(x,r[0]).subs(y,r[1])
    H[1,1] = f.diff(y,2).subs(x,r[0]).subs(y,r[1])
    return H

r = [ np.pi/3, np.pi/(2*np.sqrt(2))]
print(df(r))
print(H(r))
print(r)

print(sympy.var('x').subs(x, 1))
