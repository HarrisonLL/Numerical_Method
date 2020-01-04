import pandas as pd
import numpy as np 
from numpy import linalg as la
import matplotlib.pyplot as plt

## see ipynb file for visulazation 
labels = ['patient ID', 'Malignant/Benign', 'radius (mean)', 'radius (stderr)', 'radius (worst)', 'texture (mean)',
          'texture (stderr)', 'texture (worst)', 'perimeter (mean)', 'perimeter (stderr)', 'perimeter (worst)',
          'area (mean)', 'area (stderr)', 'area (worst)', 'smoothness (mean)', 'smoothness (stderr)',
          'smoothness (worst)', 'compactness (mean)', 'compactness (stderr)', 'compactness (worst)', 'concavity (mean)',
          'concavity (stderr)', 'concavity (worst)', 'concave points (mean)', 'concave points (stderr)',
          'concave points (worst)', 'symmetry (mean)', 'symmetry (stderr)', 'symmetry (worst)',
          'fractal dimension (mean)', 'fractal dimension (stderr)', 'fractal dimension (worst)']

subset_labels = ['smoothness (worst)', 'concave points (stderr)', 'area (mean)', 'fractal dimension (mean)']
tumor_data = pd.io.parsers.read_csv("breast-cancer-train.dat", header=None, names=labels)


# Question 1
user_column = labels[30]
print(user_column)
plt.figure()
plt.title("label[30]")
plt.xlabel("values")
plt.ylabel("frequency")
tumor_data[user_column].hist()


# Question 2
A = np.array([[1,100],[5,600],[4,200]])
b = np.array([300,400,300])
w = la.lstsq(A,b)[0]
print(w)
print(w[0]+100*w[1])

# Question 3
def MOB(value1):
    if value1 == 'M':
        return 1
    else:
        return -1
        
col = tumor_data[labels[1]]
b = col.apply(MOB)
b = np.array(b, dtype = 'float64')
print(b)

# Question 4
subset_labels = labels[:10]
A_linear = []
for label in subset_labels: 
    A_lineari = tumor_data[label].values.tolist()
    A_linear.append(A_lineari)
A_linear = np.array(A_linear).T
print(A_linear)

# Question 5
A = []
for label in subset_labels:
    A.append(tumor_data[label])
for label in subset_labels:
    A.append(tumor_data[label]**2)

for i in range(len(subset_labels)-1) :
    for j in range(i+1, len(subset_labels)) :
        A.append(tumor_data[subset_labels[i]] * tumor_data[subset_labels[j]])
A_quad = np.array(A).T
print(A_quad)


# Question 6
def lls_solve(A,b) :
    u,s,vt = la.svd(A,full_matrices=False)
    z = u.T@b
    y = np.zeros(len(s), dtype = 'float64')
    for i,v in enumerate(z):
        if s[i] >  0.00000000001:
            y[i] = float(v)/(s[i])
    x = la.solve(vt, y)
    return x
    
        
weights_linear = lls_solve(A_linear,b)
weights_quad = lls_solve(A_quad,b)

# Load the data
validate_data = pd.io.parsers.read_csv("breast-cancer-validate.dat", header=None, names=labels)

A_val_linear = []

for label in labels[2:32]: 
    A_lineari = validate_data[label].values.tolist()
    A_val_linear.append(A_lineari)
A_val_linear = np.array(A_val_linear).T

A = []
for label in subset_labels:
    A.append(validate_data[label])
for label in subset_labels:
    A.append(validate_data[label]**2)

for i in range(len(subset_labels)-1) :
    for j in range(i+1, len(subset_labels)) :
        A.append(validate_data[subset_labels[i]] * validate_data[subset_labels[j]])
A_val_quad = np.array(A).T

def MOB2(value1):
    if value1 == 'M':
        return 1
    else:
        return -1
        
col = validate_data[labels[1]]
b = col.apply(MOB2)
b = np.array(b, dtype = 'float64')

p_linear = A_val_linear@weights_linear
p_quad = A_val_quad@weights_quad

fp_linear = ((b<0) & (p_linear>0)).sum()
fn_linear = ((b>0) & (p_linear<0)).sum()
fp_quad = ((b<0) & (p_quad>0)).sum()
fn_quad = ((b>0) & (p_quad<0)).sum()

graph=bar_graph(fp_linear, fn_linear, fp_quad, fn_quad))

