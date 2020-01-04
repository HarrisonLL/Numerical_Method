###########11##############
import numpy as np
from numpy import linalg as la


window_size = waldo_image.shape
norm_dict = {}

print(waldo_image.shape)
for i in range(len(group_image)) :
    for j in range(len(group_image[0])) :
        
        if ((i+window_size[0]) < len(group_image)) and ((j+window_size[1]) < len(group_image[0])) :
            temp = group_image[i:i+window_size[0], j:j+window_size[1]]
            difference = temp - waldo_image
            norm2 = la.norm(difference)
            norm_dict[(i,j)] = norm2
 

norm_dict = sorted((v,k) for (k,v) in norm_dict.items())
min_diff = norm_dict[0][0]
top_left = norm_dict[0][1]


###########12###############

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

# It may be helpful to complete this function
def norm(x, p):
    """Computes norms.
        input a 2*n array
        output a 1*n norms 
    """
    return np.sum(np.abs(x)**p, axis=0)**(1/p)

ps = [1, 2, 5, 0.5]
phi = np.linspace(0, 2 * np.pi, 500)
unit_circle = np.array([np.cos(phi), np.sin(phi)])

p1 = ps[0]
norm_unit_circle = (unit_circle/(norm(unit_circle, p1)))*r
plt.figure()
plt.plot(norm_unit_circle[0], norm_unit_circle[1])
fig_1 = plt.gca()


p2 = ps[1]
norm_unit_circle2 = (unit_circle/(norm(unit_circle, p2)))*r
plt.figure()
plt.plot(norm_unit_circle2[0], norm_unit_circle2[1])
fig_2 = plt.gca()

p3 = ps[2]
norm_unit_circle3 = (unit_circle/(norm(unit_circle, p3)))*r
plt.figure()
plt.plot(norm_unit_circle3[0], norm_unit_circle3[1])
fig_3 = plt.gca()

p4 = ps[3]
norm_unit_circle4 = (unit_circle/(norm(unit_circle, p4)))*r
plt.figure()
plt.plot(norm_unit_circle4[0], norm_unit_circle4[1])
fig_4 = plt.gca()