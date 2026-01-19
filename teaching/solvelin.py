#(c) Reza Rahemi
#This example solves the system of equations 3 * x0 + x1 = 9 and x0 + 2 * x1 = 8:

import numpy as np

a = np.array([[3,1], [1,2]])
b = np.array([9,8])
x = np.linalg.solve(a, b)

print(x)
