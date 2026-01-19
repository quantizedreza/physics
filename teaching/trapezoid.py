#(C) Reza Rahemi

#Numerical Integrating of the function y=x^3 using the trapezoid method.        

from numpy import *
a = 0.0 # Start point
b = 3.0 # End point
N = 100 # mid-points
h = (b - a)/(N - 1) #Trapezoid rule
sum = 0.0 # Setting to 0
for j in range(1, N+1):
    x = a  +  (j-1)*h
    if ((j == 0) or (j ==  N)):   c= h/2.            # starts and ends with h/2 (other points are counted twice)
    else:   c = h
    sum  = sum  +  c * (x**3)

print('Result = ', sum)                                     # The results of the integral
