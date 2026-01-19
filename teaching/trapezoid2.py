# (C) Reza Rahemi
#this program integrates a function using trapezoidal rule
#proof: https://en.wikipedia.org/wiki/Trapezoidal_rule
# Defining the funciton here:
def f(x):
    return x**4-2*x+1
N=30 #Number of points
a=0.
b=2.
h=(b-a)/N
s=0.5*(f(a)+f(b)) #this part takes care of the (delta_x/2)*[f(x_0)+f(x_N)]
print(s)
for k in range (1,N):
    s+=f(a+k*h) #this part takes care of the sum of f(x_k) for 1<k<N
    print(h*s)  #this is the same as adding 2*(delta_x/2)*f(x_k)
