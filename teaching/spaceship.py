import numpy
import math
from math import *
from numpy import *

#for plotting
import matplotlib.pyplot as plt
#Define timeline
dt=0.5
#timeline = [0:dt:20];
timeline=numpy.arange(0, 21, dt)
#constants
m=70
P=2000
V0=5
V0W=5
C=1
A=3
rho=1.3
nu=2*(1e-5)
h=1
nuWater=1*(1e-3)
#educaitonal purpose:
length_timeline=len(timeline)
size_timeline=size(timeline)
#velocity-No Air resistance
VN =zeros(size(timeline))
VN[0]=V0
#Velocity - with Air resistance (without stokes parameter)
VD=zeros(size(timeline))
VD[0]=V0
#Velocity - with Air resistance + Stokes parameters
VS=zeros(size(timeline));
VS[0]=V0;


#The loop

i=1
while i<len(timeline):
    VN[i]=VN[i-1]+(P/(m*VN[i-1]))*dt
    #VD[i]=VD[i-1]+P*dt/(m*VD[i-1])-0.5*C*rho*A*(VD[i-1]**2)*dt/m
    #VS[i]=VS[i-1]+P*dt/(m*VS[i-1])-0.5*C*rho*A*(VS[i-1]**2)*dt/m-dt*nu*A*VS[i-1]/(h*m)
    i=i+1

#CLF in matlab, not envoked here.

#Uphil motion of the rocket

g=9.807

#Velocity for going hphill
#given info: Slope: 0.1 where slope is tan(theta). Therefore theta=atan(0.1)

VH=zeros(size(timeline))
#Initial velocity
VH[1]=V0*cos(atan(0.1))

i=1
while i<len(timeline):
    #VH[i]=(VH[i-1]+P*dt/(m*VH[i-1])-0.5*C*rho*A*(VH[i-1]**2)*dt/m-dt*nu*A*VH[i-1]/(h*m))-g*sin(atan(0.1))*dt
    i=i+1

plt.plot(timeline,VN)
plt.ylabel('Velocity, No Air Resistance')
plt.xlabel('Timeline')
plt.title('Velocity of rocket vs time')
plt.show()
