#(c) Reza Rahemi
#Objective: To find the volume of the Space Needle tower using the Monte Carlo simulation
#Data : height = 603 ft  , base_diameter = 100 ft ,pod_baseh=100 ft,pod_toph=100 ft,
#V=V_pod+V_top+V_bottom
#In other words, we are trying to cumpute what fraction of the total cubic
#cell (i.e the known volume) is occupied by each part of the tower)
#to this end, three different counters are introduced, each corresponding
#to one of the three parts of the tower. These can be changed to better resemble the tower.

from numpy import *
from math import *
#Data:
base_diameter=100 #this is the diameter of the base (data approx. can be changed)
height = 603 #this is the height of the tower (data approx. can be changed)
pod_diameter =100 #this is the diameter of the pod (data approx. can be changed)
pod_baseh=100 #this is the height for the base of the pod (data approx. can be changed)
pod_toph =100 #this is the height for the top of the pod (data approx. can be changed)
bottom_baseh=0.0 #(ground, height=0)


#example: rand(2,1) in matlab becomes random.randn(2,1)
n_trials =10000

#Each parts can have their own counters
counter_pod = 0
counter_bottom = 0
counter_top = 0
#generating random numbers between 1 and 10:
#coords=random.randn(n_trials,3) can't use this because gives numbers greater than 1 and less than 0
coords=random.rand(n_trials,3)
#scaling coordinates to actual measured parameters
coords[:,2]=coords[:,2]*height #scale z coordinates
coords[:,0:2]=(coords[:,0:2]-0.5)*base_diameter #scale other coordinates
#We are comparing to a cubic cell not a cylinder!
known_volume=height*(base_diameter**2)

#% for the pod, sqrt(coords(i,1)^2+coords(i,2)^2)<(pod_diameter/2)
#!!! BUT:
#for a cone, we have b(z)=((h-z)/h )*b(0)
#therefore for the bottom part and the top part:
#sqrt(coords(i,1)^2+coords(i,2)^2)< ((height-(coords(i,3))/height*base_diameter)/2 )
#Otherwise, we would get a pod and two cylinders!

#Looping to check the points .

i=0
while i<n_trials:
    #for pod
    if coords[i,2]>pod_baseh and coords[i,2]<pod_toph:
        if sqrt(coords[i,0]**2+coords[i,1]**2)<(pod_diameter/2.0):
            counter_pod=counter_pod+1
    i=i+1
i=0
while i<n_trials:
    #for bottom
    if coords[i,2]>bottom_baseh and coords[i,2]<pod_baseh:
        if sqrt(coords[i,0]**2+coords[i,1]**2)<((height-(coords[i,2]/(height))*base_diameter)/2.0):
            counter_bottom=counter_bottom+1
    i=i+1
i=0
while i<n_trials:
    #for top
    if coords[i,2]>pod_toph and coords[i,2]<height:
        if sqrt(coords[i,0]**2+coords[i,1]**2)<((1-coords[i,2]/height)*base_diameter/2):
            counter_top=counter_top+1
    i=i+1

pod_volume=(counter_pod*1.0/n_trials*1.0)*known_volume*1.0
bottom_volume=(counter_bottom*1.0/n_trials*1.0)*known_volume*1.0
top_volume=(counter_top*1.0/n_trials*1.0)*known_volume*1.0
Total_Volume=pod_volume+bottom_volume+top_volume

print("The know volume of the cube is:", known_volume)
print("The pod_volume is:", pod_volume)
print("bottom_volume is:", bottom_volume)
print("top_volume is:", top_volume)
print("The Total volume of this tower is:", Total_Volume)
