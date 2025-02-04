import numpy as np
import os
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D


dataA = pd.read_csv(os.path.join(os.path.dirname(__file__),"dataA.csv"),header=None)
dataB = pd.read_csv(os.path.join(os.path.dirname(__file__),"dataB.csv"),header=None)

dataA = dataA.dropna(how='all', axis='columns')
dataB = dataB.dropna(how='all', axis='columns')

dataA = dataA.to_numpy()
dataB = dataB.to_numpy()

#dataA = dataA/16k
#dataB = dataB/16k

a = np.array(dataA[:,0])
b = np.array(dataA[:,1])
c = np.array(dataA[:,2])

d = np.array(dataB[:,0])
e = np.array(dataB[:,1])
f = np.array(dataB[:,2])

a = a.reshape(len(a),1)

abc = np.column_stack((a,b,c))
abc = abc.T #want a row stack

fig = plt.figure()
ax = plt.axes()

fig2 = plt.figure()
ax2 = plt.axes()



def rotation_matrix(phi, theta):
    init_matrix = np.array([  -1*math.sin(theta)*math.cos(phi), math.sin(phi), math.cos(theta)*math.cos(phi)   ])
    return init_matrix.reshape(1,3)

def roll_rotation_matrix(theta):
    roll_matrix = np.array([[math.cos(theta), 0, math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
    roll_matrix = roll_matrix.reshape(3,3)

    return roll_matrix

def pitch_rotation_matrix(theta):
    
    pitch_matrix = np.array([[1, 0, 0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
    pitch_matrix = pitch_matrix.reshape(3,3)
    return pitch_matrix

def pitch_roll_matrix(theta,phi):
    matrix = np.array([[math.cos(phi), 0, math.sin(phi)],[math.sin(theta)*math.sin(phi),math.cos(theta),-math.sin(theta)*math.cos(phi)],[-math.cos(theta)*math.sin(phi),math.sin(theta),math.cos(theta)*math.cos(phi)]])
    matrix = matrix.reshape(3,3)
    return matrix

def plot_vec(axis, array, color):
    axis.plot(np.linspace(0,2,len(array)),array, color = color)

pitch = 0
roll = 0

#pitch = pitch*2

roll = roll* 3.14/180
pitch = pitch* 3.14/180

#matrix = roll_rotation_matrix(roll).dot(pitch_rotation_matrix(pitch))
#matrix = pitch_rotation_matrix(pitch).dot(roll_rotation_matrix(roll))
matrix = pitch_roll_matrix(pitch,roll)
print(matrix)
rotation_result = matrix.dot(abc)


plot_vec(ax,a,"red")
plot_vec(ax,d,"green")
plot_vec(ax,rotation_result[0],"pink")


plot_vec(ax2,c,"red")
plot_vec(ax2,f,"green")
plot_vec(ax2,rotation_result[2],"pink")

w = 102
distad, paths = dtw.warping_paths(a , d, window=w, psi=0)
distcf, paths = dtw.warping_paths(c , f, window=w, psi=0)

rotad, paths = dtw.warping_paths(rotation_result[0] , d, window=w, psi=0)
rotcf, paths = dtw.warping_paths(rotation_result[2] , f, window=w, psi=0)

print(distad,distcf,rotad,rotcf)
print(distad-rotad,distcf-rotcf)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(c, f, paths, best_path)

plt.show()