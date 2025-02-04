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

def dtw(series_1, series_2, norm_func = np.linalg.norm):
	matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
	matrix[0,:] = np.inf
	matrix[:,0] = np.inf
	matrix[0,0] = 0
	for i, vec1 in enumerate(series_1):
		for j, vec2 in enumerate(series_2):
			cost = norm_func(vec1 - vec2)
			matrix[i + 1, j + 1] = cost + min(matrix[i, j + 1], matrix[i + 1, j], matrix[i, j])
	matrix = matrix[1:,1:]
	i = matrix.shape[0] - 1
	j = matrix.shape[1] - 1
	matches = []
	mappings_series_1 = [list() for v in range(matrix.shape[0])]
	mappings_series_2 = [list() for v in range(matrix.shape[1])]
	while i > 0 or j > 0:
		matches.append((i, j))
		mappings_series_1[i].append(j)
		mappings_series_2[j].append(i)
		option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
		option_up = matrix[i - 1, j] if i > 0 else np.inf
		option_left = matrix[i, j - 1] if j > 0 else np.inf
		move = np.argmin([option_diag, option_up, option_left])
		if move == 0:
			i -= 1
			j -= 1
		elif move == 1:
			i -= 1
		else:
			j -= 1
	matches.append((0, 0))
	mappings_series_1[0].append(0)
	mappings_series_2[0].append(0)
	matches.reverse()
	for mp in mappings_series_1:
		mp.reverse()
	for mp in mappings_series_2:
		mp.reverse()
	
	return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix
    


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

# matches, cost_x, mapping_1, mapping_2, matrix = dtw(a, d)
# matches, cost_y, mapping_1, mapping_2, matrix = dtw(b, e)
# matches, cost_z, mapping_1, mapping_2, matrix = dtw(c, f)

# print("Raw DTW",cost_x, cost_y, cost_z)

a_centered = a - (np.ones(len(a))*np.mean(a))
b_centered = b - (np.ones(len(b))*np.mean(b))
c_centered = c - (np.ones(len(c))*np.mean(c))
d_centered = d - (np.ones(len(d))*np.mean(d))
e_centered = e - (np.ones(len(e))*np.mean(e))
f_centered = f - (np.ones(len(f))*np.mean(f))

# matches, cost_x, mapping_1, mapping_2, matrix = dtw(a_centered, d_centered)
# matches, cost_y, mapping_1, mapping_2, matrix = dtw(b_centered, e_centered)
# matches, cost_z, mapping_1, mapping_2, matrix = dtw(c_centered, f_centered)

# print("Centered DTW",cost_x, cost_y, cost_z)


#debug rotation printout
# a = np.array(dataA[:,0])
# b = np.array(dataA[:,1])
# c = np.array(dataA[:,2])

# theta = math.radians(15)

# a_rotated = a*math.cos(theta) + c*math.sin(theta)
# b_rotated = b
# c_rotated = c*math.cos(theta) - a*math.sin(theta)

# a_rotated_centered = a_rotated - (np.ones(len(a_rotated))*np.mean(a_rotated))
# c_rotated_centered = c_rotated - (np.ones(len(c_rotated))*np.mean(c_rotated))


# fig = plt.figure()
# ax = plt.axes()
# #ax.plot(np.linspace(0,2,len(a_centered)),a_centered, color = "red")
# ax.plot(np.linspace(0,2,len(d_centered)),d_centered, color = "orange")

# ax.plot(np.linspace(0,2,len(a_rotated_centered)),a_rotated_centered, color = "green")
# #ax.plot(np.linspace(0,2,len(c_rotated_centered)),c_rotated_centered, color = "pink")

# matches, cost_d_to_a_rotated, mapping_1, mapping_2, matrix = dtw(a_rotated_centered, d_centered)

# print("DTW D to rotated A",cost_d_to_a_rotated)

# ########################################## C to F
# theta = math.radians(135)

# a_rotated = a*math.cos(theta) + c*math.sin(theta)
# b_rotated = b
# c_rotated = c*math.cos(theta) - a*math.sin(theta)

# a_rotated_centered = a_rotated - (np.ones(len(a_rotated))*np.mean(a_rotated))
# c_rotated_centered = c_rotated - (np.ones(len(c_rotated))*np.mean(c_rotated))

# fig2 = plt.figure()
# ax2 = plt.axes()
# #ax.plot(np.linspace(0,2,len(a_centered)),a_centered, color = "red")
# ax2.plot(np.linspace(0,2,len(f_centered)),f_centered, color = "blue")

# ax2.plot(np.linspace(0,2,len(c_rotated_centered)),c_rotated_centered, color = "black")
# #ax.plot(np.linspace(0,2,len(c_rotated_centered)),c_rotated_centered, color = "pink")

# matches, cost_c_to_f_rotated, mapping_1, mapping_2, matrix = dtw(c_rotated_centered, f_centered)

# print("DTW C to rotated F",cost_c_to_f_rotated)



lowest_dtw = 10000000000000000 #stupid
est_angle = 0
# step = 360
# for i in range(0,360,step):
	
#     theta = math.radians(i)
#     a_rotated = a*math.cos(theta) + c*math.sin(theta)
#     b_rotated = b
#     c_rotated = c*math.cos(theta) - a*math.sin(theta)

#     a_rotated_centered = a_rotated - (np.ones(len(a_rotated))*np.mean(a_rotated))
#     c_rotated_centered = c_rotated - (np.ones(len(c_rotated))*np.mean(c_rotated))
	
#     matches, cost_d_to_a_rotated, mapping_1, mapping_2, matrix = dtw(a_rotated_centered, d_centered)
	
#     if (cost_d_to_a_rotated < lowest_dtw) :
#         lowest_dtw = cost_d_to_a_rotated
#         est_angle = i
# est_angle_a_d = est_angle
# print("A to D comparison", lowest_dtw, "Est Angle is", est_angle_a_d)

################################## B to E -- dont care?
lowest_dtw = 100000000000000000000
est_angle = 0
# for i in range(0,360,step):
	
#     theta = math.radians(i)
#     a_rotated = a*math.cos(theta) + c*math.sin(theta)
#     b_rotated = b
#     c_rotated = c*math.cos(theta) - a*math.sin(theta)

#     a_rotated_centered = a_rotated - (np.ones(len(a_rotated))*np.mean(a_rotated))
#     c_rotated_centered = c_rotated - (np.ones(len(c_rotated))*np.mean(c_rotated))
	
#     matches, cost_d_to_a_rotated, mapping_1, mapping_2, matrix = dtw(a_rotated_centered, d_centered)
	
#     if (cost_d_to_a_rotated < lowest_dtw) :
#         lowest_dtw = cost_d_to_a_rotated
#         est_angle = i
		
# print("B to E comparison", lowest_dtw, "Est Angle is", est_angle)

################################## C to F
lowest_dtw = 100000000000000000000
est_angle = 0
# for i in range(0,360,step):
	
#     theta = math.radians(i)
#     a_rotated = a*math.cos(theta) + c*math.sin(theta)
#     b_rotated = b
#     c_rotated = c*math.cos(theta) - a*math.sin(theta)

#     a_rotated_centered = a_rotated - (np.ones(len(a_rotated))*np.mean(a_rotated))
#     c_rotated_centered = c_rotated - (np.ones(len(c_rotated))*np.mean(c_rotated))
	
#     matches, cost_d_to_a_rotated, mapping_1, mapping_2, matrix = dtw(c_rotated_centered, f_centered)
	
#     if (cost_d_to_a_rotated < lowest_dtw) :
#         lowest_dtw = cost_d_to_a_rotated
#         est_angle = i
# est_angle_c_f = est_angle
#print("C to F comparison", lowest_dtw, "Est Angle is", est_angle_c_f)





theta = math.radians(0)

a_rotated = a*math.cos(theta) + c*math.sin(theta)
b_rotated = b
c_rotated = c*math.cos(theta) - a*math.sin(theta)

a_rotated_centered = a_rotated - (np.ones(len(a_rotated))*np.mean(a_rotated))
c_rotated_centered = c_rotated - (np.ones(len(c_rotated))*np.mean(c_rotated))

fig = plt.figure()
ax = plt.axes()
#ax.plot(np.linspace(0,2,len(a_centered)),a_centered, color = "red")
ax.plot(np.linspace(0,2,len(d_centered)),d_centered, color = "orange")

ax.plot(np.linspace(0,2,len(a_rotated_centered)),a_rotated_centered, color = "green")
#ax.plot(np.linspace(0,2,len(c_rotated_centered)),c_rotated_centered, color = "pink")

#matches, cost_d_to_a_rotated, mapping_1, mapping_2, matrix = dtw(a_rotated_centered, d_centered)

#print("DTW D to rotated A",cost_d_to_a_rotated)

########################################## C to F
theta = math.radians(0)

a_rotated = a*math.cos(theta) + c*math.sin(theta)
b_rotated = b
c_rotated = c*math.cos(theta) - a*math.sin(theta)

a_rotated_centered = a_rotated - (np.ones(len(a_rotated))*np.mean(a_rotated))
c_rotated_centered = c_rotated - (np.ones(len(c_rotated))*np.mean(c_rotated))

fig2 = plt.figure()
ax2 = plt.axes()
#ax.plot(np.linspace(0,2,len(a_centered)),a_centered, color = "red")
ax2.plot(np.linspace(0,2,len(f_centered)),f_centered, color = "blue")

ax2.plot(np.linspace(0,2,len(c_rotated_centered)),c_rotated_centered, color = "black")
#ax.plot(np.linspace(0,2,len(c_rotated_centered)),c_rotated_centered, color = "pink")

#matches, cost_c_to_f_rotated, mapping_1, mapping_2, matrix = dtw(c_rotated_centered, f_centered)

#print("DTW C to rotated F",cost_c_to_f_rotated)

plt.show()