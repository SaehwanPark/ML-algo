# For my own purpose

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *
from sklearn.svm import LinearSVC

# train_x = np.array(
# 	[[0.15860865, 0.69981775],
# 	 [0.9758274,  0.19675629],
# 	 [0.35502879, 0.25307673],
# 	 [0.4027962,  0.07670776],
# 	 [0.49143734, 0.71308283],
# 	 [0.10175679, 0.43235141],
# 	 [0.66959402, 0.08376796],
# 	 [0.14782461, 0.98864606],
# 	 [0.70284335, 0.08237115],
# 	 [0.02813389, 0.31070766],
# 	 [0.63826246, 0.86814922],
# 	 [0.43239001, 0.98778848],
# 	 [0.52742164, 0.0474869 ],
# 	 [0.93240914, 0.96821114],
# 	 [0.5692859,  0.0591384 ],
# 	 [0.51121298, 0.78062027]])

# train_y = np.array([4,2,2,4,4,4,0,1,2,3,3,1,0,2,4,0])

# test_x = np.array(
# 	[[0.0172256,  0.43778008],
# 	 [0.61588832, 0.8327251 ],
# 	 [0.6676327,  0.11108416],
# 	 [0.16378088, 0.82837844],
# 	 [0.9413816,  0.45135086],
# 	 [0.42915915, 0.51836509],
# 	 [0.08504202, 0.36340537],
# 	 [0.70916737, 0.22065988],
# 	 [0.56753214, 0.33488033],
# 	 [0.03794588, 0.39307183],
# 	 [0.33489631, 0.88743855],
# 	 [0.66113892, 0.79184603]])

# #clf = LinearSVC(random_state = 0, C = 0.1, multi_class="crammer_singer")
# clf = LinearSVC(random_state = 0, C = 0.1, multi_class="ovr")
# clf.fit(train_x, train_y)
# pred_test_y = clf.predict(test_x)
# print(pred_test_y)

# fig, ax = plt.subplots()

# scatter = ax.scatter(train_x[:,0],train_x[:,1],c=train_y)
# legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Labels")
# ax.add_artist(legend1)

# theta = clf.coef_
# theta_0 = clf.intercept_

# print(theta)
# print(theta_0)


# x1_bnd = np.linspace(train_x[:,0].min(), train_x[:,0].max(), 100)

# for k in range(theta_0.shape[0]):
# 	a = -(theta[k,0]/theta[k,1])
# 	b = -(theta_0[k]/theta[k,1])
# 	x2_bnd = a*x1_bnd +b
# 	ax.plot(x1_bnd, x2_bnd)
# 	print(f"Now drawing k={k}: y={a}x+{b}")

# plt.ylim(0,1.2)
# plt.show()

# Testing softmax

X = np.array(
	[[ 1., 38., 78., 82., 51., 43., 64., 84.,  9., 22., 28.],
	 [ 1., 63., 90.,  8., 49., 55., 59., 69., 70., 16., 65.],
	 [ 1., 78., 33., 97., 51., 22., 20., 20., 70., 63., 33.],
	 [ 1., 74., 26., 42., 79., 97.,  2.,  9., 17., 73., 40.],
	 [ 1., 59., 95., 90., 78., 32., 51., 29., 62., 10., 96.],
	 [ 1., 64., 94.,  8., 22., 51., 90., 76., 16., 19., 90.],
	 [ 1., 68., 37., 84., 96., 88., 83., 55., 97., 39.,  3.],
	 [ 1., 59., 82., 41., 21., 32., 81., 97., 49., 25., 47.],
	 [ 1.,  4.,  4., 50., 21., 57., 14., 45., 36.,  2., 37.],
	 [ 1., 14., 79., 23., 17., 67., 20., 50., 87., 75., 50.]] 
	)

theta = np.zeros((10,11))

temp_parameter = 1
lambda_factor = 0.0001
#alpha = 0.29999
alpha = 0.3

Y = np.ones(10)

theta1 = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter=1)
theta2 = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter=0.5)

print(theta1)
print(theta2)

exp_theta1 = np.array(
	[[-0.029999, -1.562953, -1.853944, -1.574953, -1.454956, -1.631951
	 , -1.451956, -1.601952, -1.538954, -1.031969, -1.466956],
	 [0.269992, 14.066578, 16.685499, 14.174575, 13.094607, 14.687559
	  ,13.067608, 14.417567, 13.850584, 9.287721, 13.202604],
	 [-0.029999, -1.562953, -1.853944, -1.574953, -1.454956, -1.631951
	 , -1.451956, -1.601952, -1.538954, -1.031969, -1.466956],
	 [-0.029999, -1.562953, -1.853944, -1.574953, -1.454956, -1.631951
	 , -1.451956, -1.601952, -1.538954, -1.031969, -1.466956],
	 [-0.029999, -1.562953, -1.853944, -1.574953, -1.454956, -1.631951
	 , -1.451956, -1.601952, -1.538954, -1.031969, -1.466956],
	 [-0.029999, -1.562953, -1.853944, -1.574953, -1.454956, -1.631951
	 , -1.451956, -1.601952, -1.538954, -1.031969, -1.466956],
	 [-0.029999, -1.562953, -1.853944, -1.574953, -1.454956, -1.631951
	 , -1.451956, -1.601952, -1.538954, -1.031969, -1.466956],
	 [-0.029999, -1.562953, -1.853944, -1.574953, -1.454956, -1.631951
	 , -1.451956, -1.601952, -1.538954, -1.031969, -1.466956],
	 [-0.029999, -1.562953, -1.853944, -1.574953, -1.454956, -1.631951
	 , -1.451956, -1.601952, -1.538954, -1.031969, -1.466956],
	 [-0.029999, -1.562953, -1.853944, -1.574953, -1.454956, -1.631951
	 , -1.451956, -1.601952, -1.538954, -1.031969, -1.466956]]
	)

# exp_theta1: temp=1
# exp_theta2: temp=0.5
exp_theta2 = np.array(
	[[-0.059998, -3.893883, -2.489925, -2.861914, -2.699919, -2.249932
 , -2.615922, -2.981911, -2.789916, -3.965881, -3.935882],
	 [0.539984, 35.044949, 22.409328, 25.757227, 24.299271, 20.249392
	 , 23.543294, 26.837195, 25.109247, 35.692929, 35.422937],
	 [-0.059998, -3.893883, -2.489925, -2.861914, -2.699919, -2.249932
	 , -2.615922, -2.981911, -2.789916, -3.965881, -3.935882],
	 [-0.059998, -3.893883, -2.489925, -2.861914, -2.699919, -2.249932
	 , -2.615922, -2.981911, -2.789916, -3.965881, -3.935882],
	 [-0.059998, -3.893883, -2.489925, -2.861914, -2.699919, -2.249932
	 , -2.615922, -2.981911, -2.789916, -3.965881, -3.935882],
	 [-0.059998, -3.893883, -2.489925, -2.861914, -2.699919, -2.249932
	 , -2.615922, -2.981911, -2.789916, -3.965881, -3.935882],
	 [-0.059998, -3.893883, -2.489925, -2.861914, -2.699919, -2.249932
	 , -2.615922, -2.981911, -2.789916, -3.965881, -3.935882],
	 [-0.059998, -3.893883, -2.489925, -2.861914, -2.699919, -2.249932
	 , -2.615922, -2.981911, -2.789916, -3.965881, -3.935882],
	 [-0.059998, -3.893883, -2.489925, -2.861914, -2.699919, -2.249932
	 , -2.615922, -2.981911, -2.789916, -3.965881, -3.935882],
	 [-0.059998, -3.893883, -2.489925, -2.861914, -2.699919, -2.249932
	 , -2.615922, -2.981911, -2.789916, -3.965881, -3.935882]]
	)

print("\nDiff between exp_theta1 and theta1 =")
print(exp_theta1-theta1)

print("\nDiff between exp_theta1 and theta2 =")
print(exp_theta2-theta2)

