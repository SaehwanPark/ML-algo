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

train_x = np.array(
	[[0.15860865, 0.69981775],
	 [0.9758274,  0.19675629],
	 [0.35502879, 0.25307673],
	 [0.4027962,  0.07670776],
	 [0.49143734, 0.71308283],
	 [0.10175679, 0.43235141],
	 [0.66959402, 0.08376796],
	 [0.14782461, 0.98864606],
	 [0.70284335, 0.08237115],
	 [0.02813389, 0.31070766],
	 [0.63826246, 0.86814922],
	 [0.43239001, 0.98778848],
	 [0.52742164, 0.0474869 ],
	 [0.93240914, 0.96821114],
	 [0.5692859,  0.0591384 ],
	 [0.51121298, 0.78062027]])

train_y = np.array([4,2,2,4,4,4,0,1,2,3,3,1,0,2,4,0])

test_x = np.array(
	[[0.0172256,  0.43778008],
	 [0.61588832, 0.8327251 ],
	 [0.6676327,  0.11108416],
	 [0.16378088, 0.82837844],
	 [0.9413816,  0.45135086],
	 [0.42915915, 0.51836509],
	 [0.08504202, 0.36340537],
	 [0.70916737, 0.22065988],
	 [0.56753214, 0.33488033],
	 [0.03794588, 0.39307183],
	 [0.33489631, 0.88743855],
	 [0.66113892, 0.79184603]])

#clf = LinearSVC(random_state = 0, C = 0.1, multi_class="crammer_singer")
clf = LinearSVC(random_state = 0, C = 0.1, multi_class="ovr")
clf.fit(train_x, train_y)
pred_test_y = clf.predict(test_x)
print(pred_test_y)

fig, ax = plt.subplots()

scatter = ax.scatter(train_x[:,0],train_x[:,1],c=train_y)
legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Labels")
ax.add_artist(legend1)

theta = clf.coef_
theta_0 = clf.intercept_

print(theta)
print(theta_0)


x1_bnd = np.linspace(train_x[:,0].min(), train_x[:,0].max(), 100)

for k in range(theta_0.shape[0]):
	a = -(theta[k,0]/theta[k,1])
	b = -(theta_0[k]/theta[k,1])
	x2_bnd = a*x1_bnd +b
	ax.plot(x1_bnd, x2_bnd)
	print(f"Now drawing k={k}: y={a}x+{b}")

plt.ylim(0,1.2)
plt.show()