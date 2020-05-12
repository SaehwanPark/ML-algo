import numpy as np

def value(r, q):
	dists = np.array([[0, 1, 1, 2], [1, 0, 2, 1], [1,2,0,1], [2,1,1,0]])
	dist = dists[r,q]
	
	reward=0

	for t in range(dist+1):
		if t==dist:
			reward += 0.5**t * 1
		else:
			reward += (0.5)**t * (-0.01)

	return reward

exp_val = 0

for r in range(4):
	for q in range(4):
		print(value(r,q))
		exp_val += (1/16) * value(r,q)

print("E[V] = ", exp_val)