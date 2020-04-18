import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
results = np.zeros(5)

for K in range(1,5):
	for seed in range(5):
		mixture, post = common.init(X, K, seed)
		mixture, post, cost = kmeans.run(X, mixture, post)
		# common.plot(X, mixture, post, f"K={K} seed={seed} cost={cost}")
		results[seed] = cost
	print(results.min())
	results = np.zeros(5)

