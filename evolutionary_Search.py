#Evolutionary Search
#module by Raghav Atal
#MSc Analytics

import numpy as np

sample_target = np.array([0.3,-0.4,0.1])

def evolutionary_search(population_size,sigma,alpha,iterations):

	weights = np.random.randn(3) #initial random weights
	for i in range(iterations):
		eps = np.random.randn(population_size,3)
		F_returns = np.zeros(population_size)
		for j in range(population_size):
			F = weights + sigma * eps[j]
			F_returns[j] = -np.sum((F - sample_target)**2)
			print(F_returns)
		#print(F_return)
		F_normalized = (F_returns- np.mean(F_returns))/np.std(F_returns)
		weights = weights + (alpha/(population_size*sigma)) * np.dot(eps.T,F_normalized)

	print(weights)


population_size = 100
sigma = 0.1
alpha = 0.01
iterations = 100

evolutionary_search(population_size,sigma,alpha,iterations)
