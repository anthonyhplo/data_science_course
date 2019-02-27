import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
	df_train = pd.read_csv("../dataset/mnist/train.csv")

	X = df_train.iloc[:,1:].values
	y = df_train.iloc[:,0].values


	covX = np.cov(X.T)
	lambdas, Q = np.linalg.eigh(covX)

	idx = np.argsort(-lambdas)
	lambdas = lambdas[idx] # sort in proper order
	lambdas = np.maximum(lambdas, 0)
	Q = Q[:,idx]

	X_transformed = X.dot(Q)

	plt.scatter(X_transformed[:,0], X_transformed[:,1], c=y, alpha=0.5)
	plt.show()