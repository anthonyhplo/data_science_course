import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

class LinearReg():
	
	'''
	*******************************************************
	*Remember to Normalize features before fit and predict*
	*******************************************************
	'''

	def __init__(self, fit_intercept=True):
		self._coef = None
		self._intercept = None
		self._fit_intercept = fit_intercept
		
		
	def predict(self, X):
		"""
		Output: model prediction f(x)
		f(x) = wx + b
		
		Arguments:
		X: 2D numpy array
		"""

		return np.dot(X, self._coef)
		
	def cost_function(self, predictions, y):
		"""
		Look for the MSE
		Arguments:
		predictions:
		y:
		"""
		sq_error = (predictions - y)**2
		return (1.0/(2*(y.shape[0]))) * sq_error.sum()
		
	def normalize(self, X):
		"""
		normalize features
		"""

		for i in X.T:
			fmean = np.mean(i)
			frange = np.amax(i) - np.amin(i)

			#Vector Subtraction
			i -= fmean

			#Vector Division
			X /= frange

		return X
		
	def add_bias(self, X):
		"""
		add bias to features
		"""
		return np.c_[np.ones(X.shape[0]), X]
		
	def fit(self, X, y, learning_rate = 0.01, iters = 50000, log = True):
		"""
		Fit  model coefficients.
		
		Arguments:
		X: 2D numpy array
		Y: 1D numpy array
		"""
			
		#Initialize the weights
		self._coef = np.zeros(X.shape[1])
		
		for i in range(iters):
			#1 Get Prediction:
			predictions = self.predict(X)
			#2 Calculate cost for auditing purposes
			cost = self.cost_function(predictions, y)
			
			#3 Calculate error/loss
			error = predictions - y
			
			#Calculate new coef
			theta = self._coef - (1/(X.shape[0]))*learning_rate*(np.dot(X.T, error))
			
			self._coef = theta
			
			if log == True and i % 10000 == 0:
				print("iter: "+str(i) + " cost: "+str(cost) + "")

	
if __name__ == "__main__":
	boston = load_boston()
	X = boston.data
	y = boston.target
	model  = LinearReg()
	features = model.add_bias(model.normalize(X))
	model.fit(features, y)
	predict_result = model.predict(features)
	plt.scatter(y, predict_result)
	plt.xlabel("Prices: $Y_i$")
	plt.ylabel("Predicted prices: $\hat{Y}_i$")
	plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")