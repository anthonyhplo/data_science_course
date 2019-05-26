import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Logistic:

	def __init__(self, fit_intercept=True):
		self._coef = None
		self._intercept = None
		self._fit_intercept = fit_intercept
		
	#def sigmoid(z):
		#return 1 / (1 + np.exp(-z))
		
	def predict_proba(self, X):
	
		def sigmoid(z):
			return 1 / (1 + np.exp(-z))
		"""
		Output: model prediction f(x)
		f(x) = wx + b
		
		Arguments:
		X: 2D numpy array
		"""
		z = np.dot(self._coef,X.T)
		h = sigmoid(z)
		return h
		
	def predict(self, X, threshold = 0.5):
		return self.predict_proba(X) >= threshold
	
	def cost_function(self, predictions, y):
		return (-1/y.shape[0])*np.sum((y.T * np.log(predictions)) + ((1 - y.T) * np.log(1 - predictions)))
		
		
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
		
	def fit(self, X, y, learning_rate= 0.01, iters = 100000, log = True):
		"""
		Fit  model coefficients.
		
		Arguments:
		X: 2D numpy array
		Y: 1D numpy array
		"""
		
		self._coef = np.zeros(X.shape[1])
		for i in range(iters):
		#1 Get Prediction:
			predictions = self.predict_proba(X)
		#2 Calculate cost for auditing purposes
			cost = self.cost_function(predictions, y)
		#3 Calculate gradient
			#gradient = (1/X.shape[0])*(np.dot(X.T,  (predictions-y.T).T))
		#4 - Multiply the gradient by our learning rate
			#Calculate new coef
			theta = self._coef - (1/(X.shape[0]))*learning_rate*(np.dot(X.T, (predictions-y.T).T))
			self._coef = theta
			
			if log == True and i % 10000 == 0:
				print("iter: "+str(i) + " cost: "+str(cost) + "")

	def plot_boundary(self, X, y):
		plt.figure(figsize=(10, 6))
		plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
		plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
		plt.legend()
		x1_min, x1_max = X[:,0].min(), X[:,0].max(),
		x2_min, x2_max = X[:,1].min(), X[:,1].max(),
		xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
		grid = np.c_[xx1.ravel(), xx2.ravel()]
		probs = self.predict_proba(grid).reshape(xx1.shape)
		plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='green');



if __name__ == "__main__":
	iris = load_iris()
	X, y = iris.data, iris.target
	X = iris.data[:, :2]
	y = (iris.target != 0) * 1
	model = Logistic()
	model.fit(X, y)
	y_predict = model.predict(X)
	model.plot_boundary(X,y)
	print(confusion_matrix(y, y_predict))