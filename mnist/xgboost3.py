# API: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import roc_auc_score, accuracy_score


if __name__ == "__main__":

	## read train.csv and test.csv into dataframes
	df_train = pd.read_csv("../dataset/mnist/train.csv")
	df_test = pd.read_csv("../dataset/mnist/test.csv")

	X = df_train.iloc[:,1:]
	y = df_train.iloc[:,0]
	k = 3
	best_score = 0
	bst_param = None
	train_avg_score = 0
	test_avg_score = 0

	param_grid = {
				"eta" : [0.3, 0.5, 1],
				"objective" : "multi:softmax",
				"num_class" : 10,
				"gamma" : [0, 0.2, 0.3],
				"max_depth" : [2 ,3 , 4],
				"num_boost_round" : [5, 10, 15],
			}
	for param in ParameterGrid(param_grid):
		print(param)
		for train_index, test_index in KFold(n_splits=k).split(X):
		    train_X = X.loc[train_index,:]
		    train_y = y.loc[train_index]
		    test_X = X.loc[test_index,:]
		    test_y = y.loc[test_index]
		    dtrain = xgb.DMatrix(train_X, label=train_y)
		    dtest = xgb.DMatrix(test_X)
		    
		    # Training
		    bst = xgb.train(param, dtrain)
		    
		    # Testing
		    train_yhat = bst.predict(dtrain)
		    test_yhat = bst.predict(dtest)

		    train_score = accuracy_score(y.loc[train_index], train_yhat)
		    test_score = accuracy_score(y.loc[test_index], test_yhat)
		    train_avg_score += train_score
		    test_avg_score += test_score
		train_avg_score /= k
		test_avg_score /= k
		print("Avg score, %0.3f(train)  %0.3f(test)"%(train_avg_score, test_avg_score))
		print("======")

		if test_avg_score > best_score:
			best_param = dict(param) 

	## training with whole dataset
    dtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(param, dtrain)

	with open("m003.model","wb") as f:
		pickle.dump(bst, f)