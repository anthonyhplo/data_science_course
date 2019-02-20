# API: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, accuracy_score


if __name__ == "__main__":

	## read train.csv and test.csv into dataframes
	df_train = pd.read_csv("../dataset/mnist/train.csv")
	df_test = pd.read_csv("../dataset/mnist/test.csv")

	X = df_train.iloc[:,1:]
	y = df_train.iloc[:,0]
	k = 5
	train_avg_score = 0
	test_avg_score = 0

	param = {
				"eta" : 1,
				"verbosity" : 1,
				"objective" : "multi:softmax",
				"num_class" : 10,
				"gamma" : 0.1,
				"max_depth" : 4
				"num_boost_round" : 1,
			}

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
	    print("Train: %0.3f, Test: %0.3f"%(train_score, test_score))
	print("Avg score, %0.3f(train)  %0.3f(test)"%(train_avg_score/k, test_avg_score/k))

	with open("m001.model","wb") as f:
		pickle.dump(bst, f)