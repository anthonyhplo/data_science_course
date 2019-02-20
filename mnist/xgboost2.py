## Model comparision http://yann.lecun.com/exdb/mnist/
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

	kfold = KFold(n_splits=3)
	train_avg_score = 0
	test_avg_score = 0
	k = 3


	xgb_model = xgb.XGBClassifier(objective="multi:softmax", n_estimators=10, n_jobs=4)
	param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'muti'}

	for train_index, test_index in kfold.split(X):
	    xgb_model.fit(X.loc[train_index], y.loc[train_index])
	    train_yhat = xgb_model.predict(X.loc[train_index])
	    test_yhat = xgb_model.predict(X.loc[test_index])
	    train_score = accuracy_score(y.loc[train_index], train_yhat)
	    test_score = accuracy_score(y.loc[test_index], test_yhat)
	    print("Train: %0.3f, Test: %0.3f"%(train_score, test_score))

	with open("m002.model","wb") as f:
		pickle.dump(bst, f)