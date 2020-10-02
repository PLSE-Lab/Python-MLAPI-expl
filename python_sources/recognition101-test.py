import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
# Test for null values
train.loc(pd.isnull(train))
print(train.shape)
X=train.iloc[:,1:]
Y=train.iloc[:,0]
print(Y.unique())
print(Y.head(5))
print(X.shape, Y.shape)
'''
clf=RFC(n_estimators=800, n_jobs=-1)
clf=clf.fit(X,Y)
scores=cross_val_score(clf, X, Y, cv=10)
print(scores, "mean: "+str(scores.mean()))
test_Y=clf.predict(test)
result=pd.DataFrame()
result["ImageId"]=range(len(test_Y))+np.ones(len(test_Y))
result["Label"]=test_Y
#result=pd.DataFrame( {'ImageId': range(len(test_Y)+1, 'Label': test_Y} )
result.to_csv("result.csv", index=False)
'''
'''
clf_extra=ETC(n_estimators=800,n_jobs=-1, bootstrap=True, oob_score=True)
clf_extra=clf_extra.fit(X,Y)
print(clf_extra.oob_score_)
test_Y_extra=clf_extra.predict(test)
result=pd.DataFrame()
result["ImageId"]=range(len(test_Y_extra))+np.ones(len(test_Y_extra), dtype=np.int32)
result["Label"]=test_Y_extra
#result=pd.DataFrame( {'ImageId': range(len(test_Y_extra)+1, 'Label': test_Y_extra} )
result.to_csv("result_extra.csv", index=False)
'''

clf_NN = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(250, ), random_state=1)
clf_NN=clf_NN.fit(X,Y)
test_Y_NN=clf_NN.predict(test)
result=pd.DataFrame()
result["ImageId"]=range(len(test_Y_NN))+np.ones(len(test_Y_NN), dtype=np.int32)
result["Label"]=test_Y_NN
#result=pd.DataFrame( {'ImageId': range(len(test_Y_extra)+1, 'Label': test_Y_extra} )
result.to_csv("result_NN.csv", index=False)
