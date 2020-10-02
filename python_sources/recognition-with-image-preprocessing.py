import pandas as pd
import skimage as smg
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology as mph
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import cross_validation

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
X=train.iloc[:,1:]
Y=train.iloc[:,0]


'''
# Formal code blocks to do skeletonization
new_X=np.zeros((28, 28*X.shape[0]), dtype=np.int32)
for i in range(X.shape[0]):
    tmp=X.values[i,:]
    tmp.shape=(28,28)
    new_X[:, (28*i):(28*(i+1))]=tmp

#new_X=mph.binary_dilation(new_X)
#new_X=mph.skeletonize(new_X)
#new_X=np.int32(new_X)

new_X=np.bool_(new_X)
new_X=mph.remove_small_objects(new_X, min_size=64)
new_X=np.int32(new_X)


# Do reverse arrangement
for i in range(X.shape[0]):
    tmp=new_X[:, (28*i):(28*(i+1))].copy()
    tmp.shape=28*28
    X.values[i,:]=tmp
'''
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
clf_extra=ETC(n_estimators=300,n_jobs=-1, bootstrap=True, oob_score=True)
clf_extra=clf_extra.fit(X,Y)
print(clf_extra.oob_score_)
test_Y_extra=clf_extra.predict(test)
print(test_Y_extra)

result=pd.DataFrame()
result["ImageId"]=range(len(test_Y_extra))+np.ones(len(test_Y_extra), dtype=np.int32)
result["Label"]=test_Y_extra
#result=pd.DataFrame( {'ImageId': range(len(test_Y_extra)+1, 'Label': test_Y_extra} )
result.to_csv("result_extra.csv", index=False)
'''
'''
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.5, random_state=1)
clf_SVM=svm.SVC('poly', degree=2)
clf_SVM.fit(float(X_train),Y_train)
test_Y_SVM=clf_SVM.predict(X_test)
print(clf_SVM.score(X_test, Y_test))
'''

# Best Score ever, 97%
clf_NN = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(400, 300, 270, 250, 200, 150), random_state=1)
clf_NN=clf_NN.fit(X,Y)
test_Y_NN=clf_NN.predict(test)
result=pd.DataFrame()
result["ImageId"]=range(len(test_Y_NN))+np.ones(len(test_Y_NN), dtype=np.int32)
result["Label"]=test_Y_NN
#result=pd.DataFrame( {'ImageId': range(len(test_Y_extra)+1, 'Label': test_Y_extra} )
result.to_csv("result_NN.csv", index=False)


