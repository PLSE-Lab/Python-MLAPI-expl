from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pylab as plt
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn import svm


# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("../input/train.csv")
Y = dataset[[0]].values.ravel()
X = dataset.iloc[:,25:].values
datatest = pd.read_csv("../input/test.csv")
test = datatest.iloc[:,24:].values

dev_cutoff = len(Y) * 0.9
X_train = X[:dev_cutoff]
Y_train = Y[:dev_cutoff]
X_test = X[dev_cutoff:]
Y_test = Y[dev_cutoff:]


clf = svm.SVC(kernel='poly')
clf.fit(X_train, Y_train) 
pred = clf.predict(X_test)
print(clf.score(X_test,Y_test))

np.savetxt('Predicciones.csv', np.c_[Y_test,pred], delimiter=',', header = 'Label,PrediccionSVC', comments = '', fmt='%d')  

clf.fit(X,Y) 
predtest = clf.predict(test) 

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
# gbm = xgb.XGBClassifier(max_features=24,max_depth=50, n_estimators=100, learning_rate=0.01).fit(x, y)
#pred = gbm.predict(test)

#rf = RandomForestClassifier(n_jobs=-1,n_estimators=2000 ,max_features=25,criterion='entropy',max_depth=500, min_samples_leaf=2,min_samples_split=2,oob_score=True,warm_start=False )
#rf.fit(train_x, train_y)
#pred = rf.predict(test_x)
#print(rf.oob_score_)
# print(clf.score(test_x,test_y))
# # print (metrics.classification_report(test_y,pred))
# # # print (metrics.confusion_matrix(test_y,pred))

np.savetxt('submission_SCV.csv', np.c_[range(1,len(test)+1),predtest], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
