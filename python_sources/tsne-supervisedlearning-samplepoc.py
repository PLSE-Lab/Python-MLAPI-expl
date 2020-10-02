# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import os
import random
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate


train = pd.read_csv("../input/train.csv")
test =  pd.read_csv("../input/test.csv")

Response_ = 'label'

Y = train[[Response_]]
X = train.drop(Response_,1)
X = X.apply(lambda x: x/float(255))


# Sampling for POC checking

unique_labels = np.unique(Y.values)
dsindex=[]
for j in unique_labels:
    indexlist = Y[Y == j].index
    uniform = np.random.choice(range(0,len(indexlist)),1500,False)
    [dsindex.append(k) for k in uniform]
    
X,Y = X.loc[dsindex],Y.loc[dsindex]
df = pd.concat([X,Y],axis=1).sample(frac=1)

X,Y = df.drop('label',1),df['label']
tsne_mine =  TSNE(n_components=2,learning_rate=100.0)
tsne_train = tsne_mine.fit(X)
X = tsne_train.embedding_
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


GBM_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=0.8,max_depth=5).fit(X_train, y_train)
gbm_prediction = GBM_model.predict(X_test)


accuracygbm = GBM_model.score(X_test, y_test)
# creating a confusion matrix
cmgbm = confusion_matrix(y_test, gbm_prediction)
precsiongbm, recallgbm, fbeta_scoregbm, _gbm = precision_recall_fscore_support(y_test, gbm_prediction)

print("Accuracy Compare ::::: GBM = ",accuracygbm)

print("***** Confusion Matrix GBM ******************")
print(cmgbm)


cv_results = cross_validate(GBM_model, X, Y, return_train_score=True)

print( " Cross Validation Results " )
print(cv_results)