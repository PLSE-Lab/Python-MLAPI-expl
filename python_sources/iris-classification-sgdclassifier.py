# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

import os
print(os.listdir("../input"))

iris = pd.read_csv("../input/Iris.csv")

# Any results you write to the current directory are saved as output.

#To check data
iris.head(2)
iris.info()

#Drop Id column it does not mean for Data Analytics

iris.drop("Id",axis=1,inplace=True)
iris.info()

#Split data into Independent and dependent Vaiable

X = iris.iloc[:,:2].values
Y = iris.iloc[:,-1].values

#Label Encode the data

y = preprocessing.LabelEncoder().fit_transform(Y)

#Split data into Train and Test

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)
X_train.shape
y_test.shape

#Standerdise Independent Variable
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Explotary Analysis of data

colors = ['red','greenyellow','blue']

for i in range(len(colors)):
    x = X_train[:,0][y_train==i]
    y1 = X_train[:,1][y_train==i]
    plt.scatter(x,y1,c=colors[i])

plt.legend(np.unique(Y))
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
    

#To fit SGDClassifier

clf = SGDClassifier()
clf.fit(X_train,y_train)

#Calculated Coefficent Value
clf.coef_

#Calculated Intercept
clf.intercept_

#Test Accuracy on training set

from sklearn import metrics

y_train_pred = clf.predict(X_train)
metrics.accuracy_score(y_train,y_train_pred)

#Accuracy on Test Set

y_test_pred = clf.predict(X_test)
metrics.accuracy_score(y_test,y_test_pred)

metrics.classification_report(y_test,y_test_pred,target_names = np.unique(Y))

#Print Confusion Matrix
metrics.confusion_matrix(y_test,y_test_pred)

#By Cross-Validation 

from sklearn.model_selection import cross_val_score,KFold
from sklearn.pipeline import Pipeline

clf = Pipeline([('scaler',preprocessing.StandardScaler()),('linear_model',SGDClassifier())])

cv = KFold(5,shuffle = True,random_state=33)

scores = cross_val_score(clf,X,Y,cv=cv)

from scipy.stats import sem
def mean_score(scores):
    return ("Mean Score {0:.3f} (+/-{1:.3f})".format(np.mean(scores),sem(scores)))

#Model has an average Accuracy 0.72    
print(mean_score(scores))    