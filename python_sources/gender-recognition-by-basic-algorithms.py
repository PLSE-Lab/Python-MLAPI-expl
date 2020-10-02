# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv('../input/voice.csv')   # reading the file as a csv


# to check for Nan values
df.isnull().sum()


y=[]

# encoding male=1 and female=0 
for i in range(len(df.label)):
  if df.label[i]=='male':
    y.append(1)
  elif df.label[i]=='female':
    y.append(0)
 
df=df.drop('label',axis=1) # drop th ecolumn with labels
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.3,random_state=0)

stdSc=StandardScaler() #preprocessing
X_train=stdSc.fit_transform(X_train)
X_test=stdSc.fit_transform(X_test)




#using logistic regression
C=[0.0001,0.001,0.01,0.1,1,10,100,1000]
for i in enumerate(C):
    lr=LogisticRegression(C=i[1])
    lr.fit(X_train,y_train)
    print(accuracy_score(lr.predict(X_test),y_test))
    
    
    
 #using Multi layer perceptron
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(solver='lbfgs',alpha=11,hidden_layer_sizes=(400,3),random_state=1)
clf.fit(X_train,y_train)
print(accuracy_score(clf.predict(X_test),y_test))



#using decision tree
from sklearn import tree
tr=tree.DecisionTreeClassifier(criterion='gini')
tr2=tree.DecisionTreeClassifier(criterion='entropy')
tr.fit(X_train,y_train)
tr2.fit(X_train,y_train)

print(accuracy_score(tr.predict(X_test),y_test))
print(accuracy_score(tr2.predict(X_test),y_test))




#Using SVM
from sklearn import svm


clf1=svm.SVC(C=2,kernel='rbf')
clf1.fit(X_train,y_train)
print(accuracy_score(clf1.predict(X_test),y_test))



#using naive bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
y_pred=gnb.fit(X_train,y_train).predict(X_test)
print(((y_pred == y_test).sum())/len(y_pred))




#using k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
for i in range (1,10):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    print(accuracy_score(knn.predict(X_test),y_test))
    
    
    
#using k-means clusterring
from sklearn.cluster import KMeans
for i in range(1,10):
    km=KMeans(n_clusters=i)
    km.fit(X_train,y_train)
    print(accuracy_score(km.predict(X_test),y_test))
    
    
#using random forest
from sklearn.ensemble import RandomForestClassifier
for i in range(1,20):
    rfc=RandomForestClassifier(n_estimators=10)
    rfc.fit(X_train,y_train)
    print(accuracy_score(rfc.predict(X_test),y_test))
#subject to changes in the accuracy as it is random:p


#using gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

  
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=0)
model.fit(X_train,y_train)
print(accuracy_score(model.predict(X_test),y_test))


