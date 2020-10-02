# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import  matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

dataset=pd.read_csv('../input/Iris.csv')
#,["Sepallength","Seapalwidth","Petallength","Petalwidth","Species"])
#print (dataset.head())
#print (dataset.describe())

dataset.plot(kind='box', subplots=True)
#, layout=(2,2), sharex=False, sharey=False)
#plt.show()

array=dataset.values
X=array[:,1:5]
Y=array[:,5]
seed=4


X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=0.20,random_state=seed)
print (X_train[0:10])
print (Y_train[0:10])

clf=SVC()
clf.fit(X_train,Y_train)
predictions=clf.predict(X_validation)
print ("normal predictions for SVM ", accuracy_score(Y_validation,predictions))

models=[]
models.append(('SVM',SVC()))
models.append(('KNN',KNeighborsClassifier(5) ))
models.append(('DecisionTree',DecisionTreeClassifier()))
models.append(('Naive_Bayes',GaussianNB()))

for name,model in models:
    kfold=model_selection.KFold(n_splits=12,random_state=seed)
    cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    msg="With KFold %s : %f , %f" %( model, cv_results.mean(),cv_results.std())
    print (msg)

print ("========================================================")

X=normalize(X)
X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=0.20,random_state=seed)

#print (X[0:10])
clf=SVC()
clf.fit(X_train,Y_train)
predictions=clf.predict(X_validation)
print (' predictions for SVM after normalize ', accuracy_score(Y_validation,predictions))
# Any results you write to the current directory are saved as output.