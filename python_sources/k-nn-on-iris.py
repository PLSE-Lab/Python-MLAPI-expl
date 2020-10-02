# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
@script author:Arpitha Shibu
@script decription:Applying k-NN on iris dataset
@script start date:11.01.20
@script last updated:13.01.20
"""
#importing the necessary packages      
import pandas as pd
import sklearn.preprocessing as pre
import sklearn.model_selection as ms
import numpy as np
from sklearn import metrics

#Reading the csv file
iris=pd.read_csv("/kaggle/input/iris/Iris.csv")
#checking the existence of null values
iris.isnull().sum()
#to get the attribute info
iris.info()
#to find out the categorical data
col_obj=iris.select_dtypes(include='object').columns
#convertion of categorical data
for i in col_obj:
    iris[i]=pre.LabelEncoder().fit_transform(iris[i])
print(iris)
#storing the depended variable
y_dep=iris['Species']
#storing the indept variable by dropping the dependent variable
x_ind=iris.drop(['Species','Id'],axis=1)
#splitting into train and test dataset
x_train,x_test,y_train,y_test=ms.train_test_split(x_ind,y_dep,test_size=0.3,random_state=2233)
a=[]
b=[]
#using sklearn applying k-NN
from sklearn.neighbors import KNeighborsClassifier
for j in range(1,46):
    k=KNeighborsClassifier(n_neighbors=j,metric='euclidean')
    io=k.fit(x_train,y_train)
    pred=k.predict(x_test)
    a.append(np.mean(pred !=y_test))
    b.append(metrics.accuracy_score(y_test,pred))
print(b)
