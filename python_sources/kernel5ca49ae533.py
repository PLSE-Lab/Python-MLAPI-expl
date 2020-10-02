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
"""
    @script-author: Joy Preetha
    @script-name: K_Nearest_Neighbour
    @script-description: Implementation of KNN algorithm Iris dataset 
    @external-packages-used: sklearn
"""
#Loading Iris Dataset
from sklearn import datasets
iris=datasets.load_iris() 
x=(iris.data).tolist()
y=(iris.target).tolist()

#Splitting train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3) 

#Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#Training and prediction
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

#Accuracy
mode=[i for i in range(len(y_pred)) if y_pred[i]==y_test[i]]  
accuracy=(len(mode)/len(y_test))*100
print('0: setosa, 1: versicolor, 2: verginica')
print('predicted_values : ',y_pred)
print('accuracy : ',accuracy)