#@author Kunkum - 19BDA71038
#@Date 13/01/2020
#@jupyter - The version of the notebook server is: 6.0.1
#The server is running on this version of Python:
#Python 3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)]

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

import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn import preprocessing  #import label encoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

rdata = pd.read_csv('/kaggle/input/iris/Iris.csv')  #read the csv file
rdata.isnull().sum() #check for null values
cdata = rdata.copy()     #keep the copy of the csv file

label = preprocessing.LabelEncoder()  #understands the labels
cdata['Species'] = label.fit_transform(cdata['Species'])
cdata['Species'].unique()
x = cdata.drop(columns=['Id','Species'])
y = cdata[cdata.columns[5:]]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=46)


#for different k value from 1-29
range_of_k = range(1,30)
#create a list to store k-value and error value for each 'k'
value_k=[]
error_k=[]
#looping for k 
for k in range_of_k:    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    predict = knn.predict(x_test)
    value_k.append(metrics.accuracy_score(y_test,predict))
print("accuracy of each k-value\n",value_k)

