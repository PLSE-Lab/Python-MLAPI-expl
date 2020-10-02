"""
    @script_author : Hemanth Kumar M R
    @script_name : Predictions of flower spicies using KNN algorithm
    @script_description : For the given test dataset the algorithm predicts the spicies of flower
    @script_packages_used : sklearn
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
#load data set
iris = datasets.load_iris()
#iris
#Get target and data
target_var=iris.target
train_data=iris.data
X=train_data.tolist()
Y=target_var.tolist()
#split the dataset into train and test
X_train,X_test,Y_train,Y_test=ms.train_test_split(X,Y,test_size=0.2,random_state=0)
sc=StandardScaler()
#Transform the data
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
#define the KNN algorithm
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,Y_train)
classifier.fit(X_train,Y_train)
#predict the variables
Y_pred=classifier.predict(X_test)
print("The predicted values " +str(list(Y_pred)))
print("The test values "+ str(Y_test))