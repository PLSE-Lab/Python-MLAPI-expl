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


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt 

from sklearn import datasets
iris = datasets.load_iris()

#splittingthe dataset 
from sklearn.model_selection import train_test_split
X = iris.data  
y = iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Determining the best k value using cross validation 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
k_range = range(1, 30)
k_score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    k_score.append(scores.mean())
# plot to see clearly and determine best k value to fit the model
plt.plot(k_range, k_score)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

#Create KNN model
knn = KNeighborsClassifier(n_neighbors=4,metric='euclidean')
#Train the model using the training sets
knn.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Finding the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) 
print("Confusion Matrix :",cm) 
#Getting the classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print("Classification Report  :",cr)