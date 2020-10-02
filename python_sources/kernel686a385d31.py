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
@script_author: Bindu
@script_name: KNN
@script_decribe: To implement K Nearest Neighbor algorithm using IRIS dataset
@eternal-packages-used: sklearn,pandas,numpy.
"""
import numpy as np #to get numpy package
import pandas as pd #to get panada package

data=pd.read_csv("../input/iris/Iris.csv")
data.head() #import data

X = data.iloc[:, 1:-1].values #Independent  varaible
y = data.iloc[:, 5].values #Target variable

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40) #Assign test_size
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train) #KNN algorithm

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2) #it helps to print accuracy