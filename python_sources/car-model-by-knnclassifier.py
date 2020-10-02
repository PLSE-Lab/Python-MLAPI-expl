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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pickle
data = pd.read_csv("../input/car.csv",sep=',')
print(data.head())

#precprocesssing to convert non numeric data numeric retuns numpy array
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
predict = "class"

X = list(zip(buying, maint, lug_boot, door, persons, safety, door))
y = list(cls)

X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
best = 0
os.mkdir("../model")
for k in range(1, len(X_train)):
    X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test,y_test)
    if acc > best:
        best = acc
        print("Accuracy : ", acc)
        with open("../model/Car-Model.pickle", "wb") as f:
            pickle.dump(knn, f)


pickle_in = open("../model/Car-Model.pickle", "rb")
knn = pickle.load(pickle_in)