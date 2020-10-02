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
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from statistics import mode

# Importing the dataset
training_dataset = pd.read_csv("/kaggle/input/titanic/train.csv")
test_dataset = pd.read_csv("/kaggle/input/titanic/test.csv")

X = training_dataset.iloc[:,4:8].values
y = training_dataset.iloc[:,1].values
ids = test_dataset.iloc[:,0].values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# Label Encoding 
label_encoder = LabelEncoder()
X[:,0] = label_encoder.fit_transform(X[:,0])
# Replace missing values with mean values
imputer = SimpleImputer(strategy='mean')
imputer = imputer.fit(X[:,1:2])
X[:,1:2] = imputer.transform(X[:,1:2])
# Feature Scaling
sc = StandardScaler()
sc.fit_transform(X)

classifier = LogisticRegression(random_state = 0).fit(X,y)
classifier.score(X,y)

y_test = test_dataset.iloc[:,3:7].values
y_test[:,0] = label_encoder.transform(y_test[:,0])
imputer = imputer.fit(y_test[:,1:2])
y_test[:,1:2] = imputer.transform(y_test[:,1:2])
sc.transform(y_test)
prediction = classifier.predict(y_test)
result = np.vstack((ids, prediction))
result = result.T
result1 = np.around(result, decimals=2)

a = np.array([["PassengerId", "Survived"]])
# a = ["PassengerId", "Survived"]
# a = np.array(a)
# a = a.T
result2 = np.vstack((a, result1))
# print(result1)

# print()

matrix = result2
matrix = pd.DataFrame(data=matrix)
matrix.to_csv("result2.csv")
# print(matrix[0])
# result1.savetxt('csv_to_submit.csv', index = False)

# np.savetxt("result3.sv", result2,delimiter=",")