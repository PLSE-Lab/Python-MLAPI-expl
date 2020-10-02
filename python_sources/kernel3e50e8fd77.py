# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
path = r"../input/diabetes/diabetes.csv"
data = pd.read_csv(path)
array = data.values
X = array[:,[0,1,2,4,5,6,7]]
Y = array[:,8]
print(data.info())
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=0)
model = LogisticRegression(C = 4,penalty= 'l2')
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(accuracy_score(y_test, y_predict))
# Any results you write to the current directory are saved as output.
#This Simple Has Achieved 83% Accuracy