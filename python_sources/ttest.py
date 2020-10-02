# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/Iris.csv")
# print(type(data))

data = data.iloc[np.random.permutation(len(data))]

Y = data['Species']
# print(Y)

X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
# print(X)

vY = Y[0:15]
vX = X[0:15]
X = X[15:]
Y = Y[15:]

clf = svm.SVC()
clf.fit(X, Y)  
prediction = clf.predict(vX);
accuracy = prediction == vY;
print(accuracy)
print("Accuracy: ", np.sum(accuracy), "/", len(vX));





