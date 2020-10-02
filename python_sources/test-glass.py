# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/glass.csv").sample(frac=1)
div = int(len(data) * 0.9)
train_data, test_data = data[:div], data[div:]
train_data_X, train_data_y = np.asarray(train_data.iloc[:, :-1]), np.asarray(train_data.iloc[:, -1])
test_data_X, test_data_y = np.asarray(test_data.iloc[:, :-1]), np.asarray(test_data.iloc[:, -1])

def find_diff(x, y):
    return sum(1 if x[i] - y[i] != 0 else 0 for i in range(len(x)))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train_data_X, train_data_y)
y = knn.predict(test_data_X)
print(y)
print(test_data_y)
print(find_diff(y, test_data_y), find_diff(y, test_data_y) / len(y))

