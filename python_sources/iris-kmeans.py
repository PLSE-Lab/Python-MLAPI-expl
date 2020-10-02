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
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split


data = pd.read_csv("../input/Iris.csv")

x = data[[1, 2, 3, 4]]

le = preprocessing.LabelEncoder()
y = le.fit_transform(data[[5]])

n = np.unique(np.array(y)).shape[0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

clf = KMeans(n_clusters=n)
clf.fit(x_train, y_train)  

y_pred = clf.predict(x_test)

print(metrics.accuracy_score(y_pred, y_test))
