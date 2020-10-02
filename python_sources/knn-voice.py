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

from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier


## Load data
df = pd.read_csv('../input/voice.csv')

Y = LabelEncoder().fit_transform(df.label)
X = StandardScaler().fit_transform(df.iloc[:, :-2])

trainX, _X, trainY, _Y = train_test_split(X, Y, test_size =0.33)
valX, testX, valY, testY = train_test_split(_X, _Y, test_size =0.5)

clf = KNeighborsClassifier()
clf.fit(trainX, trainY)

scores = []
for i in range(1, 10):
    clf.set_params(n_neighbors=i)
    valScore = clf.score(valX, valY)
    scores.append(valScore)
scores = np.array(scores)
n = scores.argmax() + 1

clf.set_params(n_neighbors=n)
testScore = clf.score(testX, testY)
print(testScore)