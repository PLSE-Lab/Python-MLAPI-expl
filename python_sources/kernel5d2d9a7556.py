# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

X= data.drop(columns="target")
y= data["target"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

clf = RandomForestClassifier(n_estimators=100, max_depth=15)
clf.fit(X_train,y_train)

print("model score: %.3f" % clf.score(X_test, y_test))