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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 09:19:32 2017

@author: moriano
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
raw = pd.read_csv("../input/Iris.csv")

y = raw['Species']
del raw['Species']
del raw['Id']

X = raw
model = LogisticRegression()
model.fit(X, y)

print("-----------------")
print("Linear")
print(accuracy_score(y, model.predict(X)))

X = normalize(X)
model = SVC(kernel="rbf", C=10)
model.fit(X, y)
print("-----------------")
print("SVM")
print(accuracy_score(y, model.predict(X)))

model = MLPClassifier(verbose = False, max_iter=1000)
model.fit(X, y)
print("-----------------")
print("Neural network")
print(accuracy_score(y, model.predict(X)))