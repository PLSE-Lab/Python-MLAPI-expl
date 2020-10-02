# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

#data = pd.read_csv("../input/train.csv", header = 0)
#print(data.describe())
#print(data.index)
#print(data.columns)

from sklearn import linear_model, datasets
digits = datasets.load_digits()
clf = linear_model.LinearRegression()
x, y = digits.data[:-1], digits.target[:-1]
clf.fit(x, y)
y_pred = clf.predict([digits.data[-1]])
y_true = digits.target[-1]
print(y_pred)
print(y_true)

# Any results you write to the current directory are saved as output.