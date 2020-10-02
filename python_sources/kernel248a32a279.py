# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Read data
data = pd.read_csv("/kaggle/input/success-of-bank-telemarketing-data/Alpha_bank.csv")

# Data encoder
enc = OrdinalEncoder()

# X & y
X = enc.fit_transform(data.drop('Subscribed', axis=1))
y = enc.fit_transform(data[['Subscribed']])

#Split data Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Logistic Regression
clf = LogisticRegression(solver='lbfgs', C=100, max_iter=250)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))