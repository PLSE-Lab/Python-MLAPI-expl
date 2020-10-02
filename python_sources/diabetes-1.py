# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print('Available datasets')
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

d = pd.read_csv('../input/diabetes.csv')

# Logistic Regression

# fit a logistic regression model to the data

model = LogisticRegression()
model.fit(d.drop('Outcome', axis=1), d.Outcome)
print(model)

# make predictions
expected = d.Outcome
predicted = model.predict(d.drop('Outcome', axis=1))

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))