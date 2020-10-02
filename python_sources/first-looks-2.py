# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
index = pd.read_csv("../input/index.csv")
index['Real GDP'] = index['Real GDP (Percent Change)']
del index['Real GDP (Percent Change)']

print(list(index))

spec1 = index['Federal Funds Target Rate'].notnull()
spec2 = index['Unemployment Rate'].notnull()
spec3 = index['Real GDP'].notnull()
index = index[spec1 &spec2 & spec3]
print(index['Real GDP'].head(50))
regr = lm.LinearRegression();
y1 = index['Real GDP']
y2 = [None]*len(index['Real GDP'])

count = 1
while count < len(index['Real GDP']):
    y2[count] = (index['Real GDP'].iloc[count] - index['Real GDP'].iloc[count - 1])/index['Real GDP'].iloc[count - 1] 
    count += 1
y2 = pd.DataFrame(y2)

X = index
del X['Real GDP']
del X['Federal Funds Target Rate']
del X['Day']
del X['Year']
del X['Month']
del X['Federal Funds Upper Target']
del X['Federal Funds Lower Target']
del X['Inflation Rate'] #GDP is inflation-adjusted
print("X: ", X.head())
regr.fit(y1.values.reshape(-1, 1), X)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The coefficients
regr.fit(y2.iloc[1:len(y2) - 1].values.reshape(-1, 1), X.iloc[1:len(y2) - 1])
print('Coefficients: \n', regr.coef_)
