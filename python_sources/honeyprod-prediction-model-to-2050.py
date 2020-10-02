# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../input/honeyproduction.csv")
print(df.head())
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
X = df['year']
X = X.values.reshape(-1,1)
y = df['totalprod']
y = y.values.reshape(-1,1)
regr = LinearRegression()
regr.fit(X,y)
print(regr.coef_[0])
y_predict = regr.predict(X)

X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)

future_predict = regr.predict(X_future)

plt.scatter(X, y)
plt.plot(X, y_predict)
plt.plot(X_future, future_predict)
plt.show()