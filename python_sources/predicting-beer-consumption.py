# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv('../input/Consumo_cerveja.csv', header=0, decimal=',')
df.columns = ['date','temp_avg','temp_min','temp_max',
                        'rain','finalsemana','consumption']
df['consumption'] = df['consumption'].astype('float')
df.dropna(inplace=True)

sns.heatmap(df.corr(), square=True ,annot=True, linewidths=1, linecolor='k')

sns.scatterplot(df['temp_max'],df['consumption'])

sns.scatterplot(df['rain'],df['consumption'])

X = df[['temp_max']]
y = df['consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print("shapes of train (%s) and test (%s)" % (X_train.shape, X_test.shape))

reg = LinearRegression()
reg.fit(X_train, y_train)
reg.score(X_test, y_test)