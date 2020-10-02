# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.



df = pd.read_csv("../input/vgsales.csv")
#using labelEncoder convert categorical data into numerical data
number = LabelEncoder()
df['Platform'] = number.fit_transform(df['Platform'].astype('str'))
df['Genre'] = number.fit_transform(df['Genre'].astype('str'))
df['Publisher'] = number.fit_transform(df['Publisher'].astype('str'))

dff = df.drop(['Rank','Name', 'Year'], axis =1)

df3 = dff.drop(['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales'], axis =1)

#columns = ["Platform", "Genre", "Publisher"]
#columns = ["Platform", "Genre", "Publisher", "NA_Sales", "EU_Sales"]
columns = ["Platform", "Genre", "Publisher", "NA_Sales", "EU_Sales"]
  
labels = df3["Global_Sales"].values
features = dff[list(columns)].values

regr = linear_model.LinearRegression()

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

scaler = StandardScaler()
#scaler = preprocessing.MinMaxScaler()

# Fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)

regr.fit(X_train, y_train)

Accuracy = regr.score(X_train, y_train)
print ("Accuracy in the training data: ", Accuracy*100, "%")

accuracy = regr.score(X_test, y_test)
print ("Accuracy in the test data", accuracy*100, "%")