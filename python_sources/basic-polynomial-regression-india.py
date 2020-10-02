# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle

# Reading the data from the csv file into the datafram 'df'
df = pd.read_csv("../input/GlobalLandTemperaturesByCountry.csv", index_col=0, parse_dates=True)
# Selecting the rows which correspond to India
df = df.loc[df["Country"]=="India"]
# Dropping the rows where average temperature is unavailable
df = df.dropna(subset = ["AverageTemperature"])

# Shuffling the Series Average Temperature
temp = shuffle(df["AverageTemperature"])
# Calculating the number of days since the first day i.e 1796-01-01
days_since = pd.Series((temp.keys().year * 365 + temp.keys().month * 30 + temp.keys().day) - (1796*365 + 1*30 + 1),index=temp.keys(),name="DaysSince")
# Concatenating the series temp and days_sice to make the dataframe 'data'
data = pd.concat([temp, days_since],axis=1)
# Deleting the dataframe 'df'
del df

# Creating the polynomial features of degree 2
poly = PolynomialFeatures(degree=2)
data = pd.DataFrame(poly.fit_transform(data))

# Dividing the data into the training and test set
train = data.iloc[:-int(len(data)*.2)]
test = data.iloc[-int(len(data)*.2):]

# Using the plynominal fetures to train linear regression
regr = linear_model.LinearRegression()
# Selecting the fetures 1,x,x**2
regr.fit(train[[0,2,5]],train[1].to_frame())

# Printing the calculated coefficients, mean squared error on the test set and variance score
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
       % pd.DataFrame.mean((regr.predict(test[[0,2,5]]) - test[1].to_frame()) ** 2))
print('Variance score: %.2f' % regr.score(test[[0,2,5]],test[1].to_frame()))

# Sorting and plotting the data
test = test.sort_values(by=2)
plt.scatter(test[2], test[1],  color='black')
plt.plot(test[2], regr.predict(test[[0,2,5]]), color='blue',
         linewidth=1)

plt.savefig("graph.png")