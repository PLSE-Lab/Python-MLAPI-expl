# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.utils import shuffle

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Open the file and saving th data into a dataframe df
df = pd.read_csv("../input/GlobalLandTemperaturesByCountry.csv", index_col=0, parse_dates=True)
# Select the rows in which the country is India
df = df.loc[df["Country"]=="India"]
# Drop the rows in which the average temperature is not available
df = df.dropna(subset = ["AverageTemperature"])

# Shuffle the dataset after selecting the column average temperature
temp = shuffle(df["AverageTemperature"])
# Calculating the number of days since day 1 in the dataset i.e 1796-01-01
days_since = pd.Series((temp.keys().year * 365 + temp.keys().month * 30 + temp.keys().day) - (1796*365 + 1*30 + 1),index=temp.keys(),name="DaysSince")
# Concatenating the series' temp and day_since to create a dataframe 'data' with all the necessary data for basic linear regression
data = pd.concat([temp, days_since],axis=1)
# Deleting the initial dataframe as it is useless now
del df

# Dividing the shuffled data into a training set and test set
train = data.iloc[:-int(len(data)*.2)]
test = data.iloc[-int(len(data)*.2):]

# Using the linear model from Sci-kit Learn to train a linear regression
regr = linear_model.LinearRegression()
regr.fit(train["DaysSince"].to_frame(), train["AverageTemperature"].to_frame())

# Printing the coefficients after training the linear regression model on the training data
print('Coefficients: \n', regr.coef_)
# Printing the mean squared error calculated from the data in the test set
print("Mean squared error: %.2f"
      % pd.DataFrame.mean((regr.predict(test["DaysSince"].to_frame()) - test["AverageTemperature"].to_frame()) ** 2))
# Printing the variance score on the test set
print('Variance score: %.2f' % regr.score(test["DaysSince"].to_frame(), test["AverageTemperature"].to_frame()))

# Plotting the test data as points
plt.scatter(test["DaysSince"], test["AverageTemperature"],  color='black')
# Plotting the regression line
plt.plot(test["DaysSince"], regr.predict(test["DaysSince"].to_frame()), color='blue',
         linewidth=1)

plt.savefig("graph.png")