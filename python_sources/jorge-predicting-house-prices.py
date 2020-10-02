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

#First we will want to import our dataset 
dataset = pd.read_csv("../input/kc_house_data.csv")
#Our dependent variable is the price so lets get that into a array 
housePrices = dataset.iloc[:,2].values
#The things that effect the house prices are everything else (except id and date), lets get those into an array
housePriceFactors = dataset.iloc[:,3:].values

#Lets split this data up into tests and training sets
from sklearn.model_selection import train_test_split
housePriceFactors_training, housePriceFactors_testing, housePrices_training, housePrices_testing = train_test_split(housePriceFactors, housePrices, test_size = .2)

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(housePriceFactors_training, housePrices_training)

#Predicting the test set results
housePrices_prediction = regressor.predict(housePriceFactors_testing)