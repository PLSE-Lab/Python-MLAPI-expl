# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import statistics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

dataset = pd.read_csv('../input/BlackFriday.csv')
xdf = dataset.iloc[:,:-1]
ydf = dataset.iloc[:,-1]

xdf['Product_Category_1'].fillna(0,inplace = True)
xdf['Product_Category_2'].fillna(0,inplace = True)
xdf['Product_Category_3'].fillna(0,inplace = True)

#check if any missing values are there
xdf.isnull().sum()

#delete unnecesary columns like userid and prodid
xdf = xdf.drop(xdf.columns[[0, 1]], axis=1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Encoding Age
labelEncoderAge = LabelEncoder()
xdf.iloc[:,1] = labelEncoderAge.fit_transform(xdf.iloc[:,1])

#Encoding gender
labelEncoderGender = LabelEncoder()
xdf.iloc[:,0] = labelEncoderGender.fit_transform(xdf.iloc[:,0])

#Encoding City category
labelEncoderCity = LabelEncoder()
xdf.iloc[:,3] = labelEncoderCity.fit_transform(xdf.iloc[:,3])

#Encoding current city yr
labelEncoderCityyr = LabelEncoder()
xdf.iloc[:,4] = labelEncoderCityyr.fit_transform(xdf.iloc[:,4])

xdf1 = xdf.iloc[:,:].values
ydf1 = ydf.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xdf1, ydf1, test_size = 0.25, random_state = 0)


#accuracy = []
from sklearn.ensemble import RandomForestRegressor
"""for x in range(100, 1000, 100):
    regressor = RandomForestRegressor(n_estimators = x, random_state = 0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    error = statistics.mean(100 * abs(y_test-y_pred)/y_test)
    accuracy.append(100-error)
    #print("Accuracy: "+str(accuracy.mean())+"%")"""
    
regressor = RandomForestRegressor(n_estimators = 900, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
error = statistics.mean(100 * abs(y_test-y_pred)/y_test)
print(100-error)

"""plt.scatter(range(100, 1000, 100),accuracy,color = 'blue')
plt.title('Random Forest Regression Accuracies')
plt.xlabel('No. of Trees')
plt.ylabel('Accuracy')
plt.show()"""




# Any results you write to the current directory are saved as output.