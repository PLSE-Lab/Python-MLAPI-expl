# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

cars =pd.read_csv("/kaggle/input/CarPrice.csv")
print(cars.info())


#Understading Data dictionary

#symboling: -2 (least risky) to +3 most risky
cars['symboling'].astype('category').value_counts()

#aspiration: An engine property showing
#whether the oxygen intake is through standard ( atmospheric pressure)
#or through turbocharging

cars['aspiration'].astype('category').value_counts()

#drivewheel: frontwheel, rarewheel or four-wheel drive

cars['drivewheel'].astype('category').value_counts()


#wheelbase: distance between centre of front and rarewheels
sns.distplot(cars['wheelbase'])
plt.show()

#curbweight: weight of car without occupants or baggage
sns.distplot(cars['curbweight'])
plt.show()

#target variable: price of car
sns.distplot(cars['price'])
plt.show()


#data Exploration
#to perform linear regression

#all numeric (float and int ) variables in the data

cars_numeric = cars.select_dtypes(include=['float','int'])
cars_numeric.head()
                                           

#droppping symboling and car_id
cars_numeric = cars_numeric.drop(['symboling', 'car_ID'], axis=1)
cars_numeric.head()
         

#pairwise scatter plot

plt.figure(figsize=(20,10))
sns.pairplot(cars_numeric)
plt.show()



#correlation matrix

cor = cars_numeric.corr()
cor

#plotting correlation on a heatmap

#figure size
plt.figure(figsize=(16,18))

#heatmap
sns.heatmap(cor, cmap='YlGnBu', annot=True)
plt.show()


#Data Cleaning
cars.info()


cars['symboling'] = cars['symboling'].astype('object')
cars.info()


#extracting car name

#method 1: str.split() by space

carnames= cars["CarName"].apply(lambda x: x.split(" ")[0])
carnames[:30]

#method 2: Use regular expression 
import re

#regex: any alphanumeric sequence before a space, may contain a hyphen 

p = re.compile(r'\w+-?\w+')
carnames = cars['CarName'].apply(lambda x: re.findall(p,x)[0])
print(carnames)

#new column car_company
cars['car_company'] = cars['CarName'].apply(lambda x: re.findall(p,x)[0])

print(cars['car_company'] )

#look at all values
cars['car_company'] .astype('category').value_counts()

#replacing misspelled car_company names

#volkswagen
cars.loc[(cars['car_company'] == "vw") |
          (cars['car_company'] == "vw") ,
         'car_company'] = 'volkswagen'

#porsche
cars.loc[cars['car_company'] == "porcshce", "car_company" ] = "porsche"

#toyota
cars.loc[cars['car_company'] == "toyouta", "car_company" ] = "toyota"

#nissan
cars.loc[cars['car_company'] == "Nissam", "car_company" ] = "nissan"

#mazda
cars.loc[cars['car_company'] == "maxda", "car_company" ] = "mazda"


cars['car_company'].astype('category').value_counts()


#drop carname variable
cars = cars.drop('CarName', axis=1)

cars.info()


#Data Preparation

# split into X and Y

X = cars.drop(columns=['price','car_ID'])
y = cars['price']


#creating dummy variables for categorical variables
#subset all categorical variables

cars_categorical = X.select_dtypes(include=['object'])
cars_categorical.head()

#convert into dummies

cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)
cars_dummies.head()

#drop categorical variables
X = X.drop(list(cars_categorical.columns), axis=1)

#concat dummy variables with X
X = pd.concat([X, cars_dummies], axis=1)

#scalling the features

from sklearn.preprocessing import scale

#storing column names in cols, since columnnames are lost after
#scalling (df is converted to a numpy array)

cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns


#split into train and test

from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=100)


#Model Building and Evaluation

lm = LinearRegression()

lm.fit(X_train,y_train)

#print coefficient and intercept

print(lm.coef_)
print(lm.intercept_)

#predict
y_pred = lm.predict(X_test)

#metrics
from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred))


#model building Using RFE (Recursive feature elimination) - You can try PCA and TSNE

from sklearn.feature_selection import RFE

#RFE with 15 features
lm = LinearRegression()
rfe_15 = RFE(lm, 15)

rfe_15.fit(X_train,y_train)

#printing the boolean result
print(rfe_15.support_)
print(rfe_15.ranking_)

y_pred = rfe_15.predict(X_test)

print(r2_score(y_test,y_pred))





















