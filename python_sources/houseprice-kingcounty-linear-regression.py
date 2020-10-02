#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from math import sqrt, log, exp

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data= pd.read_csv("../input/kc_house_data.csv")


# In[ ]:


plt.boxplot(data["price"], 0, '')
plt.show()

data["price"].describe()


# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


print("Correlation Matrix of most corellated features")
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.loc[:,["sqft_living", "grade", "bathrooms", "sqft_above", "sqft_living15", "view","waterfront","bedrooms"]].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


pairplot = sns.pairplot(data.loc[:,["sqft_living", "grade", "bathrooms", "view","waterfront","bedrooms","price"]])
plt.show()


# In[ ]:


print("Mean value of features that seem useful after plots")
xy=data[["price", "sqft_living"]].groupby("sqft_living", as_index=False).mean()
sns.scatterplot(x=xy["sqft_living"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "grade"]].groupby("grade", as_index=False).mean()
sns.scatterplot(x=xy["grade"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "bathrooms"]].groupby("bathrooms", as_index=False).mean()
sns.scatterplot(x=xy["bathrooms"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "view"]].groupby("view", as_index=False).mean()
sns.scatterplot(x=xy["view"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "waterfront"]].groupby("waterfront", as_index=False).mean()
sns.scatterplot(x=xy["waterfront"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "bedrooms"]].groupby("bedrooms", as_index=False).mean()
sns.scatterplot(x=xy["bedrooms"].drop([0,12]), y=xy["price"], data=data)
plt.show()

# other useful features

xy=data[["price", "yr_built"]].groupby("yr_built", as_index=False).mean()
sns.scatterplot(x=xy["yr_built"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "yr_renovated"]].groupby("yr_renovated", as_index=False).mean()
sns.scatterplot(x=xy["yr_renovated"].drop(0), y=xy["price"], data=data)
plt.show()

xy=data[["price", "zipcode"]].groupby("zipcode", as_index=False).mean()
sns.scatterplot(x=xy["zipcode"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "lat"]].groupby("lat", as_index=False).mean()
sns.scatterplot(x=xy["lat"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "long"]].groupby("long", as_index=False).mean()
sns.scatterplot(x=xy["long"], y=xy["price"], data=data)
plt.show()


# In[ ]:


print("All features that seemed useful after first overview")
xy=data[["price", "sqft_living"]].groupby("sqft_living", as_index=False).mean()
sns.scatterplot(x=xy["sqft_living"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "grade"]].groupby("grade", as_index=False).mean()
sns.scatterplot(x=xy["grade"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "bathrooms"]].groupby("bathrooms", as_index=False).mean()
sns.scatterplot(x=xy["bathrooms"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "view"]].groupby("view", as_index=False).mean()
sns.scatterplot(x=xy["view"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "waterfront"]].groupby("waterfront", as_index=False).mean()
sns.scatterplot(x=xy["waterfront"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "bedrooms"]].groupby("bedrooms", as_index=False).mean()
sns.scatterplot(x=xy["bedrooms"].drop([0,12]), y=xy["price"], data=data)
plt.show()

xy=data[["price", "lat"]].groupby("lat", as_index=False).mean()
sns.scatterplot(x=xy["lat"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "yr_built"]].groupby("yr_built", as_index=False).mean()
sns.scatterplot(x=xy["yr_built"].apply(lambda x: 2016-x), y=xy["price"], data=data)
plt.show()

xy=data[["price", "yr_renovated"]].groupby("yr_renovated", as_index=False).mean()
sns.scatterplot(x=xy["yr_renovated"].drop(0).apply(lambda x: 2016-x), y=xy["price"], data=data)
plt.show()


# In[ ]:


print("Took log of price to make exp-functions linear")
xy=data[["price", "sqft_living"]].groupby("sqft_living", as_index=False).mean()
xy["price"]=xy["price"].apply(lambda x:log(x))
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["sqft_living"], y=xy["log(price)"], data=data)
plt.show()

xy=data[["price", "grade"]].groupby("grade", as_index=False).mean()
xy["price"]=xy["price"].apply(lambda x:log(x))
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["grade"], y=xy["log(price)"], data=data)
plt.show()

xy=data[["price", "bathrooms"]].groupby("bathrooms", as_index=False).mean()
xy["price"]=xy["price"].apply(lambda x:log(x))
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["bathrooms"], y=xy["log(price)"], data=data)
plt.show()

xy=data[["price", "view"]].groupby("view", as_index=False).mean()
xy["price"]=xy["price"].apply(lambda x:log(x))
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["view"], y=xy["log(price)"], data=data)
plt.show()

xy=data[["price", "waterfront"]].groupby("waterfront", as_index=False).mean()
xy["price"]=xy["price"].apply(lambda x:log(x))
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["waterfront"], y=xy["log(price)"], data=data)
plt.show()

xy=data[["price", "bedrooms"]].groupby("bedrooms", as_index=False).mean()
xy["price"]=xy["price"].apply(lambda x:log(x))
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["bedrooms"].drop([0,12]), y=xy["log(price)"], data=data)
plt.show()

xy=data[["price", "lat"]].groupby("lat", as_index=False).mean()
xy["price"]=xy["price"].apply(lambda x:log(x))
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["lat"], y=xy["log(price)"], data=data)
plt.show()

xy=data[["price", "yr_built"]].groupby("yr_built", as_index=False).mean()
xy["price"]=xy["price"].apply(lambda x:log(x))
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["yr_built"].apply(lambda x: 2016-x), y=xy["log(price)"], data=data)
plt.show()

xy=data[["price", "yr_renovated"]].groupby("yr_renovated", as_index=False).mean()
xy["price"]=xy["price"].apply(lambda x:log(x))
xy["yr_renovated"]=xy["yr_renovated"].drop(0).apply(lambda x:log(x))
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["yr_renovated"].drop(0).apply(lambda x: 2016-x), y=xy["log(price)"], data=data)
plt.show()


# In[ ]:


print("log(price) and sqft_living procused super high MSE\nlog(price) and log(sqft_living) worked well")
xy=data[["price", "sqft_living"]].groupby("sqft_living", as_index=False).mean()
sns.scatterplot(x=xy["sqft_living"], y=xy["price"], data=data)
plt.show()

xy=data[["price", "sqft_living"]].groupby("sqft_living", as_index=False).mean()
xy.rename(columns={'sqft_living': 'log(sqft_living)'}, inplace=True)
sns.scatterplot(x=xy["log(sqft_living)"].apply(lambda x: log(x)), y=xy["price"], data=data)
plt.show()

xy=data[["price", "sqft_living"]].groupby("sqft_living", as_index=False).mean()
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["sqft_living"], y=xy["log(price)"].apply(lambda x: log(x)), data=data)
plt.show()

xy=data[["price", "sqft_living"]].groupby("sqft_living", as_index=False).mean()
xy.rename(columns={'sqft_living': 'log(sqft_living)','price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["log(sqft_living)"].apply(lambda x: log(x)), y=xy["log(price)"].apply(lambda x: log(x)), data=data)
plt.show()


# In[ ]:


print("Trying to make lat linear by abs(47.63-x) \nHad good impact on Model")
xy=data[["price", "lat"]].groupby("lat", as_index=False).mean()
xy["price"]=xy["price"].apply(lambda x:log(x))
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["lat"], y=xy["log(price)"], data=data)
plt.show()

# seems very useful
xy=data[["price", "lat"]].groupby("lat", as_index=False).mean()
xy["price"]=xy["price"].apply(lambda x:log(x))
xy.rename(columns={'price': 'log(price)'}, inplace=True)
sns.scatterplot(x=xy["lat"].apply(lambda x: abs(47.63-x)), y=xy["log(price)"], data=data)
plt.show()


# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#split in X,y
y=data.loc[:,"price"].apply(lambda x: log(x))

X=data.loc[:,["sqft_living","grade","floors","lat","long","bathrooms","bedrooms","yr_built","yr_renovated","view","waterfront","zipcode","condition","sqft_lot"]]
X["sqft_living"]= X["sqft_living"].apply(lambda x: log(x))
X["lat"]= X["lat"].apply(lambda x: abs(47.63-x))
X["long"]= X["long"].apply(lambda x: log(abs(x)))
X["yr_built"]= X["yr_built"].apply(lambda x: log(abs(x-1955)+1))
X["yr_renovated"]= X["yr_renovated"].apply(lambda x: log(x+1))
X["sqft_lot"]= X["sqft_lot"].apply(lambda x: log(x))

X["lat*long"]=X["lat"]*X["long"]
X["sqft_living*sqft_lot"]=X["sqft_living"]*X["sqft_lot"]

# split in train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=123)
#split in validation, test
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=123)


# In[ ]:


# Validation Set
regr= linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred= regr.predict(X_val)

print("Validation Set")
print("Root Mean Square Error: %2.f" % sqrt(mean_squared_error(np.exp(y_val.values),np.exp(y_pred))))
print('R2 score: %.2f' % r2_score(np.exp(y_val.values),np.exp(y_pred)))


# In[ ]:


#Test set
y_pred= regr.predict(X_test)
print("Test Set")
print("Root Mean Square Error (squared): %2.f" % sqrt(mean_squared_error(np.exp(y_test.values),np.exp(y_pred))))
print('R2 score: %.2f' % r2_score(np.exp(y_test.values),np.exp(y_pred)))


# In[ ]:


# Predicting real life data (Training)
#split in X,y 
y=data.loc[:,"price"].apply(lambda x: log(x))

X=data.loc[:,["sqft_living","sqft_lot","yr_built","bedrooms","bathrooms","lat","long","waterfront"]]
X["sqft_living"]= X["sqft_living"].apply(lambda x: log(x))
X["sqft_lot"]= X["sqft_lot"].apply(lambda x: log(x))
X["lat"]= X["lat"].apply(lambda x: abs(47.63-x))
X["long"]= X["long"].apply(lambda x: abs(x))
X["yr_built"]= X["yr_built"].apply(lambda x: log(x))


X["lat*long"]=X["lat"]*X["long"]
X["sqft_living*sqft_lot"]=X["sqft_living"]*X["sqft_lot"]

# split in train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)

regr_test= linear_model.LinearRegression()
regr_test.fit(X_train,y_train)
y_pred= regr_test.predict(X_test)


print("Root Mean Square Error: %2.f" % sqrt(mean_squared_error(np.exp(y_test.values),np.exp(y_pred))))
print('R2 score: %.2f' % r2_score(np.exp(y_test.values),np.exp(y_pred)))


# In[ ]:


index=["min","max","mean","coef"]
columns=list(X_test)
test = pd.DataFrame(index=index, columns=columns)
test.loc["min"]=list(X_test.min())
test.loc["max"]=list(X_test.max())
test.loc["mean"]=list(X_test.mean())
test.loc["coef"]=regr_test.coef_

test


# In[ ]:


# Predicting real life data (real examples)

rl_X=pd.DataFrame(columns=["sqft_living","sqft_lot","yr_built","bedrooms","bathrooms","lat","long","waterfront"])
rl_y=np.empty(10)

## Sold Price
# https://www.zillow.com/homes/recently_sold/King-County-WA/49015189_zpid/207_rid/globalrelevanceex_sort/47.820992,-121.05835,47.042521,-122.534638_rect/9_zm/
rl_X.loc[len(rl_X)]=[1770,7840,1968,4,3,47.7397,-122.185,0]
rl_y[len(rl_X)-1]=720000
# https://www.zillow.com/homes/recently_sold/King-County-WA/49120416_zpid/207_rid/globalrelevanceex_sort/47.820992,-121.05835,47.042521,-122.534638_rect/9_zm/5_p/
rl_X.loc[len(rl_X)]=[1950,4887,1911,2,2,47.535,-122.388, 1]
rl_y[len(rl_X)-1]=865000
# https://www.zillow.com/homes/recently_sold/King-County-WA/48829308_zpid/207_rid/200000-600000_price/830-2490_mp/globalrelevanceex_sort/47.754559,-121.539002,46.975099,-123.01529_rect/9_zm/
rl_X.loc[len(rl_X)]=[1810,17424,1994,3,3,47.364,-122.043, 0]
rl_y[len(rl_X)-1]=432000
# https://www.zillow.com/homes/recently_sold/King-County-WA/84756911_zpid/207_rid/globalrelevanceex_sort/47.633354,-121.886616,47.439119,-122.255688_rect/11_zm/
rl_X.loc[len(rl_X)]=[1289,1000,2009,2,3,47.532,-122.072, 0]
rl_y[len(rl_X)-1]=520000

i_sale=len(rl_X)

## Sale Price
# https://www.zillow.com/homes/for_sale/King-County-WA/49127321_zpid/207_rid/globalrelevanceex_sort/47.820992,-121.05835,47.042521,-122.534638_rect/9_zm/0_mmm/
rl_X.loc[len(rl_X)]=[2760,4839,1923,4,3,47.557,-122.375, 0]
rl_y[len(rl_X)-1]=550000
# https://www.zillow.com/homes/for_sale/King-County-WA/48662094_zpid/207_rid/200000-600000_price/830-2490_mp/globalrelevanceex_sort/47.804392,-121.289063,47.025674,-122.765351_rect/9_zm/0_mmm/
rl_X.loc[len(rl_X)]=[1290,4791,1925,1,1.5,47.513,-122.387, 0]
rl_y[len(rl_X)-1]=399000
# https://www.zillow.com/homedetails/1102-E-Hemlock-St-Kent-WA-98030/49077132_zpid/
rl_X.loc[len(rl_X)]=[2020,7701,1959,5,2,47.374,-122.220, 0]
rl_y[len(rl_X)-1]=367500
# https://www.zillow.com/homes/for_sale/King-County-WA/48702491_zpid/207_rid/globalrelevanceex_sort/47.633354,-121.886788,47.439118,-122.25586_rect/11_zm/4_p/0_mmm/
rl_X.loc[len(rl_X)]=[2040,8119,1963,4,2,47.502,-122.167, 0]
rl_y[len(rl_X)-1]=435000


rl_X["sqft_living"]= rl_X["sqft_living"].apply(lambda x: log(x))
rl_X["lat"]= rl_X["lat"].apply(lambda x: abs(47.63-x))
rl_X["long"]= rl_X["long"].apply(lambda x: abs(x))
rl_X["yr_built"]= rl_X["yr_built"].apply(lambda x: log(x))
rl_X["sqft_lot"]= rl_X["sqft_lot"].apply(lambda x: log(x))
rl_X["lat*long"]=rl_X["lat"]*rl_X["long"]
rl_X["sqft_living*sqft_lot"]=rl_X["sqft_living"]*rl_X["sqft_lot"]

prices_pred= np.exp(regr_test.predict(rl_X))
print("Sold")
for i in range (0,i_sale):
    print("House %i for %i; predicted -> %i" %(i, rl_y[i], prices_pred[i]))
print("\nOn Sale")
for i in range (i_sale,len(rl_X)):
    print("House %i for %i; predicted -> %i" %(i, rl_y[i], prices_pred[i]))

