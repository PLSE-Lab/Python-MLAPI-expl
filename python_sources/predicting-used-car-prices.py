#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# # Predicting used car prices
# Working on the Craigslist Dataset uploaded on kaggle, the aim on the project is to predict the `price` based on factors decided after analysis. Decription of columns:- 
# 
# * price - entry price
# * year - entry year
# * model - model of vehicle
# * condition - condition of vehicle
# * cylinders - no. of cylinders
# * fuel type - fuel variant 
# * odometer - miles travelled by vehicle
# * vin - vehicle identification number
# * size - size of vehicle
# * type - generic type of vehicle
# 
# Let us first explore the dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor


# In[ ]:


df = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
df.head(10)


# In[ ]:


print(df.columns)
print(df.shape)


# In[ ]:


df.isna().sum()


# There are a lot of null values for certain columns. We can use a thumb rule that if any column contains atleast 35% - 40% null values, we can remove those columns from consideration in our case. Such columns do not add much to our analysis and are generally irrelevant to the goal.

# In[ ]:


def remove_col(data):
    thresh = len(data) * 0.4
    cols = data.columns
    remove = []
    for col in cols:
        n_nulls = data[col].isna().sum()
        if n_nulls >= thresh:
            remove.append(col)
    return remove

rm_cols = remove_col(df)
df = df.drop(rm_cols,axis=1)
df.head(5)


# Let us look at the unique values for each column, especially those that are measured on nominal or ordinal scale. This can give us a sense of which columns can be important to us. Some variables even on a nominal scale have too many unique values, such variables would not add anything to the final model (like url).

# In[ ]:


df.nunique()


# From above, we see that there are columns having too many unique values and not a variable on interval/ratio scale, thus we can remove some columns we know will not be any help to us such as :-
# 
# * id
# * url
# * region
# * region_url
# * image_url
# * description
# * model
# * state
# * paint_color
# 
# The reason why we removed paint_color is because the color of the vehicle does not add anything to its price, mathematically it might be correlated, but in common sense we know that is not true.

# In[ ]:


rm_cols = [
    'id',
    'url',
    'region',
    'region_url',
    'image_url',
    'description',
    'model',
    'state',
    'paint_color'
]
df = df.drop(rm_cols,axis=1)
df.head(10)


# So our target variable is going to be the price column. Let us look at the distribution of the price column.

# In[ ]:


print(df.price.describe())
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.boxplot(df.price)


# We can see some absurd values in the `price` column. These values usually skew the data and pull the mean much higher than it actually would be. We will have to remove the outliers. To do so, we will set up a threshold using the `interquartile range` (IQR). Instead of deciding on a threshold myself, this method can easily give me a bound that would be reasonable with respect to the data in general.<br>
# interquartile range = 75% - 25%

# In[ ]:


descp = interquartile = df.price.describe()
interquartile = descp['75%'] - descp['25%']
thresh = interquartile * 1.5

df = df[df.price < thresh]
df.head(3)


# Now that we have it cleaned, let us confirm this with a `boxplot`.

# In[ ]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.boxplot(df.price)


# We see we no longer have any outliers. The distribution of prices is fairly spread between 0 and 14000 dollars.<br>
# Now let us look at the `odometer` and `year` column, as they are the other discrete interval scale variables.

# In[ ]:


df[['odometer','year']].describe()


# The `year` column seems unrealistic at 2021. Let us plot a countplot for the year to see the trend in buying/selling used cars on craigslist. We can tell that sales actually began increasing after 1960. Hence we will consider data after this point in time as years below that wud have less representative data.

# In[ ]:


df.year.value_counts()
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.countplot(df[df.year.between(1950,2020)].year)
# plt.xticks([0,15,30,45,60,70])


# In[ ]:


df = df[df.year.between(1960,2020)]
df.head(3)


# Similarly, the `odometer` column looks like it has outliers, the values seem too high. Thus we will have to trim these values. Let us view its dstribution using a `boxplot` to find outliers.

# In[ ]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.boxplot(df.odometer)


# We will apply the same `interquartile range` (IQR) method to remove these outliers. As mentioned above, such values will skew the data and our resutls.

# In[ ]:


interquartile = df.odometer.quantile(0.75) - df.odometer.quantile(0.25)
thresh = interquartile * 1.5
df = df[df.odometer < thresh]
df.head(3)


# In[ ]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.boxplot(df.odometer)


# We have achieved a good spread now, and that will be enough for now.

# Let us now look at the nominal or ordinal variables that we have. Lets understand the `manufacturer` and `type` columns. Let us understand our data and see vehicle of which manufacturers and type of vehicle are selling most online.

# In[ ]:


top_manufacturers = df.manufacturer.value_counts(dropna=False).iloc[:10]
print(top_manufacturers)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.barplot(x=top_manufacturers.index,y=top_manufacturers.values)
plt.xlabel('Manufacturers')
plt.ylabel('Number of vehicles')
plt.title('Vehicles from top 10 manufacturers',y=1.02)
plt.suptitle('Number of vehicles from the top 10 manufacturers listed on Craigslist',y=0.9)


# In[ ]:


top_types = df.type.value_counts().iloc[:10]
print(top_types)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.barplot(x=top_types.index,y=top_types.values)
plt.xlabel('Vehicle type')
plt.ylabel('Number of vehicles')
plt.title('Generic top 10 vehicle types',y=1.02)
plt.suptitle('Number of vehicles from the top 10 types of generic vehicle models listed on Craigslist',y=0.9)


# The `manufacturer` and `type` columns are important when we talk about sale of a car. Hence we will convert these columns into dummy binary columns. Similarly looking at the other variables like `fuel`,`transmission`,`drive`,`year` are important features when someone is buying a car.<br>
# Location is another influencing factor as prices of vehicles may depend upon locations, thus we have `lat` and `long` for this.<br>
# The `lat` and `long` columns still have null values, the null values in the nominal variables will be taken care of when we convert into dummy variables. For now, let us ignore these rows.

# In[ ]:


df = df.dropna(subset=['lat','long'])
df.head(5)


# Let us now try to visualize the correlation between variables, to do this we will use a `heatmap` and the `corr` function.

# In[ ]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns, annot=True,cmap='YlGnBu')


# All the variables seem to be independent of eachother, there is no single variable that directly influences `price` by alot.

# Now before we split the data and make it ready for the `MinMaxScaler` to scale the values, since price and odometer can have very high values whereas year cannot, thus it is important we do this. <br>
# We will convert out categorical variables into dummy binary columns so that it is easier for the model to understand.

# In[ ]:


df_cleaned = pd.get_dummies(df)
X = df_cleaned.iloc[:,1:]
y = df_cleaned.price
X.columns


# In[ ]:


scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X


# Now that we have our target variable in y and the independent variables in X, we will split the data for training and testing.

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)


# There are a number of models to choose from, we will try out the `RandomForestRegressor` here because :-
# * It handles high-dimensionality very well since it takes subsets of data.
# * It is extremely versatile and requires very little preprocessing
# * It is great at avoiding overfitting since each decision tree has low bias

# In[ ]:


model = RandomForestRegressor(n_estimators=25,random_state=0)
model.fit(X_train,y_train)
print(model.score(X_train,y_train))
pred = model.predict(X_test)
print(mae(y_test,pred))
print(y.mean())
model.score(X_test,y_test)


# As we can see our model did wonderful on train set, and is slightly underfitting on the test set, to better this we will have to do feature engg. Below we can see a `barplot` to identify the most important features for the model.

# In[ ]:


feature_imp = pd.Series(model.feature_importances_,index=df_cleaned.columns[1:])
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
feature_imp.sort_values(ascending=False)[:20].plot.barh()
plt.ylabel('Features')
plt.xticks([])

