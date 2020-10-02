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


# In this notebook, we will attempt to predict the price of diamonds after analysing the effect of different physical variables that influence the price. We will use different regression techniques to model the price and evaluate their performance.
# 

# ### Importing the libraries and dataset

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 


# In[ ]:


df = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
df.head()


# The dataset contains the prices and other attributes of almost 54,000 diamonds. The columns are as follows:
# 
#  - carat weight of the diamond (0.2--5.01)
# 
#  - cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# 
#  - color diamond colour, from J (worst) to D (best)
# 
#  - clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
# 
#  - x length in mm (0--10.74)
# 
#  - y width in mm (0--58.9)
# 
#  - z depth in mm (0--31.8)
# 
#  - depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
# 
#  - table width of top of diamond relative to widest point (43--95)
# 
#  - price (dependent variable)
# 
#  We will use regression methods to model the price according to the different features.

# In[ ]:


df.drop('Unnamed: 0',axis=1, inplace=True)
df = df.reindex(columns=["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "price"])


# ### EDA

# In[ ]:


len(df)


# In[ ]:


df.describe()


# In[ ]:


# check for missing values 
df.isnull().any()


# Let us look at the distribution of the target variable

# In[ ]:


sns.distplot(df['price'])


# We see that the target variable is right-skewed. We can take the log transform this variable so that it becomes normally distributed. A normally distributed target variable helps in better modelling the relationship of the target variable with the independent variables.

# In[ ]:


# Skewness 
print("The skewness of the Price in the dataset is {}".format(df['price'].skew()))


# Let us now log-transform this variable and see if the distribution can get any more closer to normal 

# In[ ]:


# Transforming the target variable
target = np.log(df['price'])
print("Skewness: {}".format(target.skew()))
sns.distplot(target)


# Let us now examine each of the independent variables

# #### Carat

# In[ ]:


df['carat'].hist()


# We see that most of the diamond carats range from 0.2-1.2
# 

# #### Cut

# In[ ]:


df['cut'].unique()


# In[ ]:


sns.countplot(x='cut', data=df)


# We can infer that majority of the cuts are of "Ideal" or "Premium" type, whereas there are very few "Fair" cuts in the data.

# #### Color

# In[ ]:


df['color'].unique()


# In[ ]:


sns.countplot(x='color', data=df)


# #### Clarity 

# In[ ]:


df['clarity'].unique()


# In[ ]:


sns.countplot(df['clarity'])


# Here, we can infer that most of the diamonds have claritites of 'SI1' or 'VS2'
# 

# #### Depth and Table

# In[ ]:


fig, ax = plt.subplots(2, figsize=(10,10))
df['depth'].hist(ax=ax[0])
df['table'].hist(ax=ax[1])
ax[0].set_title("Distribution of depth")
ax[1].set_title("Distribution of table")


# #### x,y,z

# In[ ]:


fig, ax = plt.subplots(3, figsize=(10,10))
df['x'].hist(ax=ax[0])
df['y'].hist(ax=ax[1])
df['z'].hist(ax=ax[2])
ax[0].set_title("Distribution of x")
ax[1].set_title("Distribution of y")
ax[2].set_title("Distribution of z")


# #### Price

# In[ ]:


df['price'].hist()


# ### Feature Selection

# In[ ]:


# Using Pearson Correlation 
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True,cmap=plt.cm.Reds)
plt.show()


# In[ ]:


# correlation with output variable 
cor_target = abs(cor["price"])

# Selecting highly correlated features 
relevent_features = cor_target[cor_target>0.5]
relevent_features


# In[ ]:


df.drop(['depth', 'table'], axis=1, inplace=True)


# In[ ]:


df.head()


# ## REGRESSION

# In[ ]:


# Encoding the categorical data 
# Encoding the independent variables
dummy_cut = pd.get_dummies(df['cut'],drop_first=True)   # drop_first to avoid the dummy variable trap
df = pd.concat([df, dummy_cut], axis=1)
df = df.drop('cut',axis=1)
df.head()


# In[ ]:


dummy_color = pd.get_dummies(df['color'], drop_first=True)   
df = pd.concat([df, dummy_color], axis=1)
df = df.drop('color',axis=1)
df.head()


# In[ ]:


dummy_clarity = pd.get_dummies(df['clarity'], drop_first=True)
df = pd.concat([df, dummy_clarity], axis=1)
df = df.drop('clarity', axis=1)
df.head()


# ### Splitting the data into training and test sets 

# In[ ]:


order = df.columns.to_list()
order


# In[ ]:


order = ['carat',
 'x',
 'y',
 'z',
 'Good',
 'Ideal',
 'Premium',
 'Very Good',
 'E',
 'F',
 'G',
 'H',
 'I',
 'J',
 'IF',
 'SI1',
 'SI2',
 'VS1',
 'VS2',
 'VVS1',
 'VVS2',
  'price']


# In[ ]:


df = df[order]


# In[ ]:


df.head()


# In[ ]:


X = df.iloc[:,:-1].values
y = df.iloc[:,21].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


# ### Multiple Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import r2_score, mean_squared_error

regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[ ]:


# making predictions
y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


mlr_score = regressor.score(X_test, y_test)


# ### Support Vector Regression

# In[ ]:


from sklearn import preprocessing, svm

X_svm = X.copy()
X_svm = preprocessing.scale(X_svm)

X_svm_train, X_svm_test, y_svm_train, y_svm_test = train_test_split(X_svm, y, test_size=0.2, random_state=0)


# In[ ]:


clf = svm.SVR(kernel='linear')
clf.fit(X_svm_train, y_svm_train)


# In[ ]:


svr_score = clf.score(X_svm_test,y_svm_test)


# ### Decision Tree Regression 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
regressor_dt = DecisionTreeRegressor(random_state=0)
regressor_dt.fit(X_train, y_train)


# In[ ]:


regressor_dt.predict(X_test)


# In[ ]:


dt_score = regressor_dt.score(X_test, y_test)


# ### Random Forest Regression
# 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators=100, random_state=0)
regressor_rf.fit(X_train, y_train)


# In[ ]:


rf_score = regressor_rf.score(X_test, y_test)


# ### Best Model

# In[ ]:


print('Multiple Linear Regression accuracy:', mlr_score)
print('SVR score: ', svr_score)
print('Decision Tree Regression score: ', dt_score)
print('Random Forest Regression score: ', rf_score)


# We can conclude that the Random Forest Regression model performed the best with an accuracy of 97.4%

# In[ ]:




