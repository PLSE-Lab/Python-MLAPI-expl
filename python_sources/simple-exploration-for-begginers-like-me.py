#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques
# 
# [Link to the Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
# 
# Author: Diego Rodrigues [@polaroidz](https://github.com/polaroidz)

# ## Importing and Loading Data

# In[ ]:


import numpy as np # vector manipulation
import pandas as pd # dataframe manipulation

from sklearn.model_selection import train_test_split # spliting train and test dataset
from sklearn.linear_model import LinearRegression # linear regression
from sklearn.tree import DecisionTreeRegressor # decision tree
from sklearn.ensemble import RandomForestRegressor # random forest

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt # plotting

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings # just for ignoring annoying warnings
warnings.filterwarnings('ignore')


# The target variable on our dataset is called SalePrice

# In[ ]:


TARGET = 'SalePrice'


# In[ ]:


dataset = pd.read_csv('../input/train.csv')


# ### Data Exploration

# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# So, we have 81 variables.

# In[ ]:


desc = dataset.describe()
desc


# From these 81, we have 38 numerical variables. Great! Lets Start with those

# In[ ]:


count = desc.iloc[0]
count = count[count == dataset.shape[0]]
num_columns = count.index.values

num_columns


# In order to see which of the linear variables are more interesting for some type of regression, we will plot them in relation to SalePrice and notice which of them are more colinear

# In[ ]:


for column in num_columns:
    plt.title(column)
    plt.scatter(dataset[column], dataset[TARGET])
    plt.show()


# Looking at those plots we can see that these variables show the most colinerarity with SalePrice.

# In[ ]:


selected = [
    'GarageArea',       # Garage
    'GarageCars', 
    'TotRmsAbvGrd',     # Above Ground
    'LotArea',          # Area
    'OverallQual',      # Quality
    'OverallCond',
    'TotalBsmtSF',      # Second Floor
    '1stFlrSF', 
    '2ndFlrSF',
    'YearBuilt',        # Year
    'YearRemodAdd'
]


# Seeing how the variables relate with each other

# In[ ]:


corr = dataset[selected + [TARGET]].corr()
corr


# In[ ]:


corr_saleprice = corr['SalePrice']
corr_saleprice


# Lets select only those with colinearity above 0.6

# In[ ]:


corr_saleprice[corr_saleprice > 0.60]


# In[ ]:


linear_features = [
    'GarageArea',
    'GarageCars',
    'OverallQual',
    'TotalBsmtSF',
    '1stFlrSF'
]


# ### Testing the Selected Linear Variables

# In[ ]:


linear_dataset = dataset[linear_features]


# In[ ]:


y = dataset[TARGET]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(linear_dataset, y, test_size=0.30, random_state=42)


# In[ ]:


model = LinearRegression().fit(X_train, y_train)


# In[ ]:


model.score(X_test, y_test)


# We can see that only from the linear features we already achieved a good result.

# ### Selecting Categorial Variables

# In[ ]:


cat_dataset = dataset.drop(num_columns, axis=1)
cat_dataset.head()


# Dropping missing values

# In[ ]:


count = cat_dataset.isna().sum()
count = count[count > 0]
count


# In[ ]:


cat_dataset = cat_dataset.drop(count.index.values, axis=1)
cat_dataset.head()


# Lets see the value counting for each of the selected categorical variables

# In[ ]:


for col in cat_dataset.columns:
    print(cat_dataset[col].value_counts())
    print("\n")


# From those, let's select the most interesting in terms of variance

# In[ ]:


cat_features = [
    'MSZoning',
    'LandContour',
    'LotShape',
    'LotConfig',
    'HouseStyle',
    'Foundation',
    'HeatingQC',
    'ExterQual',
    'KitchenQual',
    'SaleCondition'
]


# In[ ]:


sel_dataset = cat_dataset[cat_features]
sel_dataset.head()


# ### Changing categorical variables into numerical

# #### MSZoning

# In[ ]:


# MSZoning : RL or not RL
sel_dataset['MSZoning'] = np.where(sel_dataset['MSZoning'] == 'RL', 1, 0)
sel_dataset.head()


# #### LandContour

# In[ ]:


sel_dataset['LandContour'] = np.where(sel_dataset['LandContour'] == 'Lvl', 1, 0)
sel_dataset.head()


# #### LotShape

# In[ ]:


sel_dataset['LotShape'] = np.where(sel_dataset['LotShape'] == 'Reg', 1, 0)
sel_dataset.head()


# #### LotConfig

# In[ ]:


conditions = [
    sel_dataset['LotConfig'] == 'Inside',
    sel_dataset['LotConfig'] == 'Corner'
]

choices = [2, 1]

sel_dataset['LotConfig'] = np.select(conditions, choices, default=0)
sel_dataset.head()


# #### HouseStyle

# In[ ]:


conditions = [
    (sel_dataset['HouseStyle'] == '2Story') | (sel_dataset['HouseStyle'] == '1Story') 
]

choices = [1]

sel_dataset['HouseStyle'] = np.select(conditions, choices, default=0)
sel_dataset.head()


# #### Foundation

# In[ ]:


conditions = [
    (sel_dataset['Foundation'] == 'PConc'), 
    (sel_dataset['Foundation'] == 'CBlock') 
]

choices = [2, 1]

sel_dataset['Foundation'] = np.select(conditions, choices, default=0)
sel_dataset.head()


# #### HeatingQC

# In[ ]:


conditions = [
    (sel_dataset['HeatingQC'] == 'Ex'), 
    (sel_dataset['HeatingQC'] == 'TA'),
    (sel_dataset['HeatingQC'] == 'Gd')
]

choices = [3, 2, 1]

sel_dataset['HeatingQC'] = np.select(conditions, choices, default=0)
sel_dataset.head()


# #### ExterQual

# In[ ]:


conditions = [
    (sel_dataset['ExterQual'] == 'TA'), 
    (sel_dataset['ExterQual'] == 'Gd')
]

choices = [2, 1]

sel_dataset['ExterQual'] = np.select(conditions, choices, default=0)
sel_dataset.head()


# #### KitchenQual

# In[ ]:


conditions = [
    (sel_dataset['KitchenQual'] == 'TA'), 
    (sel_dataset['KitchenQual'] == 'Gd')
]

choices = [2, 1]

sel_dataset['KitchenQual'] = np.select(conditions, choices, default=0)
sel_dataset.head()


# #### SaleCondition

# In[ ]:


conditions = [
    (sel_dataset['SaleCondition'] == 'Normal')
]

choices = [1]

sel_dataset['SaleCondition'] = np.select(conditions, choices, default=0)
sel_dataset.head()


# ### Testing the selected categorical variables

# In[ ]:


y = dataset[TARGET]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(sel_dataset, y, test_size=0.30, random_state=42)


# In[ ]:


model = DecisionTreeRegressor().fit(X_train, y_train)


# In[ ]:


model.score(X_test, y_test)


# The result isn't great as expected, but good enough to indicate that we are getting somewhere.

# ### Testing both selected categorical and numerical variables

# In[ ]:


both_dataset = pd.concat([linear_dataset, sel_dataset], axis=1, sort=False)
both_dataset.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(both_dataset, y, test_size=0.30, random_state=42)


# In[ ]:


model = RandomForestRegressor().fit(X_train, y_train)
model.score(X_test, y_test)


# Good enough accuracy for now.

# #### One-Hot encoding categorical variables 

# In[ ]:


one_hot = OneHotEncoder().fit(sel_dataset)


# In[ ]:


oh_dataset = one_hot.transform(sel_dataset)
oh_dataset = pd.DataFrame(oh_dataset.toarray())


# In[ ]:


both_dataset = pd.concat([linear_dataset, oh_dataset], axis=1, sort=False)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(both_dataset, y, test_size=0.30, random_state=42)


# In[ ]:


model = RandomForestRegressor().fit(X_train, y_train)
model.score(X_test, y_test)


# #### Scalling our numerical variables

# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(linear_dataset)


# In[ ]:


scaled_dataset = scaler.transform(linear_dataset)
scaled_dataset = pd.DataFrame(scaled_dataset)


# In[ ]:


both_dataset = pd.concat([scaled_dataset, oh_dataset], axis=1, sort=False)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(both_dataset, y, test_size=0.30, random_state=42)


# In[ ]:


model = RandomForestRegressor().fit(X_train, y_train)
model.score(X_test, y_test)


# #### Scaling the target variable

# In[ ]:


y_log = np.log(y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(both_dataset, y_log, test_size=0.30, random_state=42)


# In[ ]:


model = RandomForestRegressor().fit(X_train, y_train)
model.score(X_test, y_test)


# #### Grid Searching Model Parameters

# In[ ]:


model.get_params()


# In[ ]:


params = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}


# In[ ]:


f_model = RandomizedSearchCV(model, params, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[ ]:


f_model.fit(X_train, y_train)


# In[ ]:


f_model.score(X_test, y_test)


# In[ ]:


import pickle


# In[ ]:


with open('model.pkl', 'wb') as f:
    pickle.dump(f_model, file=f)


# ### Measuring Perfomance on the Test Dataset

# In[ ]:


dataset = pd.read_csv('../input/test.csv')
dataset.head()


# In[ ]:


X_linear = dataset[linear_features] 
X_cat = dataset[cat_features]


# In[ ]:


X_cat['MSZoning'] = np.where(X_cat['MSZoning'] == 'RL', 1, 0)
X_cat['LandContour'] = np.where(X_cat['LandContour'] == 'Lvl', 1, 0)
X_cat['LotShape'] = np.where(X_cat['LotShape'] == 'Reg', 1, 0)

conditions = [
    X_cat['LotConfig'] == 'Inside',
    X_cat['LotConfig'] == 'Corner'
]

choices = [2, 1]

X_cat['LotConfig'] = np.select(conditions, choices, default=0)

conditions = [
    (X_cat['HouseStyle'] == '2Story') | (X_cat['HouseStyle'] == '1Story') 
]

choices = [1]

X_cat['HouseStyle'] = np.select(conditions, choices, default=0)

conditions = [
    (X_cat['Foundation'] == 'PConc'), 
    (X_cat['Foundation'] == 'CBlock') 
]

choices = [2, 1]

X_cat['Foundation'] = np.select(conditions, choices, default=0)

conditions = [
    (X_cat['HeatingQC'] == 'Ex'), 
    (X_cat['HeatingQC'] == 'TA'),
    (X_cat['HeatingQC'] == 'Gd')
]

choices = [3, 2, 1]

X_cat['HeatingQC'] = np.select(conditions, choices, default=0)

conditions = [
    (X_cat['ExterQual'] == 'TA'), 
    (X_cat['ExterQual'] == 'Gd')
]

choices = [2, 1]

X_cat['ExterQual'] = np.select(conditions, choices, default=0)


conditions = [
    (X_cat['KitchenQual'] == 'TA'), 
    (X_cat['KitchenQual'] == 'Gd')
]

choices = [2, 1]

X_cat['KitchenQual'] = np.select(conditions, choices, default=0)

conditions = [
    (X_cat['SaleCondition'] == 'Normal')
]

choices = [1]

X_cat['SaleCondition'] = np.select(conditions, choices, default=0)


# In[ ]:


X_cat.head()


# In[ ]:


X_cat = one_hot.transform(X_cat)
X_cat = pd.DataFrame(X_cat.toarray())


# In[ ]:


X_linear = scaler.transform(X_linear)
X_linear = pd.DataFrame(X_linear)


# In[ ]:


X = pd.concat([X_linear, X_cat], axis=1, sort=False)


# In[ ]:


X.head()


# In[ ]:


X = X.fillna(0)


# In[ ]:


y_final = f_model.predict(X)


# In[ ]:


y_final = np.exp(y_final)


# In[ ]:


y_final = pd.DataFrame({'SalePrice': y_final})


# In[ ]:


y_final.head()


# In[ ]:


y.head()


# Seems legit

# In[ ]:


Ids = dataset['Id']
Ids.head()


# In[ ]:


submission = pd.concat([Ids, y_final], axis=1, sort=False)
submission.head()


# In[ ]:


submission.to_csv('./submission.csv', index=False)


# In[ ]:




