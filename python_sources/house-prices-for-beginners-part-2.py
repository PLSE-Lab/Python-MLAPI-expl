#!/usr/bin/env python
# coding: utf-8

# # House Prices for Beginners - Part 2 (categorical columns)

# In[96]:


import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ## Load and explore data

# In[97]:


df_train = pd.read_csv('../input/train.csv')


# In[98]:


# Shift + tab to see docs.
df_train.head()


# ## Create total square footage column

# In[99]:


df_train['TotalSF'] = df_train['GrLivArea'] + df_train['TotalBsmtSF'] + df_train['GarageArea'] + df_train['EnclosedPorch'] + df_train['ScreenPorch']


# In[100]:


df_train['TotalSF'].head()


# ## Take the log of the sale price

# In[101]:


df_train['SalePrice'] = np.log(df_train['SalePrice'])


# ## Test model performance (baseline: only total square footage)
# 
# Let's start by getting a baseline using just the total square footage.

# In[105]:


train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    df_train[['TotalSF']], df_train['SalePrice'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())


# ## Test model performance (add overall quality)
# 
# Let's add the overall quality when training the model. Overall quality is a categorical variable but is already represented numerically (1 is bad, 10 is good), so we don't actually need to do any additional preprocessing to make use of it.

# In[106]:


train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    df_train[['TotalSF', 'OverallQual']], df_train['SalePrice'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())


# As you can see, our model is doing a lot better!
# 
# Let's take a look at our model's coefficients:

# In[107]:


print(f'Intercept: {model.intercept_}, coefficients: {model.coef_}')


# Since we now have 2 features, the model has extend our linear equation to include a new coefficient for the overall quality. Our equation now looks like this:
# 
# $y = \text{intercept} + \text{coef} * \text{TotalSf} + \text{coef2} * \text{OverallQual}$

# ## Prepare ordinal categorical variables
# 
# Let's add in another categorical variable that might help us make predictions. I think that external quality might have a reasonable influence on the price of the house, so I'll experiment with adding that in.
# 
# There's 2 things to note about the `ExterQual` column.
# 
# 1. The column has a string representation so it will need to be converted to a number somehow.
# 2. There's a implicit ordering to `ExterQual` that we want our model to be aware of (Excellent is better than Poor).
# 
# We can deal with the first note by converting our column to a Pandas categorical datatype, which maps a category to an integer. We can deal with the second, by providing the model with the appropriate ordering of the category.
# 
# Let's do that.

# In[108]:


df_train['ExterQual'] = df_train.ExterQual.astype('category')

# Set the ordering of the category.
df_train['ExterQual'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)

# The codes is a column with the category string mapped to a number.
df_train['ExterQual'] = df_train['ExterQual'].cat.codes


# ## Test model performance (add external quality)

# In[109]:


train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    df_train[['TotalSF', 'OverallQual', 'ExterQual']], df_train['SalePrice'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())


# That appears to have helped a bit but not much. Perhaps the information added by the external quality was already included in the overall quality column.
# 
# My intution says that the next most important predictor of house prices is the area that it's located in. We have a column called `Neighborhood` that might help here.
# 
# `Neighborhood` is a categorical column because there are a limited range of choices for it, but it's not clear what the underlying order should be. We can deal with non-ordinal categorical variables by one-hot encoding them.
# 
# One-hot encoding is best explained by example:
# 
# <img src="https://naadispeaks.files.wordpress.com/2018/04/mtimfxh.png?w=300">
# 
# Pandas has a method called `get_dummies` that can perform one-hot encoding for a column.

# In[119]:


df_train['Neighborhood'] = df_train['Neighborhood'].astype('category')


# In[120]:


dummies = pd.get_dummies(df_train['Neighborhood'])


# In[121]:


train_df_concat = pd.concat([df_train[['TotalSF', 'OverallQual', 'ExterQual']], dummies], axis=1)


# ## Test model performance (add neighborhood)

# In[63]:


train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    train_df_concat, df_train['SalePrice'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())


# Clearly that helps a fair bit! The last thing we'll look at is dealing with missing features. I'll start by summing up the missing values, then sort to find the columns with the most na.

# In[125]:


df_train.isna().sum().sort_values(ascending=False).head(20)


# When filling in missing values, you want to think about what that means? Is missing the data meaningful? For `MasVnrArea`, it might be null because there's no Masonry veneer, so a value of 0 might be useful.
# 
# For `LotFrontage`, it's more likely that it's N/A because someone failed to input that field when inputting the house detail.
# 
# One common approach to impute fields like that is to use the column's median. I'm adding another column called `isna` that might help the model figure out how to use the NA's. I'll need to ensure I save the median value to reuse when processing the test set.

# In[132]:


df_train['LotFrontage_isna'] = df_train.LotFrontage.isna()
df_train['LotFrontage'] = df_train.LotFrontage.fillna(df_train['LotFrontage'].median())
lot_frontage_na = df_train['LotFrontage'].median()


# In[133]:


train_df_concat = pd.concat([
    df_train[['TotalSF', 'OverallQual', 'ExterQual', 'LotFrontage', 'LotFrontage_isna']], dummies], axis=1)

train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    train_df_concat, df_train['SalePrice'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())


# Looks like the performance is slightly worse. It's possible that the performance is only worse on this validation set. We can use a thing called cross validation to get around that problem. That's coming up in the next notebook.
# 
# Let's run the same preprocessing on our test set and submit.

# ## Prepare test and submit predictions

# In[134]:


test_df = pd.read_csv('../input/test.csv')

test_df['TotalSF'] = test_df['GrLivArea'] + test_df['TotalBsmtSF'].fillna(0) + test_df['GarageArea'].fillna(0) + test_df['EnclosedPorch'].fillna(0) + test_df['ScreenPorch'].fillna(0)

test_df['ExterQual'] = test_df.ExterQual.astype('category')
test_df['ExterQual'].cat.set_categories(
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True
)
test_df['ExterQual'] = test_df['ExterQual'].cat.codes

test_df['LotFrontage_isna'] = test_df['LotFrontage'].isna()
test_df['LotFrontage'] = lot_frontage_na

test_dummies = pd.get_dummies(test_df['Neighborhood'])
test_df_concat = pd.concat([test_df[['TotalSF', 'OverallQual', 'ExterQual', 'LotFrontage', 'LotFrontage_isna']], test_dummies], axis=1)


# In[135]:


test_preds = model.predict(test_df_concat)


# In[136]:


pd.DataFrame(
    {'Id': test_df['Id'], 'SalePrice': np.exp(test_preds)}).to_csv('my_sub_more_features.csv', index=False)


# In[ ]:




