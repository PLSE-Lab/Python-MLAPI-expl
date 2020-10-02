#!/usr/bin/env python
# coding: utf-8

# ## Get data on your hand
# This is a personal note of data handing.
# 
# #### Terms
# - dot-notation: Expression of . like data.Price.
# - Prediction Target: Target columns I'm trying to apply ML.
# - y: A variable name commonly used for prediction target.
# - Features: Columns with which we predict the prediction target.
# - X: A series of variable name commonly used for features.
# 
# #### Facts
# - .describe() shows numeric columns only.
# 

# # Data analysis

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
print(os.listdir("../input"))


# In[ ]:


avocado = pd.read_csv("../input/avocado.csv", index_col = "year") # DataFrame holds 'index' (ex. sort_index)
print("shape =", avocado.shape)
avocado.describe()


# In[ ]:


avocado.columns


# In[ ]:


avocado["region"]


# In[ ]:


avocado.sample(10)


# In[ ]:


avocado.region.value_counts() # Produce Series which holds 'region' as key, its number of record as value


# In[ ]:


y = avocado.AveragePrice # Setting AveragePrice as the prediction target by dot-notation
X = avocado[["Date", "Total Volume", "region"]] # Setting Date, type and region as the features
X.describe() # Hmm...


# In[ ]:


avocado[avocado.year == 2015].AveragePrice.tail(10)


# In[ ]:


avocado.region.unique()


# In[ ]:


avocado.region.value_counts()


# In[ ]:


g1 = avocado.groupby("region") # Create tables for each region
g2 = avocado.groupby("type")


# In[ ]:


g1.describe().head(10)


# In[ ]:


g2.describe().head(10)


# In[ ]:


avocado[avocado.region.isin(["SanDiego", "Chicago"])].head(10)


# In[ ]:


avocado.groupby('region').sum()


# In[ ]:


avocado.region.replace('NewYork', 'newyork')


# # Data cleaning

# In[ ]:





# In[ ]:


avocado.isnull()


# In[ ]:


# This code doesn't work
cols_with_missing = [col for col in avocado.columns 
                                 if avocado[col].isnull().any()]
train = train.drop(cols_with_missing, axis=1)
test = test.drop(cols_with_missing, axis=1)


# # Data augmentation

# In[ ]:


# Apply lambda to each value
avocado.apply(lambda n: n / 2 if n.dtype == 'float' else n, axis='columns')


# In[ ]:


columns = avocado.columns
names = {'AveragePrice':'price', 'Total Volume':'volume', 'Total Bags':'bags'}
avocado = avocado.rename(columns = names)


# In[ ]:


# Create custom column
avocado.assign(rate=(avocado.price / avocado.volume))


# # Visualization

# In[ ]:


avocado.AveragePrice.head(10).plot.bar()


# In[ ]:


avocado.sample(10).plot.scatter(x = 'Total Bags', y = 'AveragePrice') # This case has less overwrapping


# In[ ]:


avocado.sample(1000).plot.scatter(x = 'Total Bags', y = 'AveragePrice') # This case has a large overwrapping


# In[ ]:


avocado.sample(1000).plot.hexbin(x = 'Total Bags', y = 'AveragePrice', gridsize = 20)


# In[ ]:


sns.countplot(avocado.sample(1000).AveragePrice)


# In[ ]:


sns.kdeplot(avocado.sample(1000).AveragePrice)


# ## Prediction

# In[ ]:


def fit(X, y):
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
    model = DecisionTreeRegressor(random_state=1)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    mae = mean_absolute_error(test_y, pred_y)
    return model, mae


# In[ ]:


def formatting(d, nf, cf, tr=[]):
    """
    nf = numerical features
    cf = categolical features
    One stop function for
    - Drop NaN
    - One-hot encoding
    """
    
    # Drop NaN
    d = d[nf + cf + tr]
    d = d.dropna(axis=0)
    
    # One-hot encoding
    num_df = d[nf]
    cat_df = pd.get_dummies(d[cf])
    X = pd.concat([num_df, cat_df], axis=1)
    if len(tr) != 0:
        y = d[tr]
    else:
        y = None
    return X, y


# In[ ]:


train = avocado

nf = ['bags'] # numerical features
cf = ['region'] # categolical features
tr = ['price'] # target

X, y = formatting(train, nf, cf, tr)
model, mae = fit(X, y)
print(mae)


# In[ ]:




