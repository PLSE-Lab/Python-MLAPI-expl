#!/usr/bin/env python
# coding: utf-8

# <h1>HOUSE PRICE PREDICTION</h1>

# <h2>Import Libraries:</h2>

# In[ ]:


import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from pandas.api.types import is_string_dtype

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# <h2>Extract Data:</h2>

# In[ ]:


#Import data into a data frame
data = pd.read_csv('../input/Melbourne_housing_FULL.csv') #Imports data in a data frame

#Select columns for X
cols = data.columns
cols = cols.drop('Price')

data.info()


# In[ ]:


#Drop down data instances whose Sale Price is not defined
data = data[data.Price.notnull()]

#Display 1st five instances of data
data.head()


# <h2>Data Exploration, Cleaning and Transformation:</h2>

# In[ ]:


#Seperate categorical from numerical data
categorical = data.select_dtypes(exclude = [np.number])
numerical = data.select_dtypes(include = [np.number])


# In[ ]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
#Plot numerical data against Sale Price
fig, axes = plt.subplots(4, 3, figsize=(12, 12))

for idx, feat in enumerate(numerical.columns.difference(['Price'])):
    ax = axes[int(idx / 3), idx%3]
    sns.scatterplot(x=feat, y='Price', data=numerical, ax=ax);
    ax.set_xlabel(feat)
    ax.set_ylabel('Price')
    
fig.tight_layout();


# In[ ]:


#Describe Sale Price's characteristics
data['Price'].describe()


# In[ ]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
#Plot the SalePrice of each instance
sns.distplot(data['Price'])


# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % data['Price'].skew())
print("Kurtosis: %f" % data['Price'].kurt())


# In[ ]:


#Tranform categorical data into numerical data for training purpose
for col, col_data in data.items():
    if is_string_dtype(col_data):
        data[col] = data[col].astype('category').cat.as_ordered().cat.codes


# In[ ]:


# SalePrice correlation matrix
corrmat = data.corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.set(font_scale=0.5)
sns.heatmap(corrmat,annot=True, square=True, fmt='.2f', vmax=.8);


# <ul>
#     <li><b>{ ' Rooms ', ' Bedroom2 ', ' Bathroom ', ' Car ' }</b> are <b>positively correlated</b> to Price attribute.
#     <li><b>{ ' Type ', ' Distance ', ' YearBuilt ' }</b> are <b>negetively correlated</b> to Price attribute.
# </ul>

# In[ ]:


crcols = ['Price', 'Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'YearBuilt']
corrmat = data[crcols].corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.set(font_scale=0.75)
sns.heatmap(corrmat,annot=True, square=True, fmt='.2f', vmax=.8);


# <ul>
#     <li><b>Bedroom2</b> and <b>Room</b> are strongly correlated to each other. Hence, a wise choice will be to remove Bedroom2 from analysis.
# </ul>

# In[ ]:


#Feature Selection
crcols.remove('Bedroom2')
cols = crcols
data = data[cols]
data.head()


# In[ ]:


#Fill in null data cells using Simple Imputer
y = data.Price
X = SimpleImputer().fit_transform(data[cols])


# In[ ]:


#Split the data into training and validation data sets
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.3)

#Standardize the data before feeding it to PCA algorithm
scaler = StandardScaler()
scaler.fit(train_X)

train_X = scaler.transform(train_X)
val_X = scaler.transform(val_X)

#Use PCA algorithm to reduce the number of features to speed up training 
pca = PCA(.95)
pca.fit(train_X)

train_X = pca.transform(train_X)
val_X = pca.transform(val_X)


# <h2>Train Model:</h2>

# In[ ]:


#Train a model using XGBRegressor
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, nthread=10)
model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)], verbose=False)


# <h2>Cross Validate: </h2>

# In[ ]:


#Validate the trained model using validation data-set
cvt = cross_val_score(model, X, y)

mae = cross_val_score(model, X, y, scoring = 'neg_mean_absolute_error')

print("Model Accuracy:\t",cvt.mean())
print("\nMean Absolute Error:\t",(-1 * mae.mean()))


# In[ ]:


#Predict Sale Price for houses in cross validation set
predict_y = model.predict(val_X)


# In[ ]:


#Plot the actual Price value against predicted Sale Price 
g = sns.jointplot(x= val_y, y= predict_y, kind='reg', xlim=(0,8500000), ylim=(0,8500000),
                  joint_kws={'line_kws':{'color':'darkorange'}})
g.set_axis_labels(xlabel='Actual Price', ylabel='Predicted Price')


# In[ ]:




