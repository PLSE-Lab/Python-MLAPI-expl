#!/usr/bin/env python
# coding: utf-8

# Good Resources for Python
# https://datacamp.com
# https://python-graph-gallery.com
# 
# **Notes:
# 2019-06-18 (Rev 1)**
# <br>-Scored Budget, Popularity, Runtime for dependent variable Revenue.
# <br>-Score 3.09
# 
# **2019-06-18 (Rev 2)**
# <br>-Scored One Hot Encoding, Year, Month, Language
# <br>-Score 2.80
# 
# **2019-06-18 (Rev 3)**
# <br>-Small Forest Depth Pruning
# <br>-Score 2.77

# In[ ]:


# Data Processing and Cleaning
import numpy as np
import pandas as pd


# Data Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz

# Modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

#Miscellaneous
from tqdm import tqdm_notebook

# Input data files are available in the "../input/" directory.
import os
import copy
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train.head(3).T


# In[ ]:


train.info()


# # Copy data
# 

# In[ ]:


rf_train = copy.copy(train)
rf_test = copy.copy(test)


# # EDA

# In[ ]:


sns.pairplot(rf_train)


# In[ ]:


sns.jointplot(x=rf_train["popularity"], y=rf_train["revenue"], kind='scatter')


# In[ ]:


rf_train['original_language'].value_counts()


# In[ ]:


rf_train['status'].value_counts()


# In[ ]:





# # Data Normalization

# In[ ]:


#from sklearn import preprocessing
#scaler = preprocessing.StandardScaler()
#rf_train = scaler.fit_transform(rf_train)
#rf_train = pd.DataFrame(rf_train, columns=['popularity','runtime'])
#rf_test = scaler.fit_transform(rf_test)
#rf_test = pd.DataFrame(rf_test, columns=['popularity','runtime'])


# # One Hot Encoding

# In[ ]:


rf_train['release_date'] = pd.to_datetime(rf_train['release_date'])
rf_train['year'], rf_train['month'] = rf_train['release_date'].dt.year, rf_train['release_date'].dt.month

rf_test['release_date'] = pd.to_datetime(rf_test['release_date'])
rf_test['year'], rf_test['month'] = rf_test['release_date'].dt.year, rf_test['release_date'].dt.month


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le_language_train = LabelEncoder()
#Train
rf_train['language_encoded'] = le_language_train.fit_transform(rf_train.original_language)


rf_train.head()


# In[ ]:


rf_train.dtypes


# In[ ]:


le_language_test = LabelEncoder()


rf_test['language_encoded'] = le_language_test.fit_transform(rf_test.original_language)


rf_test.head()


# In[ ]:


rf_test.dtypes


# In[ ]:


#rf_train[['id','budget','popularity','runtime','revenue']]
rf_train.drop(['belongs_to_collection','genres','homepage','imdb_id','original_language','original_title',
              'overview','poster_path','production_companies','production_countries','release_date',
              'spoken_languages','status','tagline','title','Keywords','cast','crew'], axis=1, inplace=True)

rf_train.head()


# In[ ]:


#rf_train[['id','budget','popularity','runtime','revenue']]
rf_test.drop(['belongs_to_collection','genres','homepage','imdb_id','original_language','original_title',
              'overview','poster_path','production_companies','production_countries','release_date',
              'spoken_languages','status','tagline','title','Keywords','cast','crew'], axis=1, inplace=True)
rf_test.head()


# Drop Training Data NAs

# In[ ]:


#Count NAs
rf_test.isna().sum()


# In[ ]:


rf_train.isna().sum()


# In[ ]:


#Lets fill with Mean
mean_runtime=rf_train.iloc[:,3]
mean_runtime.head()


# Fill NAs with Mean

# In[ ]:


mean_runtime.mean()


# In[ ]:


rf_train=rf_train.fillna(mean_runtime.mean())
rf_test=rf_test.fillna(mean_runtime.mean())


# In[ ]:


rf_train.isna().sum()


# In[ ]:


# comparing sizes of data frames 
print("Old data frame length:", len(train), "\nNew data frame length:",  
       len(rf_train), "\nNumber of rows with at least 1 NA value: ", 
       (len(train)-len(rf_train))) 


# In[ ]:


# comparing sizes of data frames 
print("Old data frame length:", len(test), "\nNew data frame length:",  
       len(rf_test), "\nNumber of rows with at least 1 NA value: ", 
       (len(test)-len(rf_test))) 


# In[ ]:


#Dont need this since filled with Mean but kept for others to use
#rf_train =rf_train.dropna(how ='any') 
#rf_test=rf_test.dropna(how ='any') 


# In[ ]:


rf_train.dtypes


# In[ ]:





# # Random Forest Time
# 

# In[ ]:


# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
y_train = rf_train['revenue']
x_train = rf_train.drop(['revenue'], axis=1).values 
x_test = rf_test.values

# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor

# Create Decision Tree with max_depth = 6
decision_tree = DecisionTreeRegressor(max_depth = 6)
decision_tree.fit(x_train, y_train)


# Exporting Results
# 
# 

# In[ ]:


# Predicting results for test dataset
y_pred = decision_tree.predict(x_test)
submission3 = pd.DataFrame({
        "id": rf_test['id'],
        "revenue": y_pred
    })

# Output Submission
submission3.to_csv('submission3.csv', index=False)


# In[ ]:


#Verify this is the correct row count for submissions.
submission3.info()


# In[ ]:


submission3

