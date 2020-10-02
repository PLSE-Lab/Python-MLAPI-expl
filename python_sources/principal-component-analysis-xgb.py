#!/usr/bin/env python
# coding: utf-8

# ![](https://nnimgt-a.akamaihd.net/transform/v1/crop/frm/5Q2j7ezUfQBfUJsaqK3gfB/8119e2b2-63e5-48a7-8fa6-eff00e9ab4ab.jpg/r0_179_3504_2157_w1200_h678_fmax.jpg)

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


# ## EDA

# In[ ]:


import pandas as pd
df1 = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')


# In[ ]:


df1.head()


# In[ ]:


df1.describe()


# In[ ]:


df1['RainToday'].replace({'No':0,'Yes':1},inplace = True)    
df1['RainTomorrow'].replace({'No':0,'Yes':1},inplace = True)   # replacing label's values
df = df1.drop(['Date'],axis=1)  # unsignificance feature
df.shape


# Let's find categorical variables for label encoding

# In[ ]:


categorical_features = df.select_dtypes(include = ["object"]).columns
categorical_features


# Create dummy variable for every categorical column 

# In[ ]:


df = pd.get_dummies(df,columns=categorical_features,drop_first=True)


# We have lot of null values here but as out primary focus is to perform PCA so let's just fill mean values wherever null values occured 

# In[ ]:


df.isnull().sum(axis=0)


# In[ ]:


df = df.fillna(df.mean())


# Scaling all the values from dataset

# In[ ]:


from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler() 

scaled = scaler.fit_transform(df) 


# In[ ]:


scaled


# ## Need of PCA
# Machine learning needs a good amount of data to train and test. But having large or good amount of data has its own pitfall, which is curse of dimensionality as huge amount of data can have lots of inconsistencies of features which may increase computation time and increase chances of making model bad. Hence to get rid of this curse of dimensionality a technique named Principal Component Analysis (PCA) is introduced.
# ## Principal Component Analysis (PCA)
# PCA is unsupervised linear dimensionality reduction technique to extract information from high dimensional space to lower dimensional sub space by preserving most significance variation of data. PCA allows to identify correlations and patterns among features in data so that it can be transformed into less dimensionality and with only most number of significance features without an important loss in data. Principal components are eigenvectors of a covariance matrix. The data which is to be analysed has to be scaled as results are highly sensitive <br>
# <br>
# **PCA consist of three main steps** 
# 1. Computing covariance matrix of data
# 2. Computing eigen values and vectors of covariance matrix
# 3. Using these eigen values and vectors to reduce dimensions and transform data 	
# 
# These steps were had to be done individually until sklearn released PCA, now these all 
# steps has one stop solution predefined in PCA

# In[ ]:


from sklearn.decomposition import PCA 
  
pca_model = PCA(n_components = 2) 
pca = pca_model.fit_transform(scaled)  


# But we dont know if the component value is suitable with 2 or any other number.<br>
# Let's check for n_components=2 [(reference)](http://stackoverflow.com/questions/57293716/sklearn-pca-explained-variance-and-explained-variance-ratio-difference)

# In[ ]:


variance=np.var(pca,axis=0)
variance_ratio = variance/np.sum(variance)
print(variance_ratio)


# The variance ratio by rounding off gives value of 0.99 which is almost best so we can continue with n_components=2

# ## Visualisation
# Find corelation between both the components

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize =(8, 6)) 
  
plt.scatter(pca[:, 0], pca[:, 1], c = df1['RainTomorrow'], cmap ='plasma') 
  
plt.xlabel('First Principal Component') 
plt.ylabel('Second Principal Component') 


# Find correlation between both components as how close their covariance are.

# In[ ]:


import seaborn as sns
df_comp = pd.DataFrame(pca_model.components_, columns = df.columns)
  
plt.figure(figsize =(14, 6)) 
  
sns.heatmap(df_comp) 


# Let's try out training this new dataset into algorithm

# In[ ]:


test = df.copy()
test = test["RainTomorrow"].values


# In[ ]:


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(pca, test, test_size = 0.25) 


# ## XGBoost
# XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.

# In[ ]:


import xgboost as xgb 
xgb = xgb.XGBClassifier() 
xgb.fit(X_train, y_train) 
y_pred = xgb.predict(X_test) 


# **Predictions**

# In[ ]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f" % (accuracy * 100.0))


# RMSE & Accuracy Score is pretty well !!. Hence PCA can be used whenever we have more number of dimensions or we can say features
