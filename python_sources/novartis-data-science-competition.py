#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Novartis Data Science Competition

# [Novartis Data Science Competition](https://www.hackerearth.com/challenges/hiring/novartis-data-science-hiring-challenge/)

# To Predict if the server will be hacked or not

# ## Load the data from Github

# In[ ]:


import numpy as np
import pandas as pd
df_Test_url = 'https://raw.githubusercontent.com/blessondensil294/Novartis-Data-Science-Competition/master/Data/Test.csv'
df_Train_url = 'https://raw.githubusercontent.com/blessondensil294/Novartis-Data-Science-Competition/master/Data/Train.csv'
df_Train = pd.read_csv(df_Train_url)
df_Test = pd.read_csv(df_Test_url)


# ## Exploratory Data Analysis

# To the Train Head of the Dataframe

# In[ ]:


df_Train.head()


# DataFrame Information

# In[ ]:


df_Train.info()


# Description of the DataFrame

# In[ ]:


df_Train.describe()


# Shape of the DataFrame of both Test and Train

# In[ ]:


df_Train.shape


# In[ ]:


df_Test.shape


# To find the Null cells in the DataFrame for each columns

# In[ ]:


df_Train.isnull().sum()


# In[ ]:


df_Test.isnull().sum()


# Column Names of the DataFrame

# In[ ]:


df_Train.columns


# Total count of the Predictor Variable

# In[ ]:


df_Train['MULTIPLE_OFFENSE'].value_counts()


# ## Feature Engineer

# ### Remove Duplicate Rows in only the Train DataFrame

# Shape of the Train DataFrame before removing the duplicates

# In[ ]:


df_Train.shape


# Drop the columns INCIDENT_ID and DATE. This is to ensure to avoid Unique rows in the columns

# In[ ]:


df_Train = df_Train.drop(['INCIDENT_ID', 'DATE'], axis=1)


# Drop the Duplicate Rows

# In[ ]:


df_Train.drop_duplicates(keep='first', inplace=True)


# Shape of the Train DataFrame After removing the duplicates - About 5042 records are removed

# In[ ]:


df_Train.shape


# Count of the multiple Offence column after removing the Duplicates

# In[ ]:


df_Train['MULTIPLE_OFFENSE'].value_counts()


# Count of the Null column after removing the Duplicates

# In[ ]:


df_Train.isnull().sum()


# ### Fill Missing Values

# In[ ]:


df_Train['X_12'] = df_Train['X_12'].ffill()
df_Test['X_12'] = df_Test['X_12'].ffill()
df_Train['X_12'] = df_Train['X_12'].bfill()
df_Test['X_12'] = df_Test['X_12'].bfill()


# In[ ]:


df_Train.isnull().sum()


# Convert the X_12 from Float64 to Integer64 

# In[ ]:


#Convert to integer
df_Train['X_12'] = df_Train['X_12'].astype(int)
df_Test['X_12'] = df_Test['X_12'].astype(int)


# In[ ]:


df_Train.info()


# ## Feature Selection

# ### Correlation of Data

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in df_Train.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = df_Train.columns.difference(str_list) 
# Create Dataframe containing only numerical features
train_num = df_Train[num_list]
f, ax = plt.subplots(figsize=(25, 25))
plt.title('Pearson Correlation of features')
# Draw the heatmap using seaborn
sns.heatmap(train_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)


# In[ ]:


df_Train.corr()


# In[ ]:


corrmat = df_Train.corr()
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corrmat, square=True, vmax=.8)


# ### Multi Colinearity of the Data

# In[ ]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


df_multi = df_Train
df_multi = df_multi.drop('MULTIPLE_OFFENSE', axis=1)


# In[ ]:


# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_multi.values, i) for i in range(df_multi.shape[1])]
vif["features"] = df_multi.columns


# In[ ]:


vif.round(1)


# ## Data Modelling for Prediction

# ### Split the Data to x variable and y variable

# In[ ]:


x = df_Train
x = x.drop(['MULTIPLE_OFFENSE'], axis=1)
y = df_Train['MULTIPLE_OFFENSE']
x_pred = df_Test
x_pred = x_pred.drop(['INCIDENT_ID', 'DATE'], axis=1)


# ### Balancing the Train Data using SMOTE Overbalancing

# In[ ]:


from imblearn.over_sampling import RandomOverSampler
sm = RandomOverSampler(random_state=294,sampling_strategy='not majority')
x_sm, y_sm = sm.fit_resample(x,y)
x_sm = pd.DataFrame(x_sm)
x_sm.columns = x.columns


# ## Random Forest Model Classification

# We will use Entropy with 1000 trees to build the models

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, criterion='entropy', n_jobs=-1, random_state=294)


# In[ ]:


rf.fit(x_sm,np.ravel(y_sm))


# In[ ]:


y_pred = rf.predict(x_pred)


# In[ ]:


submission_df = pd.DataFrame({'INCIDENT_ID':df_Test['INCIDENT_ID'], 'MULTIPLE_OFFENSE':y_pred})
submission_df.to_csv('Sample Submission RF v1.csv', index=False)


# Random Forest gave a score of 96.25 Accuracy in the Evaluation.

# ## CatBoost Classification

# In[ ]:


pip install catboost


# In[ ]:


from catboost import CatBoostClassifier
cb_cl = CatBoostClassifier(learning_rate=0.15, n_estimators=500, subsample=0.70, max_depth=5, scale_pos_weight=2.5)


# In[ ]:


cb_cl.fit(x_sm,np.ravel(y_sm))


# In[ ]:


y_pred = cb_cl.predict(x_pred)


# In[ ]:


submission_df = pd.DataFrame({'INCIDENT_ID':df_Test['INCIDENT_ID'], 'MULTIPLE_OFFENSE':y_pred})
submission_df.to_csv('Sample Submission CB v1.csv', index=False)


# CatBoost Classification gave a score of 99.42 Accuracy in the Evaluation.

# ### This is mainly for a practice Notebook for the competition. You can try with various parameters and Boosting algorithm to improve your score to 100%
