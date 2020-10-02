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


# In[ ]:


#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

#Machine Learning
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")
train.head()


# In[ ]:


test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")
test.head()


# Understanding and Cleaning the Data

# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


train.dtypes


# In[ ]:


train.describe()


# Data Discription for String Columns

# In[ ]:


train.describe(include=[np.object])


# # Handeling Null Values

# In[ ]:


train.isna().any()


# In[ ]:


train.isna().sum()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Droping of Unwanted Data to make model more Predictive

# In[ ]:


for df in [train,test]:
    df.drop("County",axis=1,inplace=True)
    df.drop("Province_State",axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Data Visualization

# # Performing Correlation Matrix for Train Data

# In[ ]:


fig = px.pie(train, values='TargetValue', names='Target', title='ConfirmedCases & Fatalities')
fig.show()


# In[ ]:


fig = px.pie(train, values='TargetValue', names='Country_Region', title='ConfirmedCases & Fatalities Percentile by Country')
fig.update_traces(textposition='inside')
fig.show()


# # Preparing data for Model

# Converting String Date into Integer for both Train and Test Datasets

# In[ ]:


train["Date"] = pd.to_datetime(train["Date"]).dt.strftime("%m%d").astype(int)


# In[ ]:


test["Date"] = pd.to_datetime(test["Date"]).dt.strftime("%m%d").astype(int)


# Appling Label Encoding for Categorial features  

# In[ ]:


from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
train['Country_Region']= le.fit_transform(train['Country_Region']) 
train['Target']= le.fit_transform(train['Target']) 
test['Country_Region']= le.fit_transform(test['Country_Region']) 
test['Target']= le.fit_transform(test['Target']) 


# In[ ]:


train.tail()


# In[ ]:


test.head()


# Slipting Data based on Predictors and Target values

# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['TargetValue', 'Id'], axis=1)
target = train["TargetValue"]
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state =0)


# Fitting the model RandomForestRegressor 

# In[ ]:


model = RandomForestRegressor(n_jobs=-1)
scores = []
model.set_params(n_estimators=100)
model.fit(X_train, y_train)
scores.append(model.score(X_test, y_test))
score = model.score(X_test, y_test)
print(score*100)


# In[ ]:


test.drop(['ForecastId'],axis=1,inplace=True)
test.index.name = 'Id'
test.head()


# # Prediction

# In[ ]:


y_pred = model.predict(X_test)
y_pred


# # Output

# In[ ]:


predictions = model.predict(test)

pred_list = [int(x) for x in predictions]

output = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})
print(output)


# # Preparing Submission File

# In[ ]:


q05 = output.groupby('Id')['TargetValue'].quantile(q=0.05).reset_index()
q50 = output.groupby('Id')['TargetValue'].quantile(q=0.5).reset_index()
q95 = output.groupby('Id')['TargetValue'].quantile(q=0.95).reset_index()

q05.columns=['Id','0.05']
q50.columns=['Id','0.5']
q95.columns=['Id','0.95']


# In[ ]:


concatDF = pd.concat([q05,q50['0.5'],q95['0.95']],1)
concatDF['Id'] = concatDF['Id'] + 1
concatDF.head(10)


# # Submission

# In[ ]:


sub = pd.melt(concatDF, id_vars=['Id'], value_vars=['0.05','0.5','0.95'])
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head(10)

