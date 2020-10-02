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


import numpy as np
import pandas as pd
data=pd.read_csv('/kaggle/input/weather-dataset/weatherHistory.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# # Droping UnWanted Columns

# In[ ]:


data['Loud Cover'].value_counts()


# In[ ]:


#showing zeros for all the rows show drop the column
data.drop('Loud Cover',axis=1,inplace=True)


# In[ ]:


data


# In[ ]:


#Remove "Formatted Date" column as it is not neccesary and remove "Daily Summary" as "Summary" exists

data.drop(['Formatted Date','Daily Summary'],axis=1,inplace=True)

#get the feature value of Wind Bearing (degrees) almost equal to 0
data.drop(['Wind Bearing (degrees)'],axis=1,inplace=True)


# # **Final shape**

# In[ ]:


data.shape


# # Checking for NULL values

# In[ ]:


data.isnull().sum()


# In[ ]:


data['Precip Type'].value_counts()


# **FILLING NULL VALUES**

# In[ ]:


data['Precip Type'].fillna(method='ffill',inplace=True,axis=0)
data['Precip Type'].value_counts()

#with droping
data.drop('Precip Type',axis=1,inplace=True)


# In[ ]:


#Storing target values
target_values=data['Summary'].value_counts().index
target_values


# # Converting Categorical attributes into numerical

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#data['Precip Type']=le.fit_transform(data['Precip Type'])
data['Summary']=le.fit_transform(data['Summary'])


# In[ ]:


data.head()


# # Dividing the data frame into dependent and independent variables

# In[ ]:


y=data.iloc[:,0]
x=data.iloc[:,1:]


# In[ ]:


x


# In[ ]:


y


# #  checking the correlation between variables
# 

# In[ ]:


x.corr()


# In[ ]:


#  correlation between Temperature and Apparent Temparature is almost equal to 1 =======> so drop "Apparent Temparature"

x.drop('Apparent Temperature (C)',axis=1,inplace=True)
x.shape


# In[ ]:


x_cols=x.columns

data


# # Splitting the dataset

# In[ ]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


#  # Using PCA

# from sklearn.decomposition import PCA
# 
# pca=PCA(n_components=1)
# 
# x_train=pca.fit_transform(x_train)
# x_test=pca.transform(x_test)
# 
# 
# 
# 
# #tried with changing components of pca but not increased
# 

# In[ ]:





# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# # Normalizing the dataset  (to give equal importance to each attribute(Independent Variable))

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[ ]:


x_train=pd.DataFrame(x_train,columns=x_cols)
x_test=pd.DataFrame(x_test,columns=x_cols)

x_train


# # Classifying using Navie Bayeas

# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

#training the model
nb.fit(x_train,y_train)

#testing the model
y_pred=nb.predict(x_test)


# In[ ]:


from sklearn import metrics

metrics.accuracy_score(y_test,y_pred)


# # Using RANDOM FOREST CLASSIFIER

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(max_depth=32,n_estimators=120,random_state=1)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)


# In[ ]:


metrics.accuracy_score(y_test,y_pred)


# In[ ]:


rf.feature_importances_


# In[ ]:


x_cols


# 

# In[ ]:




