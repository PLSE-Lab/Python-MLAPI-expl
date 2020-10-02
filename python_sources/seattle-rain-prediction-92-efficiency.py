#!/usr/bin/env python
# coding: utf-8

# ##  Seattle Rain Prediction

# **This notebook describes how you can apply Machine Learning techniques to predict the rain. In this example, I used the seattle rain data and briefly explained how I achieved 92% efficiency using classfication algorithm -Logistic Regression. ** 
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import datetime


# In[ ]:


df_seattle=pd.read_csv("../input/seattleWeather_1948-2017.csv")


# In[ ]:


df_seattle.info()


# In[ ]:


df_seattle.head()


# In[ ]:


df_seattle.describe()


# **From the above results, we have seen that there are missing data in the dataset. Further, to check, I have drawn a heatmap which shows there are missing values in PRCP and RAIN column **

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(pd.isnull(df_seattle),yticklabels=False)


# **Now, to get the more specific detail in the missing column, there are three rows which have null values**

# In[ ]:


df_seattle[pd.isnull(df_seattle['PRCP'])]


# In[ ]:


df_seattle[pd.isnull(df_seattle['RAIN'])]


# **From the above code, we got to know that there is missing value for PRCP and RAIN column for 9/5/2005 date**

# In[ ]:


sns.countplot(data=df_seattle, x='RAIN')


# **From the above figure we can see that there are less chances of Rain. So in the missing data I am simply inserting "False"**

# In[ ]:


df_seattle['PRCP'].mean()


# **Instead of dropping one row, better to insert mean value in PRCP column**

# In[ ]:


def RAIN_INSERTION(cols):
    RAIN=cols[0]
    if pd.isnull(RAIN):
        return 'False'
    else:
        return RAIN


# In[ ]:


def PRCP_INSERTION(col):
    PRCP=col[0]
    if pd.isnull(PRCP):
        return df_seattle['PRCP'].mean()
    else:
        return PRCP


# In[ ]:


df_seattle['RAIN']=df_seattle[['RAIN']].apply(RAIN_INSERTION,axis=1)


# In[ ]:


df_seattle['PRCP']=df_seattle[['PRCP']].apply(PRCP_INSERTION,axis=1)


# **To verify if the function worked or not**

# In[ ]:


df_seattle[pd.isnull(df_seattle['RAIN'])]


# In[ ]:


df_seattle[pd.isnull(df_seattle['PRCP'])]


# ## Exploratory Data Analysis

# In[ ]:


plt.figure(figsize=(7,7))
plt.scatter(x='TMIN',y='PRCP',data=df_seattle)
plt.xlabel('Minimum Temperature')
plt.ylabel('PRCP')
plt.title('Precipitation Vs Minimum Temperature')


# In[ ]:


plt.figure(figsize=(7,7))
plt.scatter(x='TMAX',y='PRCP',data=df_seattle)
plt.xlabel('Maximum Temperature')
plt.ylabel('PRCP')
plt.title('Precipitation Vs Maximum Temperature')


# In[ ]:


sns.distplot(df_seattle['TMIN'])


# In[ ]:


sns.distplot(df_seattle['TMAX'])


# In[ ]:


sns.pairplot(data=df_seattle)


# In[ ]:


#plt.figure(figsize=(10,7))
sns.boxplot(data=df_seattle)


# **From the above figure, we can say that there are some outliers.**

# In[ ]:


#Dropping the outliers from TMIN column
df_seattle=df_seattle.drop(df_seattle[df_seattle['TMIN']<17 ].index)


# In[ ]:


#Dropping the outliers from TMAX columns i.e. the value more than 100
df_seattle=df_seattle.drop(df_seattle[(df_seattle['TMAX']>97.5) | (df_seattle['TMAX']< 21.5)].index)


# In[ ]:


#Dropping the outliers from PRCP columns i.e. the value more than 0.275
df_seattle=df_seattle.drop(df_seattle[(df_seattle['PRCP']>0.25) | (df_seattle['PRCP']< -0.15) ].index)


# **To check whether the outliers are removed or  not.**

# In[ ]:


sns.boxplot(data=df_seattle)


# **Since, this is a classfication problem, we can apply logistic regression to predict the rain in Seattle**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr= LogisticRegression()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X=df_seattle.drop(['RAIN','DATE'],axis=1)
y=df_seattle['RAIN']
y=y.astype('str')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


prediction=lr.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print('Confusion Matrix',confusion_matrix(y_test,prediction))
print('\n')
print('Classification Report',classification_report(y_test,prediction))


# ![](http://)**Result - 92% efficiency**
