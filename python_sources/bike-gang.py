#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import datetime
import matplotlib.pyplot as plt

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification
 
import seaborn as sns
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


sample_df = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')
test_df = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")


# In[ ]:


df.shape


# In[ ]:


df.head()


# **Exploratory Data Analysis of all our features **

# In[ ]:


df.dtypes


# In[ ]:


print(df['season'].value_counts())
sns.distplot(a= df['season'], kde = False)


# In[ ]:


sns.barplot(data =df, x='season', y = 'count')


# Quite clear that the data is equally distributed across all seasons.
# Also, that season does have a effect on bike sales, bike sales are especially low during "1".
# 

# In[ ]:


sns.lineplot(data =df, x='holiday', y = 'count')
print()


# In[ ]:


sns.barplot(data =df, x='holiday', y = 'count')
print(df['holiday'].value_counts())


# Holiday doesnt seem to have that much contribution towards the overall. (both 0 and 1 contrbute almost same)

# In[ ]:


df['weather'].value_counts()


# In[ ]:


sns.lineplot(data =df, x='weather', y = 'count')


# In[ ]:


sns.barplot(data =df, x='weather', y = 'count')


# * Weather data is highly imbalaced. Values ranging from 7k to only 1 data point
# 

# In[ ]:


sns.lineplot(data =df, x='workingday', y = 'count')
print(df['workingday'].value_counts())


# In[ ]:


df['workingday'].value_counts()


# In[ ]:


cont_names=['temp','atemp','humidity','windspeed']

        
#sns.boxplot(train_df['season'])   
i=0
for name in cont_names:
    i=i+1
    plt.subplot(2,2,i)
    sns.boxplot(name,data=df) 
    
plt.show()


# Temp, Atemp and humidity are normally distributed. Although, windspped has a lot of outliers.

# In[ ]:


cat_names=['season', 'holiday', 'workingday', 'weather']

i=0
for name in cat_names:
    i=i+1
    plt.subplot(2,2,i)
    sns.countplot(name, data=df) 
plt.show()


# In[ ]:






#fig, axes = plt.subplots(nrows=1,ncols=1)
#fig.set_size_inches(12, 10)
sns.boxplot(data=df,y="count",x="season",orient="v")
#print(df['count'].describe())

#print(df['count'].skew())


# In[ ]:


sns.boxplot(data=df,y="count",orient="v")


# In[ ]:


print(df[df['season']== 1]['count'].mean())
print(df[df['season']== 2]['count'].mean())
print(df[df['season']== 3]['count'].mean())
print(df[df['season']== 4]['count'].mean())


# **Checking for missing data
# (No data is missing)**

# In[ ]:


df.isnull().sum()


# In[ ]:


df["date"] = df.datetime.apply(lambda x : x.split()[0])
df['date']


# In[ ]:


df["hour"] = df.datetime.apply(lambda x : x.split()[1].split(":")[0])


# This is one way of converting your datetime feature into valueable feature, but i found much faster way

# Lets convert our date feature to date time and do it more sunnictly.
# 

# In[ ]:


df["datetime"] = pd.to_datetime(df["datetime"])
df["year"]=df["datetime"].dt.year
df["month"]=df["datetime"].dt.month
df["day"]=df["datetime"].dt.day
df["dayofweek"]=df["datetime"].dt.dayofweek


# In[ ]:


test_df["datetime"] = pd.to_datetime(test_df["datetime"])
test_df["year"]=test_df["datetime"].dt.year
test_df["month"]=test_df["datetime"].dt.month
test_df["day"]=test_df["datetime"].dt.day
test_df["dayofweek"]=test_df["datetime"].dt.dayofweek
test_df["hour"]=test_df["datetime"].dt.hour


# Ideally, I should have concatenated both the dataframes and done all manipluations at the same time. 

# In[ ]:


df.corr()


# > *Co-relation is a very useful function. As, it gives a very clear quantative answer about features contribute to the target label and how by much*
# 

# In[ ]:


sns.heatmap(df.corr(), annot=True)


# Visual representation of correlation using a heatmap.

# #  **DATA LEAKAGE**

# **'casual' and 'resgistered'** both show very high correlation with our target variable 'count', including them would result in a very accurate model when applying it on training and validation data. 
# But will perform very poorly, when tested on new data as they wont have data about 'casual' and 'resgistered' as that can only be known after someone has rented the bike

# In[ ]:


df.drop(['casual','registered'], axis =1, inplace = True)


# In[ ]:


sns.barplot(x="hour",y="count",data=df)
print(df['hour'].value_counts())


# In[ ]:


sns.barplot(x="dayofweek",y="count",data=df)


# In[ ]:


sns.barplot(x="month",y="count",data=df)


# These graphs show  how our datetime features that we manipluated might contribute towards the target variable. 

#  ## Modelling

# In[ ]:


from sklearn.model_selection import train_test_split
X = df.drop(['count', 'datetime', 'atemp','date'], axis=1)
y = df['count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


X_train


# In[ ]:




models=[RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),SVR(),KNeighborsRegressor()]
model_names=['RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','SVR','KNeighborsRegressor']
rmsle=[]
d={}
for model in range (len(models)):
    clf=models[model]
    clf.fit(X_train,y_train)
    test_pred=clf.predict(X_test)
    rmsle.append(np.sqrt(mean_squared_log_error(test_pred,y_test)))
d={'Modelling Algo':model_names,'RMSLE':rmsle}   
d
    


# Time for some parameter tuning. Going with Random Forest as it performed best.

# In[ ]:


no_of_test=[500]
params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}
clf_rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')
clf_rf.fit(X_train,y_train)
pred=clf_rf.predict(X_test)
print((np.sqrt(mean_squared_log_error(pred,y_test))))


# 

# In[ ]:




clf_rf.best_params_


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error

predictions = clf_rf.predict(X_test)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_test)))
print((np.sqrt(mean_squared_log_error(predictions,y_test))))
print(predictions)


# * Mean Absolute Error: 24.071840220385674
# * mean_squared_log_error = 0.3255321535061589 (The metric used by this competiton)

# In[ ]:


pred=clf_rf.predict(test_df.drop(['datetime', 'atemp'],axis=1))
pred


# In[ ]:


d={'datetime':test_df['datetime'],'count':pred}
ans=pd.DataFrame(d)
ans.to_csv('answer.csv',index=False)


# Submitting results to the competiton
