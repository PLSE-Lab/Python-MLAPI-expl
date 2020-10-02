#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder

#import matplotlib for visualization
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train=pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
df_test=pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
df_sub=pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

print(df_train.shape)
print(df_test.shape)
print(df_sub.shape)


# # EDA Train Data

# In[ ]:


df_train.shape


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# In[ ]:


df_train.Province_State.isna().sum()


# In[ ]:


len(df_train[df_train.Country_Region.isna()==False])


# In[ ]:


df_train.Country_Region.value_counts()


# In[ ]:


df_train[df_train.Province_State.isna()==False]


# In[ ]:


df_train['Region']=df_train['Country_Region']


# In[ ]:


df_train.Region[df_train.Province_State.isna()==False]=df_train['Province_State']+','+df_train['Country_Region']


# In[ ]:


df_train


# In[ ]:


len(df_train.Date.unique())


# In[ ]:


df_train.Date.min()


# In[ ]:


df_train.Date.max()


# In[ ]:


df_train.drop(labels=['Id','Province_State','Country_Region'], axis=1, inplace=True)


# In[ ]:


df_train


# # EDA Test Data

# In[ ]:


df_test


# In[ ]:


df_test.shape


# In[ ]:


df_test.info()


# In[ ]:


df_test.describe()


# In[ ]:


df_test.Date.min()


# In[ ]:


df_test.Date.max()


# In[ ]:


len(df_test.Date.unique())


# In[ ]:


df_test.shape[0]


# In[ ]:


df_test.Country_Region.value_counts()


# In[ ]:


df_test['Region']=df_test['Country_Region']


# In[ ]:


df_test.head()


# In[ ]:


df_test.Region[df_test.Province_State.isna()==False]=df_test.Province_State+','+df_test.Country_Region


# In[ ]:


df_test


# In[ ]:


len(df_test.Region.unique())


# In[ ]:


df_test.drop(labels=['ForecastId','Province_State','Country_Region'],axis=1,inplace=True)


# In[ ]:


len(df_test.Region.unique())


# In[ ]:


len(df_test.Date.unique())


# # EDA sub

# In[ ]:


df_sub


# In[ ]:


df_sub.info()


# In[ ]:


df_sub.describe()


# In[ ]:


df_sub.shape


# In[ ]:


train_dates=list(df_train.Date.unique())


# In[ ]:


test_dates=list(df_test.Date.unique())


# In[ ]:


only_train_dates=set(train_dates)-set(test_dates)


# In[ ]:


len(only_train_dates)


# In[ ]:


intersection_dates=set(train_dates)&set(test_dates)


# In[ ]:


len(intersection_dates)


# In[ ]:


only_test_dates=set(test_dates)-set(train_dates)


# In[ ]:


len(only_test_dates)


# # Predict

# In[ ]:


import random
df_test_temp=pd.DataFrame()
df_test_temp["Date"]=df_test.Date
df_test_temp["ConfirmedCases"]=0.0
df_test_temp["Fatalities"]=0.0
df_test_temp["Region"]=df_test.Region
df_test_temp["Delta"]=1.0


# In[ ]:


df_test_temp


# In[ ]:


get_ipython().run_cell_magic('time', '', 'final_df=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","Region"])')


# In[ ]:


final_df


# In[ ]:


for region in df_train.Region.unique():
    df_temp=df_train[df_train.Region==region].reset_index()
    df_temp["Delta"]=1.0
    size_train=df_temp.shape[0]
    for i in range(1,df_temp.shape[0]):
        if(df_temp.ConfirmedCases[i-1]>0):
            df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]


# In[ ]:



    #number of days for delta trend
    n=5     

    #delta as trend for previous n days
    delta_list=df_temp.tail(n).Delta
    
    #Average Growth Factor
    delta_avg=df_temp.tail(n).Delta.mean()

    #Morality rate as on last availabe date
    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()

    df_test_app=df_test_temp[df_test_temp.Region==region]
    df_test_app=df_test_app[df_test_app.Date>df_temp.Date.max()]

    X=np.arange(1,n+1).reshape(-1,1)
    Y=delta_list
    model=LinearRegression()
    model.fit(X,Y)
    #score_pred.append(model.score(X,Y))
    #reg_pred.append(region)

    df_temp=pd.concat([df_temp,df_test_app])
    df_temp=df_temp.reset_index()

    for i in range (size_train, df_temp.shape[0]):
        n=n+1        
        df_temp.Delta[i]=(df_temp.Delta[i-3]+max(1,model.predict(np.array([n]).reshape(-1,1))[0])+delta_avg)/3
        
    for i in range (size_train, df_temp.shape[0]):
        df_temp.ConfirmedCases[i]=round(df_temp.ConfirmedCases[i-1]*df_temp.Delta[i],0)
        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)


    size_test=df_temp.shape[0]-df_test_temp[df_test_temp.Region==region].shape[0]

    df_temp=df_temp.iloc[size_test:,:]
    
    df_temp=df_temp[["Date","ConfirmedCases","Fatalities","Region","Delta"]]
    final_df=pd.concat([final_df,df_temp], ignore_index=True)

#df_score=pd.DataFrame({"Region":reg_pred,"Score":score_pred})
#print(f"Average score (n={n}): {df_score.Score.mean()}")
#sns.distplot(df_score.Score)    
final_df.shape


# In[ ]:


df_sub.Fatalities=final_df.Fatalities
df_sub.ConfirmedCases=final_df.ConfirmedCases
df_sub.to_csv("submission.csv", index=None)

