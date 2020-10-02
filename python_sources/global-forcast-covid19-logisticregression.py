#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#For Kaggle

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#Read Data

df_train=pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
df_test=pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
df_sub=pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")


# ### EDA Train Data

# In[ ]:


df_train.head()


# In[ ]:


print(f"Unique Countries: {len(df_train.Country_Region.unique())}")
train_dates=list(df_train.Date.unique())
print(f"Period : {len(df_train.Date.unique())} days")
print(f"From : {df_train.Date.min()} To : {df_train.Date.max()}")
print(f"Unique Regions: {df_train.shape[0]/len(df_train.Date.unique())}")


# In[ ]:


df_train.Country_Region.value_counts()


# In[ ]:


print(f"Number of rows without Country_Region : {df_train.Country_Region.isna().sum()}")
#New Column UniqueRegion combining Province_State and Country_Region
df_train["UniqueRegion"]=df_train.Country_Region
df_train.UniqueRegion[df_train.Province_State.isna()==False]=df_train.Province_State+" , "+df_train.Country_Region
df_train[df_train.Province_State.isna()==False]


# In[ ]:


df_train.drop(labels=["Id","Province_State","Country_Region"], axis=1, inplace=True)
df_train


# ### EDA Test Data

# In[ ]:


df_test.head()


# In[ ]:


test_dates=list(df_test.Date.unique())
print(f"Period :{len(df_test.Date.unique())} days")
print(f"From : {df_test.Date.min()} To : {df_test.Date.max()}")
print(f"Total Regions : {df_test.shape[0]/43}")


# Total regions in test is same as train data

# In[ ]:


df_test["UniqueRegion"]=df_test.Country_Region
df_test.UniqueRegion[df_test.Province_State.isna()==False]=df_test.Province_State+" , "+df_test.Country_Region
df_test.drop(labels=["Province_State","Country_Region"], axis=1, inplace=True)
len(df_test.UniqueRegion.unique())


# ### EDA Submission data

# In[ ]:


df_sub.head()


# In[ ]:


# Dates in train only
only_train_dates=set(train_dates)-set(test_dates)
print("Only train dates : ",len(only_train_dates))
#dates in train and test
intersection_dates=set(test_dates)&set(train_dates)
print("Intersection dates : ",len(intersection_dates))
#dates in only test
only_test_dates=set(test_dates)-set(train_dates)
print("Only Test dates : ",len(only_test_dates))


# ### Predict cases

# In[ ]:


#Duplicate dataframe for test data with new column Delta
df_test_temp=pd.DataFrame()
df_test_temp["Date"]=df_test.Date
df_test_temp["ConfirmedCases"]=0.0
df_test_temp["Fatalities"]=0.0
df_test_temp["UniqueRegion"]=df_test.UniqueRegion
df_test_temp["Delta"]=1.0


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import random\nfinal_df=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","UniqueRegion"])\n\nfor region in df_train.UniqueRegion.unique():\n    df_temp=df_train[df_train.UniqueRegion==region].reset_index()\n    df_temp["Delta"]=1.0\n    size_train=df_temp.shape[0]\n    for i in range(1,df_temp.shape[0]):\n        if(df_temp.ConfirmedCases[i-1]>0):\n            df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]\n\n    #number of days for delta trend\n    n=4     \n\n    #delta as average of previous n days\n    delta_avg=df_temp.tail(n).Delta.mean()\n\n    #delta as trend for previous n days\n    delta_list=df_temp.tail(n).Delta\n\n    #Morality rate as on last availabe date\n    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()\n\n    df_test_app=df_test_temp[df_test_temp.UniqueRegion==region]\n    df_test_app=df_test_app[df_test_app.Date>df_temp.Date.max()]\n\n    X=np.arange(1,n+1).reshape(-1,1)\n    Y=delta_list\n    model=LinearRegression()\n    model.fit(X,Y)\n\n    df_temp=pd.concat([df_temp,df_test_app])\n    df_temp=df_temp.reset_index()\n\n    for i in range (size_train, df_temp.shape[0]):\n        n=n+1\n        d=df_temp.Delta[i-1]*0.5+df_temp.Delta[i-5]*0.3*df_temp.Delta[i-10]*0.2\n        m=model.predict(np.array([n]).reshape(-1,1))[0]\n        choice=[m,d]\n        df_temp.Delta[i]=max(1,random.choice(choice))\n        df_temp.ConfirmedCases[i]=round(df_temp.ConfirmedCases[i-1]*df_temp.Delta[i],0)\n        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)\n\n\n    size_test=df_temp.shape[0]-df_test_temp[df_test_temp.UniqueRegion==region].shape[0]\n\n    df_temp=df_temp.iloc[size_test:,:]\n    \n    df_temp=df_temp[["Date","ConfirmedCases","Fatalities","UniqueRegion"]]\n    final_df=pd.concat([final_df,df_temp], ignore_index=True)\n    \nfinal_df.shape')


# In[ ]:


df_sub.Fatalities=final_df.Fatalities
df_sub.ConfirmedCases=final_df.ConfirmedCases
df_sub.to_csv("submission.csv", index=None)


# In[ ]:




