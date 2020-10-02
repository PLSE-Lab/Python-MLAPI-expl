#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df_train=pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
df_test=pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
df_sub=pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

print(df_train.shape)
print(df_test.shape)
print(df_sub.shape)


# ### EDA Train Data

# In[ ]:


df_train.head()


# In[ ]:


print(f"Unique Countries: {len(df_train.Country_Region.unique())}")


# In[ ]:


train_dates=list(df_train.Date.unique())
print(f"Period : {len(df_train.Date.unique())} days")
print(f"From : {df_train.Date.min()} To : {df_train.Date.max()}")


# In[ ]:


print(f"Unique Regions: {df_train.shape[0]/len(df_train.Date.unique())}")


# In[ ]:


df_train.Country_Region.value_counts()


# In[ ]:


print(f"Number of rows without Country_Region : {df_train.Country_Region.isna().sum()}")


# In[ ]:


df_train["UniqueRegion"]=df_train.Country_Region
df_train.UniqueRegion[df_train.Province_State.isna()==False]=df_train.Province_State+" , "+df_train.Country_Region
df_train[df_train.Province_State.isna()==False]


# In[ ]:


df_train.drop(labels=["Id","Province_State","Country_Region"], axis=1, inplace=True)


# In[ ]:


df_train


# ### EDA Test Data

# In[ ]:


df_test.head()


# In[ ]:


test_dates=list(df_test.Date.unique())
print(f"Period :{len(df_test.Date.unique())} days")
print(f"From : {df_test.Date.min()} To : {df_test.Date.max()}")


# In[ ]:


print(f"Total Regions : {df_test.shape[0]/43}")


# Total regions in test is same as train data

# In[ ]:


df_test["UniqueRegion"]=df_test.Country_Region
df_test.UniqueRegion[df_test.Province_State.isna()==False]=df_test.Province_State+" , "+df_test.Country_Region
df_test.drop(labels=["Province_State","Country_Region"], axis=1, inplace=True)


# In[ ]:


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


import random
df_test_temp=pd.DataFrame()
df_test_temp["Date"]=df_test.Date
df_test_temp["ConfirmedCases"]=0.0
df_test_temp["Fatalities"]=0.0
df_test_temp["UniqueRegion"]=df_test.UniqueRegion
df_test_temp["Delta"]=1.0


# In[ ]:


get_ipython().run_cell_magic('time', '', 'final_df=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","UniqueRegion"])\n\nfor region in df_train.UniqueRegion.unique():\n    df_temp=df_train[df_train.UniqueRegion==region].reset_index()\n    df_temp["Delta"]=1.0\n    size_train=df_temp.shape[0]\n    for i in range(1,df_temp.shape[0]):\n        if(df_temp.ConfirmedCases[i-1]>0):\n            df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]\n\n    #number of days for delta trend\n    n=7    \n\n    #delta as trend for previous n days\n    delta_list=df_temp.tail(n).Delta\n    \n    #Average Growth Factor\n    delta_avg=df_temp.tail(n).Delta.mean()\n\n    #Morality rate as on last availabe date\n    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()\n\n    df_test_app=df_test_temp[df_test_temp.UniqueRegion==region]\n    df_test_app=df_test_app[df_test_app.Date>df_temp.Date.max()]\n\n    X=np.arange(1,n+1).reshape(-1,1)\n    Y=delta_list\n    model=LinearRegression()\n    model.fit(X,Y)\n    #score_pred.append(model.score(X,Y))\n    #reg_pred.append(region)\n\n    df_temp=pd.concat([df_temp,df_test_app])\n    df_temp=df_temp.reset_index()\n\n    for i in range (size_train, df_temp.shape[0]):\n        n=n+1        \n        damper=df_temp.Delta[i-5]\n        pred=max(1,model.predict(np.array([n]).reshape(-1,1))[0])\n        \n        df_temp.Delta[i]=(damper+pred+delta_avg)/3\n        #df_temp.Delta[i]=pred\n        \n    for i in range (size_train, df_temp.shape[0]):\n        df_temp.ConfirmedCases[i]=round(df_temp.ConfirmedCases[i-1]*df_temp.Delta[i],0)\n        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)\n\n\n    size_test=df_temp.shape[0]-df_test_temp[df_test_temp.UniqueRegion==region].shape[0]\n\n    df_temp=df_temp.iloc[size_test:,:]\n    \n    df_temp=df_temp[["Date","ConfirmedCases","Fatalities","UniqueRegion","Delta"]]\n    final_df=pd.concat([final_df,df_temp], ignore_index=True)\n\n#df_score=pd.DataFrame({"Region":reg_pred,"Score":score_pred})\n#print(f"Average score (n={n}): {df_score.Score.mean()}")\n#sns.distplot(df_score.Score)    \nfinal_df.shape')


# In[ ]:


df_sub.shape


# In[ ]:


df_sub.Fatalities=final_df.Fatalities
df_sub.ConfirmedCases=final_df.ConfirmedCases
df_sub.to_csv("submission.csv", index=None)


# In[ ]:




