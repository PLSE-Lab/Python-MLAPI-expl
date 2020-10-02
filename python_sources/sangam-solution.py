#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_train=pd.read_csv("../input/Train.csv")
df_test=pd.read_csv("../input/Test.csv")
df_sample=pd.read_csv("../input/sample_submission.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_train.shape


# In[ ]:


df_train.dtypes


# In[ ]:


df_test.head()


# In[ ]:


df_test.shape


# In[ ]:


df_sample.head()


# In[ ]:


df_sample.shape


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_train.head()


# In[ ]:


del df_train['date_time']


# In[ ]:


submit = pd.DataFrame(columns=['date_time','traffic_volume'])


# In[ ]:


submit_time=df_test['date_time']


# In[ ]:


submit['date_time']=submit_time


# In[ ]:


submit.head()


# In[ ]:


del df_test['date_time']


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


y=df_train['traffic_volume']


# In[ ]:


del df_train['traffic_volume']


# In[ ]:


df_train.dtypes


# In[ ]:


df_train['is_holiday'].unique()


# In[ ]:


df_train['weather_type'].unique()


# In[ ]:


df_train['weather_description'].unique()


# In[ ]:


plt.matshow(df_train.corr())
plt.show()


# In[ ]:


corr = df_train.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


is_holiday=pd.factorize(df_train['is_holiday'])[0]
is_holiday1=pd.factorize(df_test['is_holiday'])[0]
weather_type=pd.factorize(df_train['weather_type'])[0]
weather_type1=pd.factorize(df_test['weather_type'])[0]
weather_description=pd.factorize(df_train['weather_description'])[0]
weather_description1=pd.factorize(df_test['weather_description'])[0]


# In[ ]:


temp_train = pd.DataFrame({'is_holiday':is_holiday , 'weather_type':weather_type , 'weather_description':weather_description})


# In[ ]:


temp_test = pd.DataFrame({'is_holiday1':is_holiday1 , 'weather_type1':weather_type1 , 'weather_description1':weather_description1})


# In[ ]:


df_train=pd.concat([df_train,temp_train],axis=1)
df_test=pd.concat([df_test,temp_test],axis=1)


# In[ ]:


df_train.drop(['is_holiday','weather_type','weather_description'],axis=1,inplace=True)


# In[ ]:


df_test.drop(['is_holiday1','weather_type1','weather_description1','is_holiday','weather_type','weather_description'],axis=1,inplace=True)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


clf = RandomForestRegressor(n_jobs=2, random_state=0)


# In[ ]:


clf.fit(df_train, y)


# In[ ]:


preds=clf.predict(df_test)


# In[ ]:


submit['traffic_volume']=preds


# In[ ]:


submit.to_csv("output.csv",index=False)


# In[ ]:


submit.head()


# In[ ]:


df_train.columns


# In[ ]:


df_test.columns


# In[ ]:




