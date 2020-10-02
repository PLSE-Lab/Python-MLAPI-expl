#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv',index_col=False)
df_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv',index_col=False)


# In[ ]:


print(df_train.shape)
df_test.shape


# In[ ]:


from catboost import CatBoostRegressor


# In[ ]:


df_train=df_train.fillna(0)


# In[ ]:


df_test=df_test.fillna(0)


# In[ ]:


cols=df_train.select_dtypes(include=['object']).columns


# In[ ]:


model=CatBoostRegressor()


# In[ ]:


y=df_train.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df_train.iloc[:,:-1],y)


# In[ ]:


print(x_train.shape)
y_train.shape


# In[ ]:


model.fit(x_train,y_train,cat_features=cols)


# In[ ]:


model.score(x_train,y_train)


# In[ ]:


predicted=model.predict(df_test)


# In[ ]:


predicted


# In[ ]:


df_sample=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


df_test.shape


# In[ ]:


ids=pd.Series(df_test.Id)


# In[ ]:


Saleprice=pd.Series(predicted)


# In[ ]:


ss=pd.concat([ids,Saleprice],axis=1)


# In[ ]:


ss['SalePrice']=ss[0]


# In[ ]:


ss.drop(0,inplace=True,axis=1)


# In[ ]:


ss


# In[ ]:


ss.to_csv('house1.csv')

