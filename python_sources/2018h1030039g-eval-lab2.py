#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df['class'].value_counts()


# In[ ]:


df_1=df[df['class'] == 1]
df_2=df[df['class'] == 2]
df_3=df[df['class'] == 3]
df_4=df[df['class'] == 4]
df_5=df[df['class'] == 5]
df_6 = df[df['class'] == 6]
df_7=df[df['class'] == 7]


# In[ ]:


df_1 = df_1.sample(49, replace=True)
df_2 = df_2.sample(49, replace=True)
df_3 = df_3.sample(49, replace=True)
df_5 = df_5.sample(49, replace=True)
df_6 = df_6.sample(49, replace=True)
df_7 = df_7.sample(49, replace=True)


# In[ ]:


df_final = pd.concat([df_1, df_2,df_3,df_5,df_6,df_7], axis=0)


# In[ ]:


df_final['class'].value_counts()


# In[ ]:


X_data=df_final.drop(['id','class'],axis=1)


# In[ ]:


Y_data=df_final['class']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X_data,Y_data,test_size=0.35,random_state=42) 
# X_train=X_data
# y_train=Y_data


# First Approach

# In[ ]:


from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
lgbm=LGBMClassifier(objective='multiclass',learning_rate=0.3) #72.5 accuracy
lgbm.fit(X_train,y_train)
lgb_pred = lgbm.predict(X_val)
lgb_accuracy = accuracy_score(y_val, lgb_pred)
print(f"The accuracy of the Light GBM is {lgb_accuracy*100:.1f} %")


# Second Approach

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
ensemble = ExtraTreesClassifier(n_estimators =200)
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_val)
ensemble_accuracy = accuracy_score(y_val, y_pred_ensemble)
print(f"The accuracy of the Extra Trees is {ensemble_accuracy*100:.1f} %")


# Testing

# In[ ]:


test=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')


# In[ ]:


data1=test.drop(['id'],axis=1)
# data1=test[features]


# In[ ]:


pred = lgbm.predict(data1)
pred


# In[ ]:


predictions = ensemble.predict(data1)
predictions


# In[ ]:


compare = pd.DataFrame({'id': test['id'], 'class' : pred})
compare.to_csv('submission13.csv',index=False)


# In[ ]:




