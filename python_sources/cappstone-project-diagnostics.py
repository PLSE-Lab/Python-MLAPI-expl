#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data=pd.read_csv('../input/Patient Data.csv')
data.head()


# In[ ]:


data.isna().sum()


# In[ ]:


data.shape


# In[ ]:


data['smoking history'].unique()


# In[ ]:


#import pandas_profiling


# In[ ]:


#pandas_profiling.ProfileReport(data)


# In[ ]:


#import matplotlib.pyplot as plt
#%matplotlib inline


# In[ ]:


#plt.scatter(data.index,data.BMI)


# In[ ]:


data['smoking history']=data['smoking history'].fillna('Unknown')


# In[ ]:


data['smoking history'].value_counts()


# In[ ]:


data['smoking history'].unique()


# In[ ]:


data.dtypes


# In[ ]:


data.isna().sum()


# In[ ]:


train_data_cat=data.select_dtypes(include=[object])
train_data_num=data.select_dtypes(exclude=[object])
train_data_cat=pd.get_dummies(train_data_cat, columns=train_data_cat.columns, drop_first=True)
data_df=pd.concat([train_data_cat,train_data_num],axis=1)


# In[ ]:


data_df.head()


# In[ ]:


data_df.shape


# In[ ]:


from fancyimpute import IterativeImputer


# In[ ]:


XY_incomplete = data_df.copy()


# In[ ]:


data_complete_df = pd.DataFrame(IterativeImputer(n_iter=5, sample_posterior=True, random_state=8).fit_transform(XY_incomplete))


# In[ ]:


data_complete_df.columns=data_df.columns


# In[ ]:


data_complete_df.head()


# In[ ]:


data_complete_df.isna().sum()


# In[ ]:


data_complete_df['diabetes & hypertension']=data_complete_df['diabetes']*data_complete_df['hypertension']
data_complete_df['diabetes & srtoke']=data_complete_df['diabetes']*data_complete_df['stroke']
data_complete_df['diabetes & heart disease']=data_complete_df['diabetes']*data_complete_df['heart disease']
data_complete_df['hypertension & stroke']=data_complete_df['hypertension']*data_complete_df['stroke']
data_complete_df['hypertension & heart disease']=data_complete_df['hypertension']*data_complete_df['heart disease']
data_complete_df['stroke & heart disease']=data_complete_df['stroke']*data_complete_df['heart disease']


# In[ ]:


data_complete_df['diabetes,hypertension,stroke']=data_complete_df['diabetes']*data_complete_df['hypertension']*data_complete_df['stroke']
data_complete_df['diabetes,hypertension,heart disease']=data_complete_df['diabetes']*data_complete_df['hypertension']*data_complete_df['heart disease']
data_complete_df['hypertension,stroke,heart disease']=data_complete_df['hypertension']*data_complete_df['stroke']*data_complete_df['heart disease']


# In[ ]:


data_complete_df.shape


# In[ ]:


data_complete_df['diabetes & hypertension'].value_counts()


# In[ ]:


data_complete_df.head()


# In[ ]:


dd=data_complete_df[data_complete_df['diabetes']==1]


# In[ ]:


dd1=dd[dd['hypertension']==1]


# In[ ]:


dd1.shape


# In[ ]:


data_complete_df['hypertension,stroke,heart disease'].value_counts()


# In[ ]:


dataaa=data_complete_df[data_complete_df['hypertension,stroke,heart disease']==1]


# In[ ]:


dataaa.shape


# In[ ]:


dataaa.head(10)


# In[ ]:


dataaa['age'].value_counts()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


plt.hist(dataaa['age'])


# In[ ]:


plt.hist(dataaa['BMI'])


# In[ ]:


data_complete_df['diabetes,hypertension,heart disease'].value_counts()


# In[ ]:


data_complete_df['diabetes,hypertension,stroke'].value_counts()


# In[ ]:


data_complete_df['diabetes & heart disease'].value_counts()


# In[ ]:


data_complete_df['diabetes & hypertension'].value_counts()


# In[ ]:


data_complete_df['diabetes & srtoke'].value_counts()


# In[ ]:


data_complete_df['hypertension & heart disease'].value_counts()


# In[ ]:


data_complete_df['hypertension & stroke'].value_counts()


# In[ ]:


data_complete_df['stroke & heart disease'].value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_stroke=data_complete_df.drop(['stroke'],axis=1)
X_stroke.head()


# In[ ]:


Y_stroke=data_complete_df['stroke']
Y_stroke.head()


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split()


# In[ ]:





# In[ ]:




