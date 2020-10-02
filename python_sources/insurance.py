#!/usr/bin/env python
# coding: utf-8

# In[12]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[96]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[22]:


print(df_train.shape)
print(df_test.shape)


# In[23]:


df_train.sample(3)


# The following variables are all categorical (nominal):
# 
# Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41
# 
# The following variables are continuous:
# 
# Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5
# 
# The following variables are discrete:
# 
# Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32
# 
# 

# In[48]:


df_train.dtypes.unique()


# In[29]:


df_train.select_dtypes(include='O').columns


# In[32]:


df_train.select_dtypes(include='float64').columns


# In[31]:


df_train.select_dtypes(include='int64').columns


# <h2>Data Preprocessing</h2>
# 
# The first thing is to check if there is any missing value in the dataset. I used python to extract the columns which contains the missing value as well as the percentage of missing value.

# In[39]:


#proportion of null values per columns
print('proportion of null values in train set : ')
print(df_train.isnull().sum(axis = 0).sort_values(ascending = False).head(10)/len(df_train))
print('\n')
print('proportion of null values in test set : ')
print(df_test.isnull().sum(axis = 0).sort_values(ascending = False).head(10)/len(df_test))


# In[59]:


df = df_train.isnull().sum()[df_train.isnull().sum() !=0]/len(df_train)
df=pd.DataFrame(df.reset_index())
df.head(3)
total = len(df_train)
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))


# In[91]:


#Exploring missing values
train_missing= df_train.isnull().sum()[df_train.isnull().sum() !=0]
train_missing=pd.DataFrame(train_missing.reset_index())
train_missing.rename(columns={'index':'features',0:'missing_count'},inplace=True)
train_missing['missing_count_percentage']=((train_missing['missing_count'])/len(df_train))*100
plt.figure(figsize=(15,7))
#train_missing
splot = sns.barplot(y=train_missing['features'],x=train_missing['missing_count_percentage'])
for p in splot.patches:
    splot.annotate(str(format(p.get_width(), '.2f')+'%'), (p.get_width()+3,p.get_y() + p.get_height() ), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# In[97]:


df_train.head(3)


# In[93]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_train['Product_Info_2']=le.fit_transform(data_train['Product_Info_2'])
df_test['Product_Info_2']=le.transform(data_test['Product_Info_2'])


# In[94]:


df_train.describe()


# In[98]:


#Function for normalization
def normalization(data):
    return (data - data.min())/(data.max() - data.min())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




