#!/usr/bin/env python
# coding: utf-8

# In[94]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[95]:


# Import data load 

train=pd.read_csv('../input/train_file.csv')
test=pd.read_csv('../input/test_file.csv')
sample=pd.read_excel('../input/sample_submission08f968d.xlsx')


# In[96]:


#Check train data
train.head()


# In[97]:


#Check train columns and test columns
train.columns,test.columns


# In[98]:


# drop columns
train.drop(['Patient_ID','LocationDesc','Greater_Risk_Question','Description','GeoLocation'],axis=1,inplace=True)
test.drop(['Patient_ID','LocationDesc','Greater_Risk_Question','Description','GeoLocation'],axis=1,inplace=True)


# In[99]:


# check nan value in train and test
train.isnull().any().sum(),test.isnull().any().sum()


# In[100]:


# Import seaborn and maplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[101]:


plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.distplot(train['Sample_Size'])


plt.subplot(1,2,2)
sns.boxplot(train['Sample_Size'])


# In[102]:


#train=train.loc[(train['Sample_Size']<19000)]


# In[103]:


train['Sample_Size']=np.log(train['Sample_Size'])
test['Sample_Size']=np.log(test['Sample_Size'])

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.distplot((train['Sample_Size']))


plt.subplot(1,2,2)
sns.boxplot((train['Sample_Size']))


# In[105]:


# probability greater than 1
train=train[train['Greater_Risk_Probability']>1]


# In[106]:


# Check with year how is Greater_Risk_Probability 
plt.figure(figsize=(18,7))
sns.boxplot(data=train,x='YEAR',y='Greater_Risk_Probability')
# It is decreasing


# In[107]:


# Check how Subtopic is related to Greater_Risk_Probability
sns.boxplot(data=train,x='Subtopic',y='Greater_Risk_Probability')


# * Subtopic 1 decrease the risk
# * Could be important feature

# In[108]:


# Check gender 
sns.boxplot(data=train,x='Sex',y='Greater_Risk_Probability')


# * Risk is same across gender 

# In[109]:


# Check with Race
plt.figure(figsize=(16,7))
plt.xticks( rotation='45')
sns.boxplot(data=train,x='Race',y='Greater_Risk_Probability')


# * Asian have less Risk

# In[110]:


# Check with Grade
plt.figure(figsize=(16,7))
sns.boxplot(data=train,x='Grade',y='Greater_Risk_Probability')


# * Grade 3 have low Risk

# In[111]:


# With QuestionCode
plt.figure(figsize=(19,5))
plt.xticks(rotation='45')
sns.boxplot(data=train,x='QuestionCode',y='Greater_Risk_Probability')


# * H40 has very high risk.
# * H42,H46 and H43 have high risk.
# * H41,H56 and H48 risk.

# In[112]:


# With StratID1
sns.boxplot(data=train,x='StratID1',y='Greater_Risk_Probability')


# * Almost same risk.

# In[113]:


# Check with StratID2
sns.boxplot(data=train,x='StratID2',y='Greater_Risk_Probability')


# In[114]:


sns.boxplot(data=train,x='StratID3',y='Greater_Risk_Probability')


# In[115]:


sns.boxplot(data=train,x='StratificationType',y='Greater_Risk_Probability')


# In[116]:


train['Sample_Size']=np.round(train['Sample_Size'])
test['Sample_Size']=np.round(test['Sample_Size'])


# In[120]:


train.head()


# In[121]:


test.head()


# In[126]:


#train.columns
from sklearn.preprocessing import LabelEncoder
Feature =['YEAR','Sex', 'Race', 'QuestionCode','StratificationType']
for i in Feature:
    LR=LabelEncoder()
    train[i] = LR.fit_transform(train[i])


# In[128]:


for i in Feature:
    LR=LabelEncoder()
    test[i] = LR.fit_transform(test[i])


# In[129]:





# In[130]:


test.head()


# In[ ]:




