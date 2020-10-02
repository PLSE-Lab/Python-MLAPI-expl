#!/usr/bin/env python
# coding: utf-8

# # Data Munging:
# ## Its all about treating and fixing missing values in the data set

# ## Possible Solution: Imputation
# ### Types:
# #### 1) Mean Imputation: This imputation is widely used to replace null values in the dataset. The disadvantage of these imputation is that it may leads to the problem of outliers if extreme values present in the dataset

# ### 2) Median Imputation: This is better than mean imputation as extreme values does not affect the model
# ### 3) Mode Imputation: This kind of imputation is mainly used for categorical missing values in the dataset

# ## Treating missing values using Pandas

# In[1]:


import numpy as np
import pandas as pd
import os


# In[2]:


import os
print(os.listdir("../input"))
train_df=pd.read_csv("../input/train.csv",index_col='PassengerId')
test_df=pd.read_csv("../input/test.csv",index_col='PassengerId')


# In[3]:


test_df['Survived']=-888


# In[4]:


df=pd.concat((train_df,test_df),axis=0)


# In[5]:


df.info()


# ## Feature: Embarked

# In[6]:


df[df.Embarked.isnull()]


# In[7]:


df.Embarked.value_counts()


# In[8]:


#which emabarked point has higher survival count
pd.crosstab(df[df.Survived!=-888].Survived,df[df.Survived!=-888].Embarked)


# In[9]:


df.groupby(['Pclass','Embarked']).Fare.median()


# In[10]:


df.Embarked.fillna('C',inplace=True)


# In[11]:


df[df.Embarked.isnull()]


# ## Feature: Fare

# In[12]:


df.info()


# In[13]:


df[df.Fare.isnull()]


# In[14]:


df.groupby(['Pclass','Fare']).Embarked.median()


# In[15]:


median_fare=df.loc[(df.Pclass==3)& (df.Embarked=='S'),'Fare'].median()


# In[16]:


median_fare


# In[17]:


df.Fare.fillna(median_fare,inplace=True)


# In[18]:


df[df.Fare.isnull()]


# In[19]:


df.info()


# ## Feature: Age

# In[20]:


pd.options.display.max_rows=15


# In[21]:


df[df.Age.isnull()]


# ### option 1: replace age missing values with mean of age

# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.Age.plot(kind='hist',bins=20,color='c')


# In[23]:


#mean of Age
df.Age.mean()


# In[24]:


# check median of Ages
df.Age.median()


# ## option 2: replace Age with median by gender

# In[25]:


df.groupby('Sex').Age.median()


# In[26]:


df[df.Age.notnull()].boxplot('Age','Sex')


# ## option 3: replace age with median by Pclass

# In[27]:


df.groupby('Pclass').Age.median()


# ## option 4: replace by median of title

# In[28]:


def GetTitle(name):
    title_group={'mr':'Mr',
            'mrs':'Mrs',
            'miss':'Miss',
            'master':'Master',
            'don':'Sir',
            'rev':'Sir',
            'dr':'Officer',
            'mme':'Mrs',
            'ms':'Mrs',
            'major':"Officer",
            'lady':'Lady',
            'sir':'Sir',
            'mlle':'Miss',
            'col':'Officer','capt':'Officer',
            'the countess':'Lady',
            'jonkheer':'Sir',
            'dona':'Lady'}
    
    first_name_with_title=name.split(',')[1]
    title=first_name_with_title.split('.')[0]
    title=title.strip().lower()
    return title_group[title]


# In[29]:


df.Name.map(lambda x:GetTitle(x))


# In[ ]:





# In[30]:


df.Name.map(lambda x:GetTitle(x)).unique()


# In[31]:


df["Title"]=df.Name.map(lambda x: GetTitle(x))


# In[32]:


df.info()


# In[33]:


df[df.Age.notnull()].boxplot('Age','Title')


# In[34]:


title_age_median=df.groupby('Title').Age.transform('median')


# In[35]:


df.Age.fillna(title_age_median,inplace=True)


# In[36]:


df.info()


# In[37]:


df.head()


# In[ ]:




