#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# ## Import Data

# In[2]:


## load the data in the dataframe using pandas

import os
print(os.listdir("../input"))
train_df=pd.read_csv("../input/train.csv",index_col='PassengerId')
test_df=pd.read_csv("../input/test.csv",index_col='PassengerId')


# In[3]:


type(train_df)


# # Basic Structure

# In[4]:


## use .infor() to get information about the dataframe
train_df.info()


# In[5]:


test_df.info()


# In[6]:


##Information of the dataset
#1) Survived: Number of passengers survived(1=yes,0=No)
#2) PClass is the passenger class (1=1st class,2=2nd class,...)
#3) Name
#4) Sex
#5) Age
#6) Sibsp stnads for siblings and spouse
#7) Parch stands for parents/children
#8) Ticket
#9) Fare
#10) Cabin
#11) Embarked is the boarding point for the passengers


# In[7]:


test_df['Survived']=-888 #Add default value to survived column


# In[8]:


df=pd.concat((train_df,test_df),axis=0)


# In[9]:


df.info()


# In[10]:


#use head() to get top 5 records
df.head()


# In[11]:


df.tail()


# In[12]:


#column selection with dot
df.Name


# In[13]:


#select data using label based using loc[rows_range,col_range]
df.loc[5:10,'Age':'Survived']


# In[14]:


#position based indexing using iloc()
df.iloc[5:10, 0:5]


# In[15]:


#filter row based on the condition
male_passengers=df.loc[df.Sex=='male',:]
print("Number of male passengers:{0}".format(len(male_passengers)))


# In[16]:


male_passengers_first_class=df.loc[((df.Sex=='male')& (df.Pclass==1)),:]
print("Number of male passengers with first class:{0}".format(len(male_passengers_first_class)))


# # Summary Statistics

# ## Numerical: mainly mean ,median,variance,standard deviation
# ## CAtegorical: total count, unique count

# In[17]:


df.describe()


# ## centrality measure: A number to represent entire set of values. Number central to data

# ## Spread/Dispersion measure: Gives information How values are similar or disimilar in our dataset

# ## Range: Difference between maximum and minium

# In[18]:


#Box plot
get_ipython().run_line_magic('matplotlib', 'inline')
df.Fare.plot(kind='box')


# In[19]:


df.describe(include='all')


# In[20]:


df.Sex.value_counts(normalize=True)


# In[21]:


df.Sex.value_counts().plot(kind='bar')


# In[22]:


df.Pclass.value_counts().plot(kind='bar')


# In[23]:


df.Pclass.value_counts().plot(kind='bar',rot=0,title="class of passengers count",color='red');


# In[ ]:





# In[ ]:




