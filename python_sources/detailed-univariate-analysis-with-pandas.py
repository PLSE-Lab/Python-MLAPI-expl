#!/usr/bin/env python
# coding: utf-8

#  THIS NOTEBOOK IS BASED ON UNIVARIATE ANALYSIS.

# In[1]:


#Importing libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import scipy as sci
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm


# In[2]:


test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')


# In[3]:


train.shape


# In[4]:


test.shape


# In[5]:


train.head()


# In[6]:


test.head()


# # MISSING VALUES

# In[7]:


import missingno as msno


# In[8]:


msno.bar(train,figsize=(10,5))


# In[9]:


msno.bar(test,figsize=(10,5))


# param_1 has some NaN's. Approximately, half of param_2 is NaN. 60% of param_3 is also NaN

# # REGION

# In[10]:


print("no of unique values in region column of train data = ", len(set(train['region'])))


# In[11]:


print("no of unique values in region column of test data = ", len(set(test['region'])))


# In[12]:


f,ax=plt.subplots(1,2,figsize=(16,8))
train['region'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('afmhot',15),ax=ax[0])
test['region'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('afmhot',15),ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('Train')
ax[1].set_title('Test')
plt.show()


# In[13]:


#Let's check the distribution of train and test in region column


# In[14]:


pd.Series(((train['region'].value_counts())/len(train))/((test['region'].value_counts())/len(test))).plot(kind='barh',title = 'Ratio of region column in train / Ratio of region column in test',figsize=(14,7))


#  The distributions of train and test in region column look pretty similar

# # CITY

# In[15]:


print("no of unique values in city column of train data = ", len(set(train['city'])))


# In[16]:


print("no of unique values in city column of test data = ", len(set(test['city'])))


# In[17]:


f,ax=plt.subplots(1,2,figsize=(16,8))
train['city'].value_counts()[:50].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('cool',15),ax=ax[0])
test['city'].value_counts()[:50].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('cool',15),ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('Train')
ax[1].set_title('Test')
plt.show()


# # Parent Cateogry Name

# In[18]:


print("no of unique values in parent_category_name column of train data = ", len(set(train['parent_category_name'])))


# In[19]:


print("no of unique values in parent_category_name column of test data = ", len(set(test['parent_category_name'])))


# In[20]:


f,ax=plt.subplots(1,2,figsize=(16,8))
train['parent_category_name'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('prism',15),ax=ax[0])
test['parent_category_name'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('prism',15),ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('Train')
ax[1].set_title('Test')
plt.show()


# # Category Name

# In[21]:


print("no of unique values in category_name column of train data = ", len(set(train['category_name'])))


# In[22]:


print("no of unique values in category_name column of test data = ", len(set(train['category_name'])))


# In[23]:


f,ax=plt.subplots(1,2,figsize=(16,8))
train['category_name'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('gist_heat',15),ax=ax[0])
test['category_name'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('gist_heat',15),ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('Train')
ax[1].set_title('Test')
plt.show()


# # PARAM_1

# In[24]:


print("no of unique values in param_1 column of train data = ", len(set(train['param_1'])))


# In[25]:


print("no of unique values in param_2 column of test data = ", len(set(test['param_1'])))


# In[26]:


len([x for x in list(train['param_1'].value_counts()[:9].index) if x in list(test['param_1'].value_counts()[:8].index)])


# In[27]:


print("percentage of nan values in param_1 column in train data", round(train['param_1'].isnull().sum()*100/len(train),3))


# In[28]:


print("percentage of nan values in param_1 column in test data", round(test['param_1'].isnull().sum()*100/len(test),3))


#  Percentage of NaN values in train and test is similar

# In[29]:


f,ax=plt.subplots(1,2,figsize=(14,7))
train['param_1'].value_counts()[:8].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('gist_heat',15),ax=ax[0])
test['param_1'].value_counts()[:8].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('gist_heat',15),ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('Train')
ax[1].set_title('Test')
plt.show()


# # PARAM_2

# In[30]:


pd.Series([len(set(train['param_2'])),len(set(test['param_2']))],index=['Number of unique values in train','Number of unique values in test']).plot(kind='bar',title='Counts of Uniques in train vs Counts of Uniques in test')


# In[31]:


f,ax=plt.subplots(1,2,figsize=(14,7))
pd.Series([train['param_2'].isnull().sum(),train['param_2'].notnull().sum()],index=['Number of Nulls','Number of Not-Nulls']).plot(kind='pie',title='Counts of Nulls vs Counts of Not-Nulls in param_2 column in train',ax=ax[0])
pd.Series([test['param_2'].isnull().sum(),test['param_2'].notnull().sum()],index=['Number of Nulls','Number of Not-Nulls']).plot(kind='pie',title='Counts of Nulls vs Counts of Not-Nulls in param_2 column in test',ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('Train')
ax[1].set_title('Test')
plt.show()


# In[32]:


f,ax=plt.subplots(1,2,figsize=(16,8))
train['param_2'].value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('summer',15),ax=ax[0])
test['param_2'].value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('summer',15),ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('Train')
ax[1].set_title('Test')
plt.show()


# # Param_3

# In[33]:


pd.Series([len(set(train['param_3'])),len(set(test['param_3']))],index=['Number of unique values in train','Number of unique values in test']).plot(kind='bar',title='Counts of Uniques in train vs Counts of Uniques in test')


# In[34]:


f,ax=plt.subplots(1,2,figsize=(14,7))
pd.Series([train['param_3'].isnull().sum(),train['param_3'].notnull().sum()],index=['Number of Nulls','Number of Not-Nulls']).plot(kind='pie',title='Counts of Nulls vs Counts of Not-Nulls in param_2 column in train',ax=ax[0])
pd.Series([test['param_3'].isnull().sum(),test['param_3'].notnull().sum()],index=['Number of Nulls','Number of Not-Nulls']).plot(kind='pie',title='Counts of Nulls vs Counts of Not-Nulls in param_2 column in test',ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('Train')
ax[1].set_title('Test')
plt.show()


# Most of param_3 column is NULL

# In[35]:


f,ax=plt.subplots(1,2,figsize=(16,8))
train['param_3'].value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('spring',15),ax=ax[0])
test['param_3'].value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('spring',15),ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('Train')
ax[1].set_title('Test')
plt.show()


# # PRICE

# In[36]:


#Checking NaN's
f,ax=plt.subplots(1,2,figsize=(14,7))
pd.Series([train['price'].isnull().sum(),train['price'].notnull().sum()],index=['Number of Nulls','Number of Not-Nulls']).plot(kind='pie',title='Counts of Nulls vs Counts of Not-Nulls in param_2 column in train',ax=ax[0])
pd.Series([test['price'].isnull().sum(),test['price'].notnull().sum()],index=['Number of Nulls','Number of Not-Nulls']).plot(kind='pie',title='Counts of Nulls vs Counts of Not-Nulls in param_2 column in test',ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('Train')
ax[1].set_title('Test')
plt.show()


# In[59]:


sns.distplot(np.log(train['price'].dropna()+0.000001),kde=True,color='g',bins=20)


# # ITEM SEQ NUMBER

# In[56]:


train['item_seq_number'].describe()


# In[65]:


np.log(pd.concat([train['item_seq_number'],test['item_seq_number']])+0.0001).hist(bins=20)
plt.title('Log-transformed Item SEQ Number Distribution')
plt.show()


# # ACTIVATION DATE

# In[69]:


plt.plot(train.groupby('activation_date').count()[['region']], 'o-', label='train')
plt.plot(test.groupby('activation_date').count()[['region']], 'o-', label='test')
plt.title('Train and test period not overlapping.')
plt.legend(loc=0)
plt.ylabel('number of records')
plt.show()


# Look interesting . Test period is after train data.

# # USER TYPE

# In[72]:


f,ax=plt.subplots(1,2,figsize=(16,8))
train['user_type'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('afmhot',15),ax=ax[0])
test['user_type'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('afmhot',15),ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('Train')
ax[1].set_title('Test')
plt.show()


# Look similar. Very Good.

# # USER_TOP_1

# In[75]:


train['image_top_1'].describe()


# In[81]:


pd.concat([train[train['image_top_1'].notnull()]['image_top_1'],test[test['image_top_1'].notnull()]['image_top_1']]).hist(bins=20)
plt.title('Histogram of image_top_1 Distribution')
plt.show()


# # Deal Probability

# In[82]:


train['deal_probability'].describe()


# In[86]:


train['deal_probability'].hist(bins=5)

