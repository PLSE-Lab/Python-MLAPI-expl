#!/usr/bin/env python
# coding: utf-8

# # *** EDA -- Understanding our data ***

# # ** HML Machine Learning Challenge #1 **

# # Introduction:
# 
# ## This is an Exploratory Data Analysis for the [Hawaii Machine Learning Meetup Challenge (#1)](https://www.kaggle.com/c/hawaiiml0) with Python.

# The aim of this challenge is to forecast quantity of items sold.
# The [training data](https://www.kaggle.com/c/hawaiiml0/data/training.csv) is 30mb.

# In[5]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import gc
import os
# Comment this if the data visualisations doesn't work on your side
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('bmh')


# In[ ]:





# In[6]:


FILE_DIR = "../input/hawaiiml-data"
print([f for f in os.listdir(path=FILE_DIR)])

for f in os.listdir(FILE_DIR):
    print('{0:<30}{1:0.2f}MB'.format(f, 1e-6*os.path.getsize(f'{FILE_DIR}/{f}')))


# In[7]:


df = pd.read_csv(f'{FILE_DIR}/train.csv', encoding='ISO-8859-1')
df_test = pd.read_csv(f'{FILE_DIR}/test.csv', encoding='ISO-8859-1')
submission = pd.read_csv(f'{FILE_DIR}/sample_submission.csv', encoding='ISO-8859-1')


# # Training Data

# In[8]:


print(df.shape)
df.head()


# ### Training set is 371899 rows x 10 columns: id; date;	time; invoice_id; stock_id; customer_id; country;	description; unit_price; quantity
# 
# Traning data appears to be de-normalized (order header data included with order detail.  Verify this by finding duplicate ids.

# In[9]:


df.info()


# All columns are highly represented--only description (which is likely useless anyway for our predictions) has any missing values.

# In[10]:


print("id count: " + str(df['id'].count()))
#any duplicated id values?
print("duplicated id count: " + str(df[df.duplicated(['id'], keep=False)]['id'].count()))


# In[11]:


print("Of", df['id'].count(), "id values,", df[df.duplicated(['id'], keep=False)]['id'].count(), "were duplicates.")


# No, no duplicated ids, so probably no predictive value in this column.
# How are multiline orders handled? By matching invoice_id

# In[12]:


print("Of", df['invoice_id'].count(), "invoice_id values,", df[df.duplicated(['invoice_id'], keep=False)]['invoice_id'].count(), "were duplicates.")


# In[13]:


df[df.duplicated(['invoice_id'], keep=False)].head()

# Note: this is equivalent of SQL: select * from df_train 
# where invoice_id in 
#  (select invoice_id from df_train 
#   group by invoice_id having count(*) > 1)


# So yes, let's look at a single example

# In[14]:


df.loc[df['invoice_id']==6757].head()
#5 of ~163 rows for this invoice.


# ### This shows that this data is fully denormalized, with date,	time, invoice_id, customer_id, country repeated for each row in an invoice.  All that varies is id,	stock_id, description, unit_price and quantity.

# In[15]:


# Let's add count of distinct items (itemcount) for each invoice to the df for further inquiries.
df['itemcount'] = df['invoice_id'].map(df['invoice_id'].value_counts())
#df_test['itemcount'] = df_test['invoice_id'].map(df_test['invoice_id'].value_counts())
#NOTE: Remember to do this (and any) transform with test during feature engineering.


# ### Of the 371899 rows in the training set, how many are for multiple items?

# In[16]:


print('Of', df['invoice_id'].nunique(), "training invoice_id values,", df[df.duplicated(['invoice_id'], keep=False)]['invoice_id'].nunique(), "were for multiple items.")


# ### 17701/19946 (89%) of orders have multiple items.  
# 
# ### What is their distribution?  (Qty of items ordered may make a qualitative difference in the qty ordered per item).
# 

# In[17]:


print(df['quantity'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['quantity'], color='g', bins=100, hist_kws={'alpha': 0.4});


# Ummm, max quantity is 74k? What's going on there? Might need to weed that record for reasonable evaluation of the rest (and check df_test for outliers).

# In[18]:


df[(df['quantity'] >= 4000)]


# In[19]:


#describe without big outlier.  Leave others for now. Look at log(qty) as well, as this might 
#reduce the outlier distortion.
#sns.distplot(df[(df['quantity'] <= 5000)'quantity'], color='g', bins=100, hist_kws={'alpha': 0.4});
df[(df['quantity'] < 4000)].describe()


# ## Insights: 
# . mean > 3x median quantity
# . mean ~ 75th percentile.

# In[20]:


plt.figure(figsize=(9, 8))
sns.distplot(np.log1p(df['quantity'])[ (df['quantity'] < 4000) ], color='g', bins=100, hist_kws={'alpha': 0.4});
plt.title("Distribution of log1p(quantity)-excluding outliers ", fontsize=15)
plt.show()

plt.figure(figsize=(9, 8))
sns.distplot(np.log1p(df['quantity']), color='g', bins=100, hist_kws={'alpha': 0.4});
plt.title("Distribution of log1p(quantity), including outliers", fontsize=15)
plt.show()


# ## numpy's log1p - brings a reasonable spread to the plot.  Will bring this into the feature set.

# In[21]:


# Let's add log1p(quantity) to the df for further inquiries.
df['quantity_log1p'] = df['quantity'].apply(np.log1p)

#NOTE: In this case, no column for df_test because it does not contain 'quantity' field.


# ### And, finally, let's take a look at dates.  Will day of week or day of month have an impact on log1p(quantity)?
# Keeping in mind I don't really care about how many orders will happen, but want to predict for a given order, what the quantity for each item will be. Focusing on stats related to quantity for time periods.

# In[22]:


df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df["wday"] = df['date'].dt.weekday
df["day"] = df['date'].dt.day
plt.figure(figsize=(12,8))
ax = sns.boxplot(x="wday", y="quantity_log1p", data=df)
plt.ylabel('quantity_log1p', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Boxplot of log1p(quantity) by day of week", fontsize=15)
plt.show()


# No sales data for Saturday (5)? Hmm. Weird, given Sunday sales. Think about why? Analysis error?

# In[23]:


plt.figure(figsize=(12,8))
ax = sns.boxplot(x="day", y="quantity_log1p", data=df)
plt.ylabel('quantity_log1p', fontsize=12)
plt.xlabel('Day of month', fontsize=12)
plt.title("Boxplot of log1p(quantity) by day of month", fontsize=15)
plt.show()


# ### Order quantities seem pretty steady by day of month. Would not have been surprised by elevated activity around the beginning or end of the month.  
# 
# ### Take a look at hourly data:

# In[24]:


df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
plt.figure(figsize=(12,8))
ax = sns.boxplot(x="hour", y="quantity_log1p", data=df)
plt.ylabel('quantity_log1p', fontsize=12)
plt.xlabel('Hour of Day', fontsize=12)
#plt.xticks(rotation='vertical')
plt.title("Boxplot of log1p(quantity) by Hour", fontsize=15)
plt.show()


# ### Looks like mornings have higher sales quantities.  This might be a significant learning feature.
# 
# --------

# # Summary
# ### I hope you found this helpful for ideas and techniques, exploring our data.  A more thorough look might consider test data, order quantities related to price or item (for example, qty sold per order would differ for apples vs watermelons), location analysis, and more on the missing data.  I did not explore the missing data because I feel that description (only column with missing data) will have little bearing on sales qty.  Location might, so will consider one-hot encoding during training.
# 
# ## I would love to see feedback, questions and suggestions for improvements.  
# ### -Bob

# In[ ]:




