#!/usr/bin/env python
# coding: utf-8

# # In this notebook,  I will describe my approach to the data cleaning and some visualization of price

# In[ ]:


cd ../input


# In[ ]:


ls 


# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[ ]:


print(train.shape)
train.head()


# In[ ]:


print(test.shape)
test.head()


# In[ ]:


len(np.unique(train['id']))


# In[ ]:


len(np.unique(train['num_room']) )


# In[ ]:


train = train.set_index('timestamp')
test = test.set_index("timestamp")
train.head()


# In[ ]:


train.drop("id", axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.shape[0]


# In[ ]:


train['price_doc'].values


# In[ ]:


ax = train['price_doc'].plot(style=['-'])
ax.lines[0].set_alpha(0.8)
plt.xticks(rotation=90)
plt.title("linear scale")
ax.legend()


# In[ ]:


ax = train['price_doc'].plot(style=['-'])
ax.lines[0].set_alpha(0.3)
ax.set_yscale('log')
plt.xticks(rotation=90)
plt.title("logarithmic scale")
ax.legend()


# In[ ]:


# no missing value for the target value
train[train['price_doc'].isnull()]


# In[ ]:


train.columns[train.isnull().any()]


# In[ ]:


train2 = train.fillna(train.median())


# In[ ]:


train2.head()


# In[ ]:


train2.columns[train2.isnull().any()]


# In[ ]:


train2.corr()


# In[ ]:


categorical = []
for i in train2.columns:
    if type(train2[i].values[0]) == str:
        categorical.append(i)
print(categorical)
print(len(categorical))


# In[ ]:


train2.shape


# In[ ]:


train2[categorical].head()


# In[ ]:


np.unique(train2['product_type'])


# In[ ]:


for cat in categorical:
    print(cat, ':', np.unique(train2[cat]))


# In[ ]:


yes_no_mapping = {'no': 0, 'yes': 1}


# In[ ]:


# ordinal features which could be rendered as 0 and 1,
# each corresponding to 'no' and 'yes'
categorical[2:-1]


# In[ ]:


for i in categorical[2:-1]:
    train2[i] = train2[i].map(yes_no_mapping)


# In[ ]:


categorical = []
for i in train2.columns:
    if type(train2[i].values[0]) == str:
        categorical.append(i)
print(categorical)
print(len(categorical))


# In[ ]:


np.unique(train2['ecology'].values)


# In[ ]:


rate_mapping = {'excellent': 3, 'good': 2, 'satisfactory': 2, 'poor': 1, 'no data': np.nan} 


# In[ ]:


train2['ecology'] = train2['ecology'].map(rate_mapping)


# In[ ]:


print(len(train2[train2['ecology'].isnull()]))


# In[ ]:


print(len(train2[train2['ecology'].notnull()]))


# In[ ]:


print(train2.shape[0])


# In[ ]:


print(len(train2[train2['ecology'].isnull()]) + len(train2[train2['ecology'].notnull()]))


# In[ ]:


train2 = train2.fillna(train2.median())


# In[ ]:


print(len(train2[train2['ecology'].isnull()]))


# In[ ]:


train2.corr()


# In[ ]:


ls


# In[ ]:


train2.head()


# In[ ]:


ls ../


# In[ ]:


train2.head() 


# In[ ]:


ls ../


# # Modify test data

# In[ ]:


test = pd.read_csv("test.csv")


# In[ ]:


test.head()


# In[ ]:


test = test.set_index('timestamp')
test.head()


# In[ ]:


test.drop("id", axis=1, inplace=True)
print(test.shape)


# In[ ]:


for i in test.columns:
    if i not in train.columns:
        print(i)


# In[ ]:


categorical = []
for i in test.columns:
    if type(test[i].values[0]) == str:
        categorical.append(i)
print(categorical)
print(len(categorical))


# In[ ]:


categorical[2:-1]


# In[ ]:


for i in categorical[2:-1]:
    test[i] = test[i].map(yes_no_mapping)


# In[ ]:


test['ecology'] = test['ecology'].map(rate_mapping)


# In[ ]:


len(test[test['ecology'].isnull()])


# In[ ]:


test = test.fillna(test.median())


# In[ ]:


test.columns[test.isnull().any()]


# In[ ]:


# there are 33 missing values in a column called 'producty_type'
len(test[test['product_type'].isnull()])


# In[ ]:


ls


# In[ ]:


train2.head()


# In[ ]:




