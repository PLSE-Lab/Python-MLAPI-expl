#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('bright')
import os
print(os.listdir('../input/cat-in-the-dat'))


# In[ ]:


train_df = pd.read_csv('../input/cat-in-the-dat/train.csv')
test_df = pd.read_csv('../input/cat-in-the-dat/test.csv')
print(train_df.shape, test_df.shape)


# In[ ]:


train_df.head(3)


# In[ ]:


train_df.describe(include='O')


# - id 
# - bin_0 ~ bin_4 : binary
# - norm_0 ~ norm_9 : norminal
# - ord_0 ~ ord_5 : ordinal
# - day, month : date
# - target

# In[ ]:


sns.heatmap(train_df.corr())


# ## Target Distribution

# In[ ]:


sns.countplot(train_df['target'])
print("percentage of target 1 : {}%".format(train_df['target'].sum() / len(train_df)))


# ## Date Feature

# In[ ]:


print(train_df['month'].value_counts().sort_index(axis = 0) )
print(train_df['day'].value_counts().sort_index(axis = 0) )


# In[ ]:


def percentage_of_feature_target(df, feat, tar, tar_val):
    return df[df[tar]==tar_val][feat].value_counts().sort_index(axis = 0) / df[feat].value_counts().sort_index(axis = 0)

P_month = percentage_of_feature_target(train_df, 'month', 'target', 1)


# In[ ]:


P_month.plot()


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(20, 5))
sns.countplot('month', hue='target', data= train_df, ax=ax)
plt.show()


# In[ ]:


print(percentage_of_feature_target(train_df, 'day', 'target', 1))


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(20, 5))
sns.countplot('day', hue='target', data= train_df, ax=ax)
plt.show()


# ## Binary Features
# 
# you can also use Facetgrid too 

# In[ ]:


fig, ax = plt.subplots(1,4, figsize=(20, 5))
for i in range(4):
    sns.countplot(f'bin_{i}', hue='target', data= train_df, ax=ax[i])
    ax[i].set_title(f'bin_{i} feature countplot')
    print(percentage_of_feature_target(train_df,f'bin_{i}','target',1))
plt.show()


# ## Norminal Features

# In[ ]:


for i in range(9):
    tmp_list = train_df[f'nom_{i}'].value_counts().index
    print(f'nom_{i} feature\'s unique value {len(tmp_list)}')


# There are scale differences, so let's look at the features nom 0-4 first.

# In[ ]:


for i in range(5):
    print(percentage_of_feature_target(train_df, f'nom_{i}', 'target',1))


# In[ ]:


fig, ax = plt.subplots(1,5, figsize=(25, 7))
for i in range(5):
    sns.countplot(f'nom_{i}', hue='target', data= train_df, ax=ax[i])
    plt.setp(ax[i].get_xticklabels(),rotation=30)
plt.show()


# ### nom 5 ~ nom 9 
# 
# all length is 9

# In[ ]:


print(train_df['nom_5'].value_counts()) 


# It maybe some encoding about hexadecimal. but length 9 is uncomfortable...

# In[ ]:


fig, ax = plt.subplots(4,1,figsize=(40, 40))
for i in range(5, 9):
    sns.countplot(sorted(train_df[f'nom_{i}']), ax=ax[i-5])
    plt.setp(ax[i-5].get_xticklabels(),rotation=90)
plt.show();


# How about percentage grapth..? It seems there are no relation between nom's..

# In[ ]:






for i in range(5, 9):
    fig, ax = plt.subplots(1,1,figsize=(8, 2))
    P_nom = percentage_of_feature_target(train_df, f'nom_{i}', 'target', 1)
    P_nom.plot() # easy plot
    #sns.barplot(P_nom.index, P_nom, ax=ax[i-5])
    #plt.setp(ax[i-5].get_xticklabels(),rotation=90)
plt.show();


# I think nom_8 can make total 17 group and nom5 - 6 - 7 have some relation
# but i don'k know exactly... :(

# In[ ]:





# In[ ]:


# fig, ax = plt.subplots(1,1,figsize=(40, 10))
# sns.countplot(sorted(train_df['nom_9']), ax=ax)
# plt.show();


# ## Ordinal Features

# I am just curious about this [discussion](https://www.kaggle.com/c/cat-in-the-dat/discussion/105702#latest-633384)
# 
# so I want to find out the pattern of ord_5

# In[ ]:


train_df['ord_5'].value_counts().sort_index(axis=0)


# And I find out ord3, ord4, ord5 feature target percentage is linear (in dictionary order)

# In[ ]:


P_ord5 = percentage_of_feature_target(train_df, 'ord_5', 'target', 1)
fig, ax = plt.subplots(1,1,figsize=(20,7))
sns.barplot(P_ord5.index, P_ord5, ax=ax)
plt.title('ord_5 : Percentage of target==1 in dictionary order')
plt.setp(ax.get_xticklabels(),rotation=90, fontsize=5)
plt.show()


# In[ ]:


P_ord4 = percentage_of_feature_target(train_df, 'ord_4', 'target', 1)
fig, ax = plt.subplots(1,1,figsize=(20,7))
sns.barplot(P_ord4.index, P_ord4, ax=ax)
plt.title('ord_4 : Percentage of target==1 in dictionary order')
plt.setp(ax.get_xticklabels(),rotation=90, fontsize=5)
plt.show()


# In[ ]:


P_ord3 = percentage_of_feature_target(train_df, 'ord_3', 'target', 1)
fig, ax = plt.subplots(1,1,figsize=(20,7))
sns.barplot(P_ord3.index, P_ord3, ax=ax)
plt.title('ord_3 : Percentage of target==1 in dictionary order')
plt.setp(ax.get_xticklabels(),rotation=90, fontsize=5)
plt.show()


# ## Model

# In[ ]:


import sklearn.

