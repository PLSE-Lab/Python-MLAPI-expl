#!/usr/bin/env python
# coding: utf-8

# #### Although the metric only considers our predictions on the 'target' variable, the organizers have also provided us with auxillary information on the nature of the text-some of which may be able to help us to further optimize/regularize our neural networks. So this kernel is to explore the their distributions. 

# > Reading Data

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import random
import os
os.listdir('../input')


# In[ ]:


train = pd.read_csv('../input/train.csv')


# > Extract Subset with full data (no missing values)

# In[ ]:


train_full = train


# In[ ]:


cols = ['target',
    'severe_toxicity',
    'obscene',
    'threat',
    'insult',
    'identity_attack',
    'sexual_explicit',
    'male', 'female','homosexual_gay_or_lesbian','christian','jewish','muslim','black',
    'white','psychiatric_or_mental_illness'
]
print('Numbers of columns of interest: ', len(cols))


# In[ ]:


for col in cols:
    train_full = train_full[~pd.isnull(train_full[col])]
print('Proportion of dataset with no missing value: ', len(train_full)/len(train))


# > Saving result

# In[ ]:


print('Number of samples: ', len(train_full))
train_full.to_csv('train_full.csv')


# > A quick look at the subset

# In[ ]:


train_full.head()


# In[ ]:


train_full.columns


# > Distribution of variables

# In[ ]:


f, axes = plt.subplots(4, 4,figsize=(20, 15))

color = ['b','g','c','k']

i = 0
j = 0
for name in cols:
    if j == 4:
        j = 0
        i +=1
    sns.distplot(train_full[[name]], kde = False, ax=axes[i][j])
    axes[i][j].set_yscale('log')
    axes[i][j].set_title(name)
    j += 1


# The distributions of variables of interests are generally skewed, but there are still a number of samples across all values. Maybe later I will post some results when trained on this subset. Now let's look at the correlation between target and each of the variables:

# In[ ]:


sns.pairplot(train_full[['target','sexual_explicit','obscene']])


# In[ ]:


sns.pairplot(train_full[['target','threat','identity_attack']])


# In[ ]:


sns.pairplot(train_full[['target','severe_toxicity','insult']])


# As seen above, the subtype attributes are aligned with target in the general directions: target are bounded by a certain value that is roughly proportional to each subtype value (this could be a plausible post-processing process). But notice the nuances between different subtypes: like those between "obscene" and "sexual_explicit."

# In[ ]:


sns.pairplot(train_full[['target','black','white','psychiatric_or_mental_illness']])


# In[ ]:


sns.pairplot(train_full[['target','male','female','homosexual_gay_or_lesbian']])


# In[ ]:


ax = sns.pairplot(train_full[['target','christian','jewish','muslim']])


# It seems that ideneity attributes are not correlated with the target. The challenge, then, is to enforce this homogeneity on your models. Good luck!
