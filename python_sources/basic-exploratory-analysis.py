#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.float_format = '{:,.2f}'.format


# In this kernel I forked [https://www.kaggle.com/antondergunov/basic-exploratory-analysis] and am now explaining my interpretation while also adding additional code. For this competition we are looking at comments and predicting the probability that they are toxic. Let's first look at the datasets

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


print ("The training set has %i rows" % (len(train_df)))
print ("The testing set has %s rows" % (len(test_df)))


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# So we can see that for the training set we are given a lot of different features for each comment, but we do not see that in the testing set. Unlike the first toxic challenge, they want us to detect toxic comments but mostly they want us to  minimize unintended model bias. That is where these extra feature come in handy. In other words, we need to build a model that can generate some of these features for our testing set, but it may also be good to know which features are most useful. Without looking at the data I am guessing that the tags with less than 500 examples in the test set (combined public and private) will not be that useful. These can be found in the data tab on the competition page.

# In[ ]:


train_df.describe().transpose()


# In[ ]:


print("There are %i comments without identity features" % (len(train_df)-405130))


# This means that we can only train on 405,130 comments to generate identity features. Also note that the mean of most of these tags is 0. This is expected since we expect that most comments do not talk about these tags individually. It also said in the data description of the competition that  Now let's plot the correlations between the different features.

# In[ ]:


corr = train_df.drop(["id", "publication_id", "parent_id", "article_id"], axis=1).corr()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr.columns)), corr.columns);


# I didn't really gain any new information from this plot. I wonder if we only look at target and the identity tags with >500 examples in the testset.

# In[31]:


train_identityLabs_df = train_df.copy()
identitiesInTestSet=["male","female","homosexual_gay_or_lesbian","christian","jewish","muslim","black","white","psychiatric_or_mental_illness"]
train_identityLabs_df = train_identityLabs_df.loc[:,["target"]+identitiesInTestSet]
identityLabs_corr = train_identityLabs_df.corr()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(identityLabs_corr)
plt.xticks(range(len(identityLabs_corr.columns)), identityLabs_corr.columns, rotation='vertical')
plt.yticks(range(len(identityLabs_corr.columns)), identityLabs_corr.columns);


# Yea this still doesn't say that much. It is just going to be harded to differentiate between toxic comments that are labeled as black/white or male/female than the other tags I guess.

# Next we look at the distribution of the labels given to the subcategories of toxicity.

# In[ ]:


fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(20, 10))
for i, metric in enumerate(['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']):
    ax[i // 3, i % 3].set_title(metric)
    ax[i // 3, i % 3].hist(train_df[train_df[metric] > 0][metric], bins = 10)


# In[ ]:




Looks like most of our data is not that toxic...
# Next we look at the correlation of increasing the certainty of a label and the certainty that label is toxic.

# In[ ]:


identities = ['asian', 'atheist', 'bisexual',
    'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',
    'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',
    'jewish', 'latino', 'male', 'muslim', 'other_disability',
    'other_gender', 'other_race_or_ethnicity', 'other_religion',
    'other_sexual_orientation', 'physical_disability',
    'psychiatric_or_mental_illness', 'transgender', 'white']


# In[ ]:


train_df_without_na = train_df.dropna()


# In[ ]:


fig, ax = plt.subplots(4, 6, sharex='col', sharey='row', figsize=(20, 10))
for i, identity in enumerate(identities):
    ax[i // 6, i % 6].set_title(identity)
    train_df_without_na.groupby(pd.cut(train_df_without_na[identity], np.arange(0, 1.0, 0.1))).target.mean().plot(ax = ax[i // 6, i % 6])


# It does looks like there are some trends between some label confidence scores and toxicity scores. Below we do the exact same thing but we add standard deviation values in blue.

# In[ ]:


fig, ax = plt.subplots(4, 6, sharex='col', sharey='row', figsize=(20, 10))
for i, identity in enumerate(identities):
    ax[i // 6, i % 6].set_title(identity)
    df_temp = train_df_without_na.groupby(pd.cut(train_df_without_na[identity], np.arange(0, 1.0, 0.1))).target
    df_temp = pd.DataFrame({'mean_values': df_temp.mean(),'std_values': df_temp.std()})
    
    x = df_temp.reset_index().index
    mean = df_temp.mean_values.values
    std = df_temp.std_values.values

    ax[i // 6, i % 6].fill_between(x, mean + std, mean - std, color='blue', alpha=0.5)
    ax[i // 6, i % 6].plot(x, mean, color='black');

