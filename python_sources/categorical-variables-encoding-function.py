#!/usr/bin/env python
# coding: utf-8

# # Study of categorical varibales
# 
# In several first kernels that appeared by now peopel use LabelEncoding to transform categorical features in the data into something that can be digested by a model. I suppose this was not done on purpose, but rather to have a quick first iteration
# 
# # UPDATE: There was a clarification on values corresponding to 'yes'/'no' labels
# In the Welcome discussion thread @Louis Tajerina clarified the occurance of the categorical `yes/no` values: https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359631.
# 
# As an outcome, I provide a very basic encoding function to encode categoricals into reasonable integer dummies. There are ideas in the notebook on how to improve it further with more advanced imputations.
# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    '''
    The function does not return, but transforms the input pd.DataFrame
    
    Encodes the Costa Rican Household Poverty Level data 
    following studies in https://www.kaggle.com/mlisovyi/categorical-variables-in-the-data
    and the insight from https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359631
    
    The following columns get transformed: edjefe, edjefa, dependency, idhogar
    The user most likely will simply drop idhogar completely (after calculating houshold-level aggregates)
    '''
    
    yes_no_map = {'no': 0, 'yes': 1}
    
    df['dependency'] = df['dependency'].replace(yes_no_map).astype(np.float32)
    
    df['edjefe'] = df['edjefe'].replace(yes_no_map).astype(np.float32)
    df['edjefa'] = df['edjefa'].replace(yes_no_map).astype(np.float32)
    
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])


# # Outdated
# 
# The purpose of this kernel is to have a look at those features and come up with a smarter encoding. The outline goes line this:
# - [Check categorical variables](#Check-categorical-variables)
# - [idhogar](#idhogar)
# - [edjefe and edjefa](#edjefe-and-edjefa)
#  - [Replace 'no' with 0 or NaN?](#Replace-'no'-with-0-or-NaN?)
#  - [How about 'yes' values?](#How-about-'yes'-values?)
# - [dependency](#dependency)
# 
# As an outcome, I provide a very basic encoding function to encode categoricals into reasonable integer dummies. There are ideas in the notebook on how to improve it further with more advanced imputations.

# # Prepare environment and read in the data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.info()


# Ok, so there are 5 columns with categorical data. Let's have a quick look at what are those. Most first kernel use label encoding. And there was an explicit mention in [this kernel by Ishaan Jain](https://www.kaggle.com/ishaan45/eda-for-a-good-cause), that OHE leads to an explosion in the number of features. Let's understand why...
# 
# ## Check categorical variables

# In[ ]:


cat_cols = train.select_dtypes(include=['object', 'category'])


# In[ ]:


cat_cols.head(10)


# Let's study them one-by-one
# # idhogar
# This one is clear - that's a unique identifier for a household. Most likely one does not want to use it in training, but rather to group on it to get accumulated household statistics. Therefore, it does not matter much what you do with it- you can keep it as a string, you can label-encode it. But the set of unique values is huge and this will lead to the large number of OHE feature observed by @Ishaan

# # edjefe and edjefa
# Let's start with these two, as they are easier. A brief reminder:
# > `edjefe`, years of education of male head of household
# 
# > `edjefa`, years of education of female head of household
# 

# In[ ]:


def plot_value_counts(series, title=None):
    '''
    Plot distribution of values counts in a pd.Series
    '''
    _ = plt.figure(figsize=(12,6))
    z = series.value_counts()
    sns.barplot(x=z, y=z.index)
    _ = plt.title(title)
    
plot_value_counts(train['edjefe'], 'Value counts of edjefe')
plot_value_counts(train['edjefa'], 'Value counts of edjefa')


# So, both variables are basically a number with precence of `'yes'` and `'no'` classes. Let's figure out what do those mean...

# In[ ]:


# Family member counts
hogar_df = train[[f_ for f_ in train.columns if f_.startswith('hogar_')]]
# Family member type of this person
parentesco_df = train[[f_ for f_ in train.columns if f_.startswith('parentesco')]]

# Family status dataset
family_status = pd.concat([cat_cols, train[['female', 'male', 'parentesco1', 'meaneduc']], hogar_df], axis =1)


# In[ ]:


family_status.head(5)


# The observation at this stage: **very often `edjefe/edjefa` variable is encoded to `'no'` in the case, when household head  is of opposite gender.**
# 
# For example, if the household head is male, then `edjefa == 'no'` (person with `Id == 'ID_ec05b1a7b'` is the head and is a male):

# In[ ]:


# this is a hand-picked example to illustrate the point
family_status.query('idhogar == "2b58d945f"')


# For example, if the household head is female, then `edjefe == 'no'` (person with `Id == 'ID_c51c0afa1'` is the head and is a male):

# In[ ]:


# this is a hand-picked example to illustrate the point
family_status.query('idhogar == "200099351"')


# ## Replace 'no' with 0 or NaN?
# In most cases, yes. But there are exceptions, when both `edjefe` and `edjefa`  are equal to `'no'`:

# In[ ]:


family_status.query('(edjefe=="no" & edjefa=="no")').head()


# What are those? I assume, in this case the household head either does not have education or did not provide this info. 
# 
# One might want to differentiate such `'no'` values from the previously discussed type, as the meaning is different. In such case, the  suggestion would be to encode for the head of the household the value of 0 and -1 for the `'no'` values of the type that was discussed first. 
# 
# But **if one simply fills `'no'` with 0, there will be no critical change in the behaviour**.
# 
# In either case, the advantage that one gains over simple label encoding is that one preserves the internal order of classes, while label encoding could encode `8` as `8` and `10` as `1`

# ## How about 'yes' values?

# In[ ]:


family_status.query('(edjefe=="yes") | (edjefa=="yes")').head(20)


# So it seems that `'yes'` means *the head has education, but the years are not specified*. In such case, the suggestion can be either to encode a dummy value (1?) or better the mean within a category. 
# 
# The alternative would be to try to deduce the age from the `meaneduc` and `SQBmeaned`. This is trivial for houshoulds of a single person, possible fo 2 adult people (up to ambiguity, which of 2 values corresponds to the head) and impossible(?) for more than2 adults in the houshold.
# 
# One can also use `instlevelX` for imputation
# 
# # dependency

# In[ ]:


plot_value_counts(train['dependency'], 'Value counts of dependency')


# This variable is so far unclear to me. The `'SQBdependency'` seems to suggest that providers of the data assumed that `'no' == 0` and `'yes' == 1`, is a reasonable imputation and I see no reason to argue with them about it :)

# In[ ]:


train[['dependency', 'SQBdependency']].head(20)


# # UNDER CONSTRUCTION!

# In[ ]:





# In[ ]:




