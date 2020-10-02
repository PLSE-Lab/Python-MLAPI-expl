#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# We will briefly illustrate how to calculate *Weight of Evidence* and *Information Value* in Python.
# 
# **Warning**: This is going to be super hacky and not optimal in the slightest. It is purely for illustration. 
# 
# If anyone cares to optimize with NumPy/Vectorization and replacing the for-loop with some `.apply()` method, I would be most appreciative. I'll do the same if I get to it! :)
# 
# * Quick shout-out to the inspiration for this kernel: https://www.kaggle.com/pranav84/talkingdata-with-breaking-bad-feature-engg
# 
# Here we go...

# # Import Packages

# In[1]:


import numpy as np
import pandas as pd


# # Load the Data

# In[15]:


# Load subset of the training data
X_train = pd.read_csv('../input/train.csv', skiprows=range(1,1000000), nrows=1000000, parse_dates=['click_time'])

# Show the head of the table
X_train.head()


# # Brief Feature Engineering: *ip_app_nextClick*
# 
# We'll quickly create a feature for illustrating our Information Value tests. 
# 
# I'm going to borrow some code from this kernel: https://www.kaggle.com/nanomathias/feature-engineering-importance-testing

# In[16]:


GROUP_BY_NEXT_CLICKS = [{'groupby': ['ip', 'app']}]

# Calculate the time to next click for each group
for spec in GROUP_BY_NEXT_CLICKS:
    
    # Name of new feature
    new_feature = '{}_nextClick'.format('_'.join(spec['groupby']))    
    
    # Unique list of features to select
    all_features = spec['groupby'] + ['click_time']
    
    # Run calculation
    print(f">> Grouping by {spec['groupby']}, and saving time to next click in: {new_feature}")
    X_train[new_feature] = X_train[all_features].groupby(spec['groupby']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    
X_train.head()


# # Calculate Information Value
# 
# Here is the *super hacky* function that does all of the work:

# In[17]:


# Calculate information value
def calc_iv(df, feature, target, pr=False):
    """
    Set pr=True to enable printing of output.
    
    Output: 
      * iv: float,
      * data: pandas.DataFrame
    """

    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())


    iv = data['IV'].sum()
    # print(iv)

    return iv, data


# This is way too expensive to run on `X_train`. So we'll run it on a sample of the data:

# In[18]:


df = X_train.sample(7777).copy()

iv, data = calc_iv(df, 'ip_app_nextClick', 'is_attributed')


# This falls under tha category of "too good to be true" typically:

# In[19]:


iv


# In[20]:


data.head()


# Where were the hits?

# In[21]:


data[data['Bad Rate'] > 0.0]


# Thanks! Let me know your thoughts.
# 
# *-BA*
