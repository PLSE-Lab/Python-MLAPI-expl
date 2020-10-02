#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Disclaimer: I'm a Kaggle beginner, and this may not necessarily be a good way to treat categorical variables.
# If you have any suggestions or corrections, please let me know.

get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed
from pylab import rcParams
rcParams['figure.figsize'] = 15, 7


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


target = train['target']
train.drop('target', axis=1, inplace=True)


# ## Non null columns

# In[ ]:


print(train.columns[np.where(train.isnull().any().values == False)])


# These are the columns with no null values, which seems to make a very small part of the dataset, so imputation will be required for algorithms that don't handle null values automatically.
# 
# ## Non numeric columns

# In[ ]:


non_numeric_columns = list(set(train.columns) - set(train.select_dtypes(include=[np.number]).columns))
print(non_numeric_columns)


# The columns above are the ones whose values are not numeric (probably strings), so they must be encoded in some sorts of numeric form.
# 
# ## Count of non numeric values for each feature

# In[ ]:


D = {}
for column in non_numeric_columns:
    col = getattr(train, column)
    D[column] = len(col.value_counts())

plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.show()


# ## What's that spike?
# 
# It's actually obvious from the graph if it's big enough, but I'll just leave this here.

# In[ ]:


print(max(D.keys(), key=lambda key: D[key]))


# ## What's v22?

# In[ ]:


train.v22[target == 1].value_counts()[:50].plot()
train.v22[target == 0].value_counts()[:50].plot()


# In[ ]:


target.value_counts()


# They seem quite correlated to be honest, and the discrepancy can be explained by the class difference in `target`. I'll just discard this value
# 
# While posting this, I realised that a better way is to encode all variables and actually compute relationships with Pearson's correlation or mutual information (for nonlinear relationships). Moreso, the graphs are not very good indicators, as the xticks are the ones used for the second plot, and they might differ for the two. Luckily, I don't think it mattered too much in this example.

# In[ ]:


train.drop('v22', axis=1, inplace=True)
test.drop('v22', axis=1, inplace=True)

del D['v22']


# ## Check the new non numeric variables

# In[ ]:


plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.show()


# ## Examine v56

# In[ ]:


train.v56[target == 1].value_counts()[:100].plot()
train.v56[target == 0].value_counts()[:100].plot()


# This may be silly, but it looks like only the first jump is significantly different between the 2, so I will use one-hot encoding only for the 2 most common variables for each case.

# In[ ]:


def get_common_values(df, var, number):
    return (set(df[var][target == 1].value_counts().keys()[:number]) |
                   set(df[var][target == 0].value_counts().keys()[:number]))

common_v56 = get_common_values(train, 'v56', 2)
print(common_v56)


# In[ ]:


def encode_value(df, val, col):
    positives = df[df[col] == val][col].index
    df["{}_{}".format(col, val)] = [1 if i in positives else 0 for i in range(len(df))]
    return df


# In[ ]:


for value in common_v56:
    train = encode_value(train, value, "v56")
    test = encode_value(test, value, "v56")


# In[ ]:


train.drop('v56', axis=1, inplace=True)
test.drop('v56', axis=1, inplace=True)

del D['v56']


# ## One more categorical variable

# In[ ]:


plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.show()


# ## v125

# In[ ]:


train.v125[target == 1].value_counts()[:100].plot()
train.v125[target == 0].value_counts()[:100].plot()


# This one looks a lot less correlated in my opinion, especially for top values.
# 
# ## Zooming in small a section
# 
# Just "zooming" in a bit to see if there's a trend among lower values that can't be observed due to the scale.

# In[ ]:


train.v125[target == 1].value_counts()[50:70].plot()
train.v125[target == 0].value_counts()[50:70].plot()


# For this one, I will encode the first 10 most common variables from each class.

# In[ ]:


common_v125 = get_common_values(train, 'v125', 10)
print(common_v125)


# In[ ]:


for value in common_v125:
    train = encode_value(train, value, "v125")
    test = encode_value(test, value, "v125")


# In[ ]:


print(train.shape)
print(test.shape)


# ## Conclusion
# 
# This may not be the best way, or even a good way to do it, but it's the path I happened to take for this exploration. The other variables have fewer categories, and I might just one-hot all of them or look into different methods. If anyone has any corrections or advice, I'd appreciate if you let me know, as I'm doing this primarly to learn.

# In[ ]:




