#!/usr/bin/env python
# coding: utf-8

# # Starter Kit - Preprocessing

# In this notebook, I detailed the preprocessing workflow I develop for preparing data prior models training.
# Later I will prepare a new notebook only about under/oversampling methods.
# 
# If you just want the code, you can check the last code block in this notebook.

# In[ ]:


import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load data

# In[ ]:


df = pd.read_csv('../input/Kaggle_Training_Dataset_v2.csv')
df.info()


# In[ ]:


df.isnull().sum()


# ### Imputing missing values
# 
# *lead_time* has several missing values. We use Imputer to fill this values with a new value (usually mean or median). It also appears to has a empty row (all values are missing). We will drop it.

# In[ ]:


from sklearn.preprocessing import Imputer
df['lead_time'] = Imputer(strategy='median').fit_transform(
                                df['lead_time'].values.reshape(-1, 1))
df = df.dropna()
df.isnull().sum()


# Supplier's performance (*perf_6_month_avg* and *perf_12_month_avg*) presents a scaled distribution, if the -99 missing value flag is dropped:
# 
# We could also replace missing values using the **Imputer**, this can improve performance in neural_networks for example...
# But in decision trees and its ensembles, it really do not impacte, since these models are robust against outliers.

# In[ ]:


from sklearn.preprocessing import Imputer
for col in ['perf_6_month_avg', 'perf_12_month_avg']:
    df[col] = Imputer(missing_values=-99).fit_transform(df[col].values.reshape(-1, 1))
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
sns.distplot(df["perf_6_month_avg"], ax=ax1)
sns.distplot(df["perf_12_month_avg"], ax=ax2)
plt.show()


# ### Convert to binaries
# 
# Now let's take a look into the first 5 rows of the dataset, some features has values "Yes/No".

# In[ ]:


df.head()


# In[ ]:


binaries = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
               'stop_auto_buy', 'rev_stop', 'went_on_backorder']
for col in binaries:
        df[col] = (df[col] == 'Yes').astype(int)
df[binaries].head()


# ### Visualization
# 
# Looking better... Now that we have only numeric attributes, let's take a peek into their distribution using some basics histograms:

# In[ ]:


df.hist(figsize=(12,12), alpha=0.8, grid=False)
plt.show()


# ### Normalization
# 
# Not too many bins yet, right? 
# 
# A first insight we can explore is that quantities are in different greatness levels, according to the item. For example, it is probably more usual having lower quantities of some determined items with lower prices (such as screws, stamps), then higher-priced ones (such as televisions, fridges, or whatever).
# A simple way to solve this particular issue, is normalize each part by row:

# In[ ]:


qty_related = ['national_inv', 'in_transit_qty', 'forecast_3_month', 
                   'forecast_6_month', 'forecast_9_month', 'min_bank',
                   'local_bo_qty', 'pieces_past_due', 'sales_1_month', 'sales_3_month', 
                   'sales_6_month', 'sales_9_month',]
from sklearn.preprocessing import normalize
df[qty_related] = normalize(df[qty_related], axis=1)
df[qty_related].hist(figsize=(10,11), alpha=0.7, color='orange', grid=False)
plt.show()


# Some normal distributions are starting to shows off, right? 
# Note that we used just normalization (which forces standard deviation = 1), not scaling, which also forces mean equal to 0. This is because we want to keep some important informations about the part (for example, if inventory is lower than 0, we can correct that in preprocessing, or we will be misleading our models).
# 
# ### Obsolete parts
# 
# The next insight we can explore here is that there are so many parts with 0 (no past sales or forecast).
# 
# Actually, this is quite common in **Inventory Management**: those are called "obsolete parts", materials that has no transactions in the last months, or forecast. The risk of such items going into backorder is probability much lower when compared to active items.
# 
# We will select and plot only active items, based on the following rule: *forecast_3_month > 0* and *sales_9_month > 0*.
# 

# In[ ]:


actives = df.loc[(df["forecast_3_month"]>0)&(df["sales_9_month"]>0)]
actives[qty_related].hist(figsize=(10,11), alpha=0.7, color='green', grid=False)
plt.show()


# Can you see some gaussians?
# 
# But now let's check what we lost from the original dataset by doing it:

# In[ ]:


from collections import Counter
print('Original dataset shape {}'.format(Counter(df.went_on_backorder)))
print('Reduced dataset shape {}'.format(Counter(actives.went_on_backorder)))
print('Class 0 reduction: %.2f%%' % (100*(1 - Counter(actives.went_on_backorder)[0]/Counter(df.went_on_backorder)[0])))
print('Class 1 reduction: %.2f%%' % (100*(1 - Counter(actives.went_on_backorder)[1]/Counter(df.went_on_backorder)[1])))


# This approach reduced the original dominant classes by 75%, while just loosing 25% of the class 1.
# 
# But we could use a less aggressive approach, adopting the rule *forecast_3_month > 0* or *sales_9_month > 0*:

# In[ ]:


actives2 = df.loc[(df["forecast_3_month"]>0)|(df["sales_9_month"]>0)]
from collections import Counter
print('Original dataset shape {}'.format(Counter(df.went_on_backorder)))
print('Reduced dataset shape {}'.format(Counter(actives2.went_on_backorder)))
print('Class 0 reduction: %.2f%%' % (100*(1 - Counter(actives2.went_on_backorder)[0]/Counter(df.went_on_backorder)[0])))
print('Class 1 reduction: %.2f%%' % (100*(1 - Counter(actives2.went_on_backorder)[1]/Counter(df.went_on_backorder)[1])))


# Now the reduction in dominant class is 15x higher than in the class 1 (which presents a high cost for us, since we have just a few of them).
# 
# Using this may save some computing time and also presenting a cleaner dataset for the predictive model.
# You may use this rule or not in our model, it is really up to you.

# ### Now, a function that does it all
# 
# You can just copy this cell and use it when reading training or test sets:

# In[ ]:


def process(df):
    # Imput missing lines and drop line with problem
    from sklearn.preprocessing import Imputer
    df['lead_time'] = Imputer(strategy='median').fit_transform(
                                    df['lead_time'].values.reshape(-1, 1))
    df = df.dropna()
    for col in ['perf_6_month_avg', 'perf_12_month_avg']:
        df[col] = Imputer(missing_values=-99).fit_transform(df[col].values.reshape(-1, 1))
    # Convert to binaries
    for col in ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
               'stop_auto_buy', 'rev_stop', 'went_on_backorder']:
        df[col] = (df[col] == 'Yes').astype(int)
    # Normalization    
    from sklearn.preprocessing import normalize
    qty_related = ['national_inv', 'in_transit_qty', 'forecast_3_month', 
                   'forecast_6_month', 'forecast_9_month', 'min_bank',
                   'local_bo_qty', 'pieces_past_due', 'sales_1_month', 'sales_3_month', 
                   'sales_6_month', 'sales_9_month',]
    df[qty_related] = normalize(df[qty_related], axis=1)
    # Obsolete parts - optional
    #df = df.loc[(df["forecast_3_month"]>0)|(df["sales_9_month"]>0)]
    return df

df = process(pd.read_csv('../input/Kaggle_Training_Dataset_v2.csv'))
df.info()


# If you have any other ideas for improving this dataset preprocessing, leave a comment below:
# 
# []'s
