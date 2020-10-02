#!/usr/bin/env python
# coding: utf-8

# # Pandas data tricks and baseline
# 
# Sometimes it is nice to turn the data you recieve into formats that are more easily fed to your models of interest. Let's not worry about the actual imaging data and just make it easier to work with the labels.
# 
# TL;DR: Just use these functions below do convert your sample_submission or train_csv dataframes:

# In[ ]:


import pandas as pd


# In[ ]:


def rsna_to_pivot(df, sub_type_name='HemType'):
    """Convert RSNA data frame to pivoted table with
    each subtype as a binary encoded column."""
    df2 = df.copy()
    ids, sub_types = zip(*df['ID'].str.rsplit('_', n=1).values)
    df2.loc[:, 'ID'] = ids
    df2.loc[:, sub_type_name] = sub_types
    return df2.pivot(index='ID', columns=sub_type_name, values='Label')

def pivot_to_rsna(df, sub_type_name='HemType'):
    """Converted pivoted table back to RSNA spec for submission."""
    df2 = df.copy()
    df2 = df2.reset_index()
    unpivot_vars = df2.columns[1:]
    df2 = pd.melt(df2, id_vars='ID', value_vars=unpivot_vars, var_name=sub_type_name, value_name='Label')
    df2['ID'] = df2['ID'].str.cat(df2[sub_type_name], sep='_')
    df2.drop(columns=sub_type_name, inplace=True)
    return df2


# In[ ]:


sample_sub = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')
train = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
#First step is to get rid of duplicated entries. Luckily they are consistent.


# In[ ]:


# Let's check that that is true just to be sure.
# If there were any groups that were not consistent,
# the set of labels should be more than 0 and 1, therefore we are safe.
set(train.groupby('ID').mean()['Label'].values)


# In[ ]:


# Actually remove duplicates
train = train.groupby('ID').first().reset_index()
train.head()


# If we look at the data in the train CSV, we find that each image has an ID with a corresponding sub-type and a binary label. It isn't fun to have to work with classes buried in the ID. It would be much better to separate these out into lines where each line simply contained the ID and then having separate columns for each class contained at the end of ID, delineating whether or not that type of hemmorhage was present. Luckily the pandas package has some magical functionality to do this.

# In[ ]:


print(train.shape) # Notice all the rows
train.head()


# First we will separate out the class name from the ID by using an rsplit:

# In[ ]:


split_series = train['ID'].str.rsplit('_', n=1)
split_series.head()


# Now we have the split we need. We just now need to package that back into the original dataframe. This is easy, we just need to grab these and asign them as new columns:

# In[ ]:


ids, sub_types = zip(*train['ID'].str.rsplit('_', n=1).values)
train.loc[:, 'ID'] = ids
train.loc[:, 'HemType'] = sub_types # We are using HemType as our column name for our sub_types
train.head()


# The next line is the real magic. We can use a pivot to package everything up as we described.

# In[ ]:


train = train.pivot(index='ID', columns='HemType', values='Label')
print(train.shape)
train.head()


# In[ ]:


# Yay! That's what I like to see. Let's grab some stats real quick.
# We can save these for later.
train.mean()


# Of course though, if we can transform in this direction. We want to transform in the other. This backwards operation is called a melt, and we can do that just as easily. First we will clean up the index:

# In[ ]:


train = train.reset_index()


# Now we will do our melt, which will be roughly the inverse operation of what we just performed.

# In[ ]:


unpivot_vars = train.columns[1:] # Here we need the names of categories so we can push them back in the ID
train = pd.melt(train, id_vars='ID', value_vars=unpivot_vars, var_name='HemType', value_name='Label')
train.head()


# Almoost there, now we just need to convert ID and HemType into one column:

# In[ ]:


train['ID'] = train['ID'].str.cat(train['HemType'], sep='_')
train.drop(columns='HemType', inplace=True)
train.head()


# Wow look at that! We went from one representation and back again really easily. If you want to learn more about these manipulations of data, I found Hadley Wickham's Tidy Data paper to be extremely helpful. (https://vita.had.co.nz/papers/tidy-data.pdf) 
# 
# With all said, our code is really short and fits in these two functions:

# In[ ]:


# The only additions I"m adding is copying dataframes so we don't accidentally change data we want to keep.

def rsna_to_pivot(df, sub_type_name='HemType'):
    """Convert RSNA data frame to pivoted table with
    each subtype as a binary encoded column."""
    df2 = df.copy()
    ids, sub_types = zip(*df['ID'].str.rsplit('_', n=1).values)
    df2.loc[:, 'ID'] = ids
    df2.loc[:, sub_type_name] = sub_types
    return df2.pivot(index='ID', columns=sub_type_name, values='Label')

def pivot_to_rsna(df, sub_type_name='HemType'):
    """Converted pivoted table back to RSNA spec for submission."""
    df2 = df.copy()
    df2 = df2.reset_index()
    unpivot_vars = df2.columns[1:]
    df2 = pd.melt(df2, id_vars='ID', value_vars=unpivot_vars, var_name=sub_type_name, value_name='Label')
    df2['ID'] = df2['ID'].str.cat(df2[sub_type_name], sep='_')
    df2.drop(columns=sub_type_name, inplace=True)
    return df2


# # Averaged baseline
# 
# Now let's use these functions to create a dead simple baseline that we can use without waiting for all the CT data to unzip.
# 
# We will look at the training data, and just assign the probability of a sub_type of hemmorhage to be the frequency of each type of hemmorhage in the dataset.

# In[ ]:


# Just prep work we did before
sample_sub = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')
train = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
train = train.groupby('ID').first().reset_index()


# In[ ]:


# Easy. Abstraction makes life great.
train_pivot = rsna_to_pivot(train)
sample_sub_pivot = rsna_to_pivot(sample_sub)


# In[ ]:


train_pivot.head()


# In[ ]:


sample_sub_pivot.head()


# In[ ]:


# We did this before.
averages = train_pivot.mean()
averages


# In[ ]:


# Go through the averages and deliver them to the columns of the submission.
for label, value in averages.items():
    sample_sub_pivot.loc[:, label] = value


# In[ ]:


sample_sub_pivot.head()


# In[ ]:


# We pivoted and now let's melt this back on in.
submission = pivot_to_rsna(sample_sub_pivot)
submission.head()


# In[ ]:


# What's easier than this?
submission.to_csv('submission.csv', index=False)


# This is a really nice way to work with your data, not only for feeding data to a neural network, but also just to make it more expressive for your exploratory data analysis. Hope this helps!
