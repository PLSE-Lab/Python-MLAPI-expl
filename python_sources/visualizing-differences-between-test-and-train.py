#!/usr/bin/env python
# coding: utf-8

# ## Visualization of Statistical Differences Between Test and Train Datasets

# A visual inspection of the test and train dataset values reveals structure in the number formats. Many numbers are rounded to a few significant figures, and some are rounded to exactly two decimal places. In the test set there is a large block of rows with high decimal precision, not seen in the train dataset.
# <p>
# Statistical analysis reveals further structure around the relative proportion of unique values. Plotting the count of nonzero values vs the number of unique nonzero values reveals interesting structure in both the test and train datasets that are unique to each and not present in the other.
# <p>
# Finally, there is a significant difference in how sparse the test and train datasets are. Also, within the test dataset, there is a significant difference in sparsity between rows that have all unique values vs rows that do not.  For test, the rows with all unique values have a sparsity of 0.70%, and represent 56% of the data. The remaining test data has a sparsity of 2.3%, much closer to the 3.1% sparsity of the train data.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pd.__version__


# In[ ]:


# read datasets, force object data type to preserve number formatting
train = pd.read_csv('../input/train.csv', dtype=object)
train.shape


# In[ ]:


test = pd.read_csv('../input/test.csv', dtype=object)
test.shape


# ### Examine Numeric Format

# In[ ]:


# the most common values have a single significant figure for target
train.target.value_counts()


# In[ ]:


# remove ID and target columns, examine population of remaining values
# note there are two formats for zero, some values do not have a trailing ".0"
train.iloc[:,2:].stack().value_counts()


# In[ ]:


# remove ID column, examine population of values
# test data has high precision values not present in train
test.iloc[:,1:].stack().value_counts()


# The test and train dataset values have significant differences in the populations of number formats. Train data has many numbers that are rounded to a few significant figures, and some are rounded to exactly two decimal places. Test data has a large block of rows with high decimal precision, not seen in the train dataset.

# ### Examine Sparsity of Test and Train Datasets

# The datasets are obviously sparse, but are they equally sparse?

# In[ ]:


# flatten dataframe into a series
train_series = pd.Series(train.iloc[:,2:].astype(np.float64).values.flatten())


# In[ ]:


train_series.shape


# In[ ]:


# how many nonzero values?
train_series[train_series>0].shape[0]


# In[ ]:


# how sparse is the train dataset? 3.1%
train_series[train_series>0].shape[0]/train_series.shape[0]


# In[ ]:


test_series = pd.Series(test.iloc[:,1:].astype(np.float64).values.flatten())


# In[ ]:


test_series.shape


# In[ ]:


# how sparse is the test dataset? 1.4%, significantly less than train
test_series[test_series>0].shape[0]/test_series.shape[0]


# In[ ]:


# clean up to fit this notebook into Kaggle's 17 GB limit
del train_series
del test_series


#  There is a significant sparcity difference between test and train, 1.4% vs 3.1%.

# ### Visualizing Dataset Count vs Unique Differences

# In[ ]:


# some test dataset statistics, look at nonzero values only
train_stats = train.ID.to_frame()
train_stats['count']  = train.iloc[:,2:].astype(np.float64).replace(0.0, np.nan).count(axis=1)
train_stats['unique'] = train.iloc[:,2:].astype(np.float64).replace(0.0, np.nan).nunique(axis=1)


# In[ ]:


# some test dataset statistics, look at nonzero values only
test_stats = test.ID.to_frame()
test_stats['count']  = test.iloc[:,1:].astype(np.float64).replace(0.0, np.nan).count(axis=1)
test_stats['unique'] = test.iloc[:,1:].astype(np.float64).replace(0.0, np.nan).nunique(axis=1)


# In[ ]:


# there are unusual clusters, a series of points with a negative slope, not seen in the test dataset
train_stats.plot.scatter(x='count', y='unique', figsize=(12,10), alpha=0.3)


# In[ ]:


# examine the lower left more closely, there are more of these clusters
train_stats[train_stats['count']<900].plot.scatter(x='count', y='unique', figsize=(12,10), alpha=0.3)


# In[ ]:


test_stats.plot.scatter(x='count', y='unique', figsize=(12,10), alpha=0.3)


# In[ ]:


# this cluster stands out in the test dataset
test_stats[test_stats['count'] == test_stats['unique']].plot.scatter(x='count', y='unique', alpha=0.3)


# Visualizing the count of unique values vs the count of nonzero values for each row reveals structural differences between the two datasets. The train dataset has clusters that show an interesting negative slope. The test dataset has a large number of rows (56%) that have a unique property, the unique values per row equal the nonzero count of values, not present in the test dataset.

# ### Exploring Sparsity in the Test Dataset

# Finally, let's test if the sparsity of this unusual count=unique set is similar to the training dataset sparsity.

# In[ ]:


# how many 'customers' are in this count=unique set? 
test_stats[test_stats['count'] == test_stats['unique']].count()['ID']


# In[ ]:


# how many 'customers' are not in this count=unique set? 
test_stats[test_stats['count'] != test_stats['unique']].count()['ID']


# In[ ]:


# test the sparsity of the values in this count=unique set
test_series = pd.Series(test.iloc[:,1:].astype(np.float64).values.flatten())


# In[ ]:


test[test.ID.isin(test_stats[test_stats['count'] == test_stats['unique']]['ID'])].shape


# In[ ]:


test_series_count_unique = pd.Series(test[test.ID.isin(test_stats[test_stats['count'] == test_stats['unique']]['ID'])].iloc[:,1:].astype(np.float64).values.flatten())


# In[ ]:


# this count=unique set is very sparse, 0.70%
test_series_count_unique[test_series_count_unique>0].shape[0]/test_series_count_unique.shape[0]


# In[ ]:


test[test.ID.isin(test_stats[test_stats['count'] != test_stats['unique']]['ID'])].shape


# In[ ]:


test_series_count_not_unique = pd.Series(test[test.ID.isin(test_stats[test_stats['count'] != test_stats['unique']]['ID'])].iloc[:,1:].astype(np.float64).values.flatten())


# In[ ]:


# this count!=unique set less sparse, closer to the 3.1% of the train dataset
test_series_count_not_unique[test_series_count_not_unique>0].shape[0]/test_series_count_not_unique.shape[0]


# The count=unique subset of the test dataset represents 56% of total test rows and has very low sparsity, 0.70%. The remaining rows have a sparsity of 2.3%, much closer to the 3.1% seen in the train dataset. 

# In[ ]:




