#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


len(df) # number of samples


# In[ ]:


# number of categories. http://devdocs.io/pandas~0.22/generated/pandas.dataframe.nunique
df.nunique()


# In[ ]:


# number of samples per category.
# Found the cookbook at http://devdocs.io/pandas~0.22/tutorials
# http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/blob/v0.1/cookbook/Chapter%204%20-%20Find%20out%20on%20which%20weekday%20people%20bike%20the%20most%20with%20groupby%20and%20aggregate.ipynb#4.2-Adding-up-the-cyclists-by-weekday
# use 'count' aggregate function https://stackoverflow.com/a/19385591/630752
grouped = df.groupby('landmark_id').aggregate('count')
grouped # note how we now have the same number of rows as the number of categories


# In[ ]:


# visualize distribution. http://devdocs.io/pandas~0.22/generated/pandas.series.sort_values
sorted_counts = grouped['id'].sort_values(ascending=False)
sorted_counts


# In[ ]:


plt.hist(sorted_counts) # histogram


# In[ ]:


sorted_counts.mean() # mean of series. http://devdocs.io/pandas~0.22/generated/pandas.series.mean


# In[ ]:


sorted_counts.median() # median http://devdocs.io/pandas~0.22/generated/pandas.series.median


# Mean is larger than median, so the distribution has samples well above the median (14) dragging the mean (81) up.

# In[ ]:


sorted_counts_no_outliers = [x for x in sorted_counts if x < 100] # remove outliers
sorted_counts_outliers = [x for x in sorted_counts if x >= 100] # outliers


# In[ ]:


# zoom in histogram. http://devdocs.io/matplotlib~2.1/_as_gen/matplotlib.pyplot.hist
plt.hist(sorted_counts_no_outliers)


# In[ ]:


# plot outliers starting from 100. http://devdocs.io/matplotlib~2.1/_as_gen/matplotlib.pyplot.hist
plt.hist(sorted_counts_outliers, range=(100,2000))


# The number of samples in the dataset is exhibiting a power law distribution and self-symmetry at different scales.

# Next steps:
# * generate grid of 10 categories with most samples, median # of samples and lowest number of samples, 5 images per category (row)
# * plot box and whisker of distribution of samples per category
# * build keras sequential model on data https://keras.io/getting-started/sequential-model-guide/
# * generate submission.csv and get on leaderboard
# * look into image2vec pretrained models and apply it on categories with few samples. See https://arxiv.org/pdf/1507.08818.pdf
