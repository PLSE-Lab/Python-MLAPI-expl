#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_set = pd.read_csv('../input/movie_metadata.csv')
data_set


# Film Gross
# ----------
# 
# One of the metrics to regard a film as successful is its film gross. The film gross is the money the film makes when released onto theaters. Revenue alone isn't sufficient, however, in analyzing in depth how the money was generated. This research is about analyzing film gross through correlating other factors. The data set used for the research is the IMDB 500 Movies Dataset. The following research questions are as follows:
# 
# 1. What is the film duration range that generates the most gross?
# 
# 2. Does content/maturity rating affect film gross?
# 
# 3. What is the correlation between IMDB ratings and film gross?

# What is the film duration range that generates the most gross?
# -----------

# In[ ]:


# Get all values for film duration
duration_set = data_set['duration']
duration_set


# In[ ]:


# Get all values for film gross
gross_set = data_set['gross']
gross_set


# In[ ]:


plt.plot(duration_set, gross_set, 'bo')
plt.ylabel('Film gross (in powers of 10)')
plt.xlabel('Film duration (in minutes)')
plt.show()


# **Insights:**
# 
# - Most of the films in the data set lasts between 101 to 150 minutes; very condensed in this region
# 
# - The highest gross, however, is on the 151-200 minute range at nearly a hundred million
# 
# - Long films may be experimental or independently-made films
# 
# - Short films are short films; no theatre exposure
# 
# **Conclusion**
# 
# - Most ideal duration for a film to make money is between 101 to 150 minutes
# 
# - Too short (0 to 50 minutes -- short films) or too long (200+ minutes) won't generate as much

# ## Does content/maturity rating affect film gross? ##

# In[ ]:


pg_set = data_set[data_set['content_rating'].isin(['PG'])]
pg_set


# In[ ]:


pg13_set = data_set[data_set['content_rating'].isin(['PG-13'])]
pg13_set


# In[ ]:


r_set = data_set[data_set['content_rating'].isin(['R'])]
r_set


# In[ ]:


pg_gross = pg_set['gross']
pg_grossIQR = pg_gross.quantile(.75) - pg_gross.quantile(.25)
pg_grossIQR


# In[ ]:


pg_gross.describe()


# In[ ]:


pg_gross.mean()


# In[ ]:


pg13_gross = pg13_set['gross']
pg13_grossIQR = pg13_gross.quantile(.75) - pg13_gross.quantile(.25)
pg13_grossIQR


# In[ ]:


pg13_gross.describe()


# In[ ]:


pg13_gross.mean()


# In[ ]:


r_gross = r_set['gross']
r_grossIQR = r_gross.quantile(.75) - r_gross.quantile(.25)
r_grossIQR


# In[ ]:


r_gross.describe()


# In[ ]:


r_gross.mean()


# **Insights:**
# 
# - PG-rated films on the data set are the fewest among the content/maturity ratings considered
# 
# - R-rated films are the most inside the data-set
# 
# - R-rated films, however, generated the least based on mean revenue
# 
# - R-rated films also has greater variability based on IQR; PG-rated the least
# 
# 
# **Conclusion**
# 
# - Content/Maturity Rating does affect the g

# In[ ]:





# ## What is the correlation between IMDB ratings and film gross? ##

# In[ ]:


ratings_arr = {'IMDB Rating': data_set['imdb_score'], 'Film Gross': data_set['gross']}
ratings_df = pd.DataFrame(ratings_arr)
ratings_df.corr('pearson')


# In[ ]:


#plt.plot(duration_set, gross_set, 'bo')
plt.scatter(data_set['imdb_score'], gross_set)
plt.ylabel('Film gross (in powers of 10)')
plt.xlabel('IMDB Rating')
plt.show()


# **Conclusion**
# 
# - Based from the computed correlation, there is a weak correlation between IMDB Rating and Film Gross.
# - However, looking at the graph, there is an upward trend especially if the film has an IMDB rating of 7-8.
