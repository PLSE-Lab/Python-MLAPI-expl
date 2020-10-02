#!/usr/bin/env python
# coding: utf-8

# ### It is my first kernel in python. I wanted to try my hand on a nice and small dataset, and this dataset was just perfect for me. Let's get started.

# In[ ]:


import pandas as pd
import numpy as np
import datetime 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


choc_df = pd.read_csv("../input/flavors_of_cacao.csv")


# In[ ]:


choc_df.head()


# ### Checking Ratings' Distribution

# In[ ]:


choc_df['Rating'].plot.hist(bins = 16)


# Year-Wise Ratings

# In[ ]:


g = sns.FacetGrid(choc_df, col = 'Review\nDate', col_wrap=4, size=2.5)
g = g.map(plt.hist, "Rating")


# ### Checking Missing Values

# In[ ]:


choc_df.isnull().sum()


# Columns 'Bean Type' and 'Broad Bean Origin' both have one NaN value. Dropping entire rows with NaN.

# In[ ]:


choc_df.dropna(axis=0, how='any', inplace=True)


# ### Converting Cocoa %age from string to numeric

# In[ ]:


choc_df['Cocoa\nPercent'] = choc_df['Cocoa\nPercent'].replace("%","", regex = True)
choc_df['Cocoa\nPercent'] = choc_df['Cocoa\nPercent'].apply(pd.to_numeric, errors = 'coerce')


# In[ ]:


choc_df.head()


# ### Distribution of Data

# In[ ]:


sns.jointplot(x = "Rating", y = "Cocoa\nPercent", data = choc_df)


# Year-Wise Ratings & Cocoa %

# In[ ]:


p = sns.FacetGrid(choc_df, col = 'Review\nDate', col_wrap=4, size=2.5)
p = p.map(plt.scatter, "Rating", "Cocoa\nPercent")


# Year-Wise Cocoa %

# In[ ]:


sns.boxplot(x = 'Review\nDate', y = 'Cocoa\nPercent', data = choc_df)


# ## Work is still in progress. Let me know if you have any suggestions. Take care.

# In[ ]:




