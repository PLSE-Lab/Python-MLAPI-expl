#!/usr/bin/env python
# coding: utf-8

# # Explore Dividends
# This is a quick example notebook of how to get a look and feel for the dividend dataset. It was greatly inspired by Samuel's [exploratory notebook](https://www.kaggle.com/samlac79/beer-recipe-exploratory-analysis) which I really liked.    
# Ask any questions in the comments below or on twitter [@jonnylangefeld](https://twitter.com/jonnylangefeld).
# 
# These are the libraries we are going to use:

# In[ ]:


import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
pd.set_option('display.max_columns',None) #To see all the columns since there are a lot of them


# ### Import the Data
# Let's import the data and have a quick look at a few rows, that are mostly filled. `simple_name`and `div_growth` are often `NA`, so I selected some rows that are even filled for these fields (Don't worry, `div_growth` is empty a lot of times, but we have an alternative, I'll get to that in a second).    
# I use the pickle file, as most of the data has already been classified in the DataFrame stored as a pickle.

# In[ ]:


instruments = pd.read_pickle('../input/instruments.p')


# In[ ]:


instruments[instruments.simple_name.notna() & instruments.div_growth.notna()].head()


# Let's see some general information about the dataset.

# In[ ]:


print(instruments.info(verbose=False))


# We have 3651 rows and 60 columns. It is a quite small dataset only using 1.5 MB of memory.
# 
# Next thing we are interested in is how many missing values do we have. Samuel [used](https://www.kaggle.com/samlac79/beer-recipe-exploratory-analysis) the missingno library for that, which I found quite helpful. I sampled 500 rows and orderd by completenes:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
msno.matrix(instruments.sample(500), labels=True, sort='descending')


# Turns out that we have quite some columns that aren't filled fully. That is pretty easy to explain if we look [at this notebook](https://github.com/jonnylangefeld/dividend-data-download/blob/master/download-dividend-data.ipynb), that shows how the data has been gathered.    
# It looks like that the data set on dividend.com is way smaller than the one on robinhood.com. In fact, only 5% of stocks that I found on robinhood, I could also find on dividend.com. So pretty much all the columns that come from dividend.com aren't filled very well. But hey, it's better than nothing!   
# 
# As [the table in this repository](https://github.com/jonnylangefeld/dividend-data-download) shows, `dividend_yield`comes from robinhood.com and `div_yield` comes from dividend.com. Hence the first one is filled fully (because the data is selected for only dividend paying assets). So I recommend to use `dividend_yield`.
# 
# Anyway, let's have a look at all categorial columns:

# In[ ]:


category_cols =  list(instruments.select_dtypes(include='category').columns)
category_cols


# We will just loop over all categorial columns and print a countplot with the top 30 categories (some of the charts looked really weired when I included all available categories). This gives us a feeling of the distribution of the dataset. 

# In[ ]:


for a in category_cols:
    f, ax = plt.subplots(figsize=(20, 7))
    sns.countplot(x = a, data = instruments, orient='h', order = instruments[a].value_counts().iloc[:30].index)
    ax.set_title(a)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)


# So it turns out most of the assets are US assets, however what's interesting is that a lot of these stocks are listed in Zurich.
# Also all the stocks are tradable (because I selected only tradables in the [download script](https://github.com/jonnylangefeld/dividend-data-download/blob/master/download-dividend-data.ipynb)), however some where only tradable to close a position.
# 
# Next let's look at the numeric columns:

# In[ ]:


numeric_cols =  list(instruments.select_dtypes(include='float').columns)
numeric_cols


# In[ ]:


instruments.loc[:, numeric_cols].describe().T


# Right now it's pretty late at night and I will leave it at that for now. Feel free to fork this notebook or come up with your own data exploration. And where it get's really interesting is obviously a rating, of which stocks with the best risk/reward ratio you should buy for your dividend portfolio :)
# 
# Have fun!    
# [@jonnylangefeld](https://twitter.com/jonnylangefeld)

# In[ ]:




