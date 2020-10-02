#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Histograms and KDEs in Python
# 
# In this tutorial, we will walk through plotting histograms and kde (kernel density estimate) charts using Matplotlib and Seaborn in Python. We use the dataset of **Kaggle Datasets Collection** in this example. This notebook can also be used as a quick reference when you need to make histograms and kdes of various types.
# 
# ## Loading Data and Cleaning

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/kaggle_datasets.csv')
data.head()


# In[2]:


data.shape


# The dataset contains 8036 rows, but there are many datasets uploaded on Kaggle have very few audience or activity. In this case we want to exclude datasets that have neither kernels nor upvotes.

# In[3]:


data_ex0 = data.loc[(data.kernels > 0 ) | (data.upvotes > 0), :]
data_ex0.shape


# Only 3443 rows remain. And we will show the histograms based on this subset of data. Let's run some descriptive statistics as well:

# In[4]:


data_ex0.describe()


# ## Basic Histograms

# As a starting point, we plot a histogram of the distribution of the number of discussion threads using **plt.hist()** function in Matplotlib:

# In[5]:


plt.hist(data_ex0.discussions); # The ';' is to avoid showing a message before the chart


# We can also plot by .plot() method in pandas, which also uses Matplotlib:

# In[6]:


data_ex0.discussions.plot(kind='hist'); 


# In both charts, it seems there is nothing beyond 50 or so. Let us look at the maximum values:

# In[7]:


data_ex0.nlargest(10, 'discussions').loc[:,['title','discussions']]


# Two datasets have more than 100 discussions but most of them have zero or a few. When the data is very skewed and has a long tail, a default histogram will not give much information. We will show how to display the histograms in different ways for a closer look of data.

# ## Customization of Histograms
# 
# In this part we will show various arguments we can use to show the histograms differently to get a better sense of how the data is distributed. We will use the distribution of the number of kernels in this example:

# In[8]:


# Default plot
plt.figure(figsize=(8,5)) # Specify the figure size
plt.hist(data_ex0.kernels)
plt.show()


# In[9]:


# Zoom in to distribution of 0-20 kernels
plt.figure(figsize=(8,5))
plt.hist(data_ex0.kernels, range = (0, 21))
plt.show()


# In[10]:


# Look at the tail ends
plt.figure(figsize=(8,5))
plt.hist(data_ex0.kernels, range = (100, data_ex0.kernels.max())) # 100 up to highest number of kernels
plt.show()


# In[11]:


# Zero to 100 kernels in 20 bins
plt.figure(figsize=(8,5))
plt.hist(data_ex0.kernels, range = (0, 100), bins = 20)
plt.show()


# In[12]:


# Use a numpy array to specify how the bins are separated
plt.figure(figsize=(8,5))
plt.hist(data_ex0.kernels, bins = np.arange(5, 51, 5)) # 5-10, 10-15... up to 45-50
plt.show()


# In[13]:


# Taking logarithm on the x-axis
plt.figure(figsize=(8,5))
plt.hist(np.log1p(data_ex0.kernels)) # Use np.log1p instead of np.log to avoid error taking log of 0
plt.show()


# In[14]:


# Taking logarithm on the y-axis
plt.figure(figsize=(8,5))
plt.hist(data_ex0.kernels, bins=30, log=True)
plt.show()


# ## Completing the Charts
# 
# It's time to add title, axes labels, and change color of the histograms.

# In[15]:


plt.figure(figsize=(8,5))
plt.hist(data_ex0.kernels, bins=30, log=True, color = 'fuchsia')
plt.title('Distribution of Kernels Created', fontsize=16)
plt.xlabel('No. of kernels')
plt.ylabel('Frequency')
plt.show()


# Reference:
# - [matplotlib.pyplot.hist() documentation](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html)
# - [List of color names](https://matplotlib.org/gallery/color/named_colors.html)

# ## KDE Charts - The Basics
# 
# Instead of histograms, we can also plot kdes so that the distribution is shown as lines instead of rectangular bars. We will plot the distribution of upvotes:

# In[16]:


data_ex0.upvotes.plot.kde();


# The above chart looks strange as the number of upvotes cannot be positive. We can set the range to display with ind argument:

# In[17]:


data_ex0.upvotes.plot.kde(ind = np.arange(0, data_ex0.upvotes.max()));


# In[18]:


sns.kdeplot(data_ex0.upvotes);


# We can use the clip arguments to limit the datapoints to which the kde fits:

# In[19]:


sns.kdeplot(data_ex0.upvotes, clip = (0,200));


# ## KDE Plots of More Than One Factor
# 
# Then we make the last sets of charts by plotting kdes grouped by factor so that we can visualize the difference in distribution among factor values. Here we separate the data by whether they are 'featured' datasets and plot the distribution of upvotes:

# In[20]:


# Do it with Seaborn
plt.figure(figsize=(8,5))
sns.kdeplot(data_ex0.loc[data_ex0.featured == 0, 'upvotes'], color='green', label='non-featured')
sns.kdeplot(data_ex0.loc[data_ex0.featured == 1, 'upvotes'], color='red', label='featured')
plt.xlim(0, 100) # Limit the view from 0 to 100
plt.show()


# In[21]:


# Do it with Matplotlib
plt.figure(figsize=(8,5))
data_ex0.loc[data_ex0.featured == 1, 'upvotes'].plot.kde(color='red')
data_ex0.loc[data_ex0.featured == 0, 'upvotes'].plot.kde(color='green')
plt.legend(('Yes', 'No'), title='Featured?')
plt.xlim(0,100)
plt.show()


# It is seen that featured datasets tend to have more upvotes than the non-featured ones.
# 
# The looks of kde plots using Seaborn and Matplotlib are different because they use a different kernel estimate, which is beyond the scope of this notebook.
# 
# Reference documentation:
# - [pandas.DataFrame.plot.kde()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.kde.html)
# - [Seaborn.kdeplot()](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)
# 
# That's it for now. Happy plotting!
