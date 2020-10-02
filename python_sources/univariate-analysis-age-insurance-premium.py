#!/usr/bin/env python
# coding: utf-8

# # **UNIVARIATE ANALYSIS | AGE VARIABLE | INSURANCE PREMIUM PREDICTION**
# 
# ## Problem
# Describe the age variable.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[ ]:


plt.rc('figure', figsize=(15, 5))


# In[ ]:


insurance = pd.read_csv('/kaggle/input/insurance-premium-prediction/insurance.csv')


# # Solution
# The age variable is discrete numerical variable. 

# In[ ]:


age = insurance.age


# ## Counts

# In[ ]:


pd.Series({'Count Age': age.count(), 
           'Count Missing Age': age.isna().sum(), 
           'Minimum Age': age.min(), 
           'Maximum Age': age.max()})


# ## Measures of Frequency

# ### Absolute frequency

# In[ ]:


age.hist();


# The rectangular shape of the histogram represents [uniform distribution.](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))

# ### Relative Frequency

# In[ ]:


sns.distplot(age);


# A histogram with a single peak in center represents a normal distribution. However, the distribution of this variable represents discrete uniform distribution.

# ## Measure of Central Tendency (Location)
# With this we try to estimate a point that explain the whole population. Generally, two major measures are mean and median:
# * Mean is the typical measure that describes whole population.
# * Median is the value of the point which has half the data smaller than that point and half the data larger than that point (i.e. 50th percentile of the population.)

# In[ ]:


pd.Series({'Mean Age': round(age.mean(), 2), 
           'Median Age': age.median(), 
           'Mode Age': list(age.mode())})


# For the uniform distribution mean is calculated as follows:

# In[ ]:


(age.max() + age.min()) / 2


# ### Cumulative Frequency
# 
# Reference: [Documentation for cumulative density function at Matplotlib](https://matplotlib.org/3.2.1/gallery/statistics/histogram_cumulative.html)

# In[ ]:


age.hist(bins=10, density=True, cumulative=True, histtype='step');


# ## Measure of Dispersion (Scale, Variability & Spread)
# Measures of scale are simply attempts to estimate this variability:
# 1. How spread out are the data values near the center?
# 2. How spread out are the tails?

# In[ ]:


pd.Series({'Range': age.max() - age.min(), 
           'Variance': age.var(), 
           'Standard Deviation': age.std(), 
           'Mean Absolute Definition': age.mad(), })


# In[ ]:


age.plot.box(vert=False);


# ## Measure of Shape
# **Skewness** is a measure of symmetry, or more precisely, the lack of symmetry. A distribution, or data set, is symmetric if it looks the same to the left and right of the center point.
# 
# **Kurtosis** is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. That is, data sets with high kurtosis tend to have heavy tails, or outliers. Data sets with low kurtosis tend to have light tails, or lack of outliers. A uniform distribution would be the extreme case.
# 
# Reference: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm

# In[ ]:


pd.Series({'Skewness': round(age.skew(), 2), 
           'Kurtosis': round(age.kurtosis(), 2)})


# Uniform distribution's 
# * skewness is close to zero.
# * kurtosis is close to -6/5 (-1.2).
# 
# Hence, both of these are being met.

# # Conclusion
# 
# Based on above analysis, we can conclude that the data is discrete uniform distribution.

# In[ ]:




