#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


pisa = pd.read_csv("../input/pisa-scores-males-students-math-data-2015/pisa math male.csv")
gdp  = pd.read_csv("../input/world-happiness/2015.csv")


# Let's explore the data

# In[ ]:


pisa.head()


# In[ ]:


gdp.head()


# Looks like the country name is our link between the two datasets.

# In[ ]:


pisagdp = pisa.join(gdp.set_index('Country'), on='Country Name')
pisagdp


# In[ ]:


drop_cols = ['Region',
             'Happiness Rank',
			 'Happiness Score' ,
			 'Standard Error' ,
			 'Family',
             'Health (Life Expectancy)' ,
			 'Freedom' ,
			 'Trust (Government Corruption)' ,
			 'Generosity' ,
			 'Dystopia Residual']
pisagdp.drop(drop_cols, axis = 1, inplace = True)
pisagdp

   


# Looks like the data set includes some missing data and outliers. Time to clean that up.

# In[ ]:


pisagdp.dropna(axis=0, how='any')


# Let's visualise the data for initial correlation analysis

# In[ ]:


pisagdp.plot(kind='scatter', x='Pisa', y='Economy (GDP per Capita)');


# Looks like there is a correlation. 

# In[ ]:


import matplotlib.pyplot as plt
import scipy.stats                  # for pearson correlation

pisagdp['Pisa'].corr(pisagdp['Economy (GDP per Capita)'])


# As expected, there seems to be a strong correlation between Pisa scores and GDP per capita
