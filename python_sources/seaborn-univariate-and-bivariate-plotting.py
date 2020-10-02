#!/usr/bin/env python
# coding: utf-8

# Datasets possess numberical and categorical features. To analyse the distribution of these variables over a range of variables, various types of plots are used .

# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
sns.set(color_codes=True)


# In[ ]:


titanic=pd.read_csv('../input/train.csv')
titanic.head()


# ***One Dimensional Plotting-Histograms***
# 
# Histograms bin the numerical values into various ranges. Then a continous function is passed on top of these unique bins by using Kernel Density Estimate, to get a continous probability distribution of the random variable.

# In[ ]:


sns.distplot(titanic['Fare'])


# We can even change the number of bins in the plot by setting the bins functions. As you can see that the fares of 500 correspond to outliers in the titanic dataset.

# In[ ]:


sns.distplot(titanic['Fare'],bins=100)


# The histogram bins can be switched off

# In[ ]:


sns.distplot(titanic['Fare'],hist=False, rug=True)


# The vertical 'rugged' lines correspond to the actual data distribution , and the blue continuos curve is the kernel estimate for the data distribution.

# Now lets analyze how the dependence of two variables can be analzed with respect to each other .We use joint plots for this. Two two features should be numberic in the case of joint plots.

# In[ ]:


titanic.head()


# In[ ]:


sns.jointplot(x="Fare", y="Age", data=titanic);


# The count gets cluttered here. We can use the hexagonal approximation to better understand the plots by colouring the different hexagonal regions with different colours .

# In[ ]:


sns.jointplot(x="Fare", y="Age",kind='hex', data=titanic);


# In[ ]:


sns.jointplot(x="Fare", y="Age",kind='kde', data=titanic);


# A more powerful technique is to use the pairplots which are an awesome way to get the double relationships betweeen multiple features. Once  you get that, you can choose interesting ones and plot them frther with faceting.

# In[ ]:


sns.pairplot(titanic)


# In[ ]:


g = sns.PairGrid(titanic)
g.map_diag(sns.barplot)
g.map_offdiag(sns.jointplot,kind='kde')


# You have understood plotting for numerical,categorical features, and crossplotting them using faceting. Use these tools in your toolbox for awesome diagrams and obtainingg nice insights from the dataset.
