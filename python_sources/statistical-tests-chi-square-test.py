#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Aim of this kernel is to perform a chi-square test to check whether two variables are independent or not. 
# 
# ## Data
# 
# The data comes in from the aircraft wildlife strikes data set and it contains records of wildlife strikes against aircrafts from 1990 to 2015. 
# 
# We would be looking at indepedence testing for these two scenarios:
# 
# - visibility and flight impact
# - If there is a different impact caused by Seagulls and Mourning doves. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import chisquare, chi2_contingency, chi2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


aircraft = pd.read_csv("../input/database.csv")
aircraft.head()


# In[ ]:


# One-way chisquare test to see if states are evenly distributed
chisquare(aircraft['State'].value_counts())


# In[ ]:


# One-way chisquare test to see if states are evenly distributed
chisquare(aircraft['Species Name'].value_counts())


# In[ ]:


# Combine the engine shutdown and engine shut down entries in flight impact column
aircraft.loc[aircraft["Flight Impact"] == "ENGINE SHUT DOWN", "Flight Impact"] = "ENGINE SHUTDOWN"


# In[ ]:


# Two-way Chisquare test for the relationship between visibility and flight impact
contingencyTable = pd.crosstab(aircraft['Visibility'],aircraft['Flight Impact'])
print(contingencyTable)
stat, p, dof, expected = chi2_contingency(contingencyTable)
stat, p, dof, expected


# Let's setup the null hypothesis vs alternate hypothesis testing via below code to understand if the hypothesis holds or not. 

# In[ ]:


# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# As we have used the chi2_contigency method, let's understand the output:
# - 601.172 = The test statistic.
# - 1.29019 = p-value
# - 16      = degrees of freedom
# - array   = expected frequencies
# 
# > We can find more info here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

# In[ ]:


# Two-way chisquare test to see whether mourning doves and gulls cause different impact
subset = aircraft.loc[aircraft['Species Name'].isin(['MOURNING DOVE','GULL'])]

bird_impact = pd.crosstab(subset["Species Name"], subset["Flight Impact"])
stat, p, dof, expected = chi2_contingency(bird_impact)
stat, p, dof, expected


# In[ ]:


# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# - 81.843  = The test statistic.
# - 7.08391 = p-value
# - 4       = degrees of freedom
# - array   = expected frequencies

# As we can see from the above results, the variables are dependent. 

# In[ ]:


# Two-way Chisquare test for the relationship between Aircraft Make and flight impact
contingencyTable = pd.crosstab(aircraft['Aircraft Make'],aircraft['Flight Impact'])
print(contingencyTable)
stat, p, dof, expected = chi2_contingency(contingencyTable)
stat, p, dof, expected


# In[ ]:


# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# In[ ]:


# Two-way Chisquare test for the relationship between Operator and flight impact
contingencyTable = pd.crosstab(aircraft['Operator'],aircraft['Flight Impact'])
print(contingencyTable)
stat, p, dof, expected = chi2_contingency(contingencyTable)
stat, p, dof, expected


# In[ ]:


# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# In[ ]:




