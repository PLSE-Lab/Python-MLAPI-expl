#!/usr/bin/env python
# coding: utf-8

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


datam=pd.read_csv(os.path.join(dirname, filename))
datam


# # General informations

# ### Columns meaning
# ### sl_no = Serial Number 
# ### gender = Male or Female
# ### ssc_p = senior sceondary percentage (10)
# ### ssc_b = senior secondary board
# ### hsc_p = higher secondary percetage (12)
# ### hsc_b = higher secondary board (12)
# ### hsc_s = higher secondary stream (12)
# ### degree_p = degree percentage (UG)
# ### degree_t = degree type (UG)
# ### workex = work experience
# ### etest_p = its a test percentage
# ### specialisation = the specialisation you are doing in MBA
# ### mba_p = MBA Percentage
# ### status = are you placed or not
# ### salary = package 

# # Let's try making One way ANOVA. ( I tried doing this to the best of my knowledge)

# In[ ]:


#Since degree_t is of categorical type, we have to make it numeral type
#So as to perform the computation
#for this we are making an entire different datatype
datam1=datam.copy(deep=True)


# In[ ]:


dummy=pd.get_dummies(datam1["degree_t"])
dummy


# In[ ]:


datam1=pd.concat([datam1,dummy], axis=1)
datam1


# In[ ]:


datat=pd.concat([datam1['mba_p'],dummy], axis=1)
datat


# In[ ]:


import scipy.stats as stats
# stats f_oneway functions takes the groups as input and returns F and P-value
fvalue, pvalue = stats.f_oneway(datat['Comm&Mgmt'], datat['Others'], datat['Sci&Tech'], datat['mba_p'])
print(fvalue, pvalue)


# In[ ]:


data_new=pd.melt(datat.reset_index(), id_vars=['index'], value_vars=['mba_p',"Sci&Tech",'Others','Comm&Mgmt'])
data_new.columns=['index','treatments','value']
data_new


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[ ]:


model=ols('value ~ C(treatments)', data=data_new).fit()


# In[ ]:


model.summary()


# In[ ]:


anov_table=sm.stats.anova_lm(model, typ=1)
anov_table


# In[ ]:


#to know the pairs of signufucant different treaments
#lets perform mmultiple pairwise comparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd

m_comp=pairwise_tukeyhsd(endog=data_new['value'], groups=data_new['treatments'], alpha=0.05)
print(m_comp)


# #### Tukey HSD shows that 1st, 2nd & 4th accepts the null hypothesis.

# ### lets do Shapiro-Wilk test to check whether the data is drawn from normal distribution or not
# ### null hypothesis = data is normally distributed

# In[ ]:


w, pvalue = stats.shapiro(model.resid)
print(w, pvalue)


# #### since pvalue isn't significant, i.e., p<0.05, we reject our null hypothesis

# ### Bartlett test checks the homogeneity of variances.
# ### Null hypothesis is that samples from populatons have equal variance
# 

# In[ ]:


w,pval=stats.bartlett(datat['mba_p'],datat['Comm&Mgmt'],datat['Others'],datat['Sci&Tech'])
print(w,pval)


# #### since pvalue is significant i.e., pvalue<0.05, therefore we reject our null hypothesis

# In[ ]:




