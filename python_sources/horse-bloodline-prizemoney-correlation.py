#!/usr/bin/env python
# coding: utf-8

# ## Attempt to see if a horses quality is determined by the quality of their parents and grandparents ##
# 
# I receive a very poor correlation between horse quality and lineage.
# This may be because of several factors, such as;
# 
# - The best stallions and mares are taken out of racing before they accumulate big prize money
#   - If this is the case, a better "quality" metric will be needed instead of prize money.
# - The calculations that I have applied are not optimal.
#   - I use mean to calculate "quality", "parents_quality" and "gparents_quality"
#   - There may be a better method of calculating these metrics.
# 
# Any comments or suggestions would be most welcome.

# In[1]:


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
#! head ../input/forms.csv


# ## Does prize money increase with age.##
# We can see that it does...... Pearsons correlation is close to 1, so we would need to do some weighting on age to see how valuable a horse is in terms of prize money.

# In[2]:


horses = pd.read_csv('../input/horses.csv')
horseage = horses.groupby('age')['age', 'prize_money'].apply(lambda x: np.mean(x)).astype(int)
horseage['age'].corr(horseage['prize_money'])


# Also, a graph to demonstrate this age/prize_money increase

# In[3]:


horseage.plot()


# I will create a lookup dictionary for each age, and use this to calculate the difference for each horse based on their age/prize_money

# In[4]:


horse_lookup = horseage.to_dict()['prize_money']
horses['weighted_prize_money'] = horses['prize_money'] - horses['age'].map(horse_lookup)


# ## Ranking horses into quality bins, from 1 to 10 ##
# 
# We want to get the mean of the rankings.
# Although, because some horses have only 1 foal, it can be skewed

# In[5]:


horses['quality'] = pd.qcut(horses['weighted_prize_money'], 10, labels=[1,2,3,4,5,6,7,8,9,10])

horses1 = pd.merge(horses, horses[['id', 'quality', 'sire_id', 'dam_id']], left_on=['sire_id'], right_on=['id'], suffixes=('', '_sire'))
horses1 = pd.merge(horses1, horses[['id', 'quality', 'sire_id', 'dam_id']], left_on=['dam_id'], right_on=['id'], suffixes=('', '_dam'))


# In[6]:


# Show a subset of the dataframe
horses.plot(x='quality', y='weighted_prize_money')
#horses.head()


# ## Correlation between horse quality, and the mean quality of their parents ##
# 
# From below, we can see this is a very poor correlation, close to random.

# In[ ]:


horses1['parents_quality'] = horses1[['quality_sire', 'quality_dam']].mean(axis=1)
horses1['quality'].corr(horses1['parents_quality'])
horses1.plot.scatter(x='quality', y='parents_quality')


# In[ ]:


plt.hexbin(horses1['quality'], horses1['parents_quality'], gridsize=10)


# ## The same correlation for parents and grandparents ##
# 
# This value is closer, but still very low

# In[ ]:


horses2 = pd.merge(horses1, horses1[['id', 'quality']], left_on=['sire_id_sire'], right_on=['id'], suffixes=('', '_sire_sire'))
horses2 = pd.merge(horses2, horses1[['id', 'quality']], left_on=['sire_id_dam'], right_on=['id'], suffixes=('', '_sire_dam'))

horses2 = pd.merge(horses2, horses1[['id', 'quality']], left_on=['dam_id_sire'], right_on=['id'], suffixes=('', '_dam_sire'))
horses2 = pd.merge(horses2, horses1[['id', 'quality']], left_on=['dam_id_dam'], right_on=['id'], suffixes=('', '_dam_dam'))


# In[ ]:


horses2['gparents_quality'] = horses2[['quality_sire',
                                      'quality_dam',
                                      'quality_sire_sire',
                                     'quality_sire_dam',
                                     'quality_dam_sire',
                                     'quality_dam_dam']].mean(axis=1)
horses2['quality'].corr(horses2['gparents_quality'])


# ## Same Correlation, with only grandparents ##

# In[ ]:


horses2['gparents_quality'] = horses2[['quality_sire_sire',
                                     'quality_sire_dam',
                                     'quality_dam_sire',
                                     'quality_dam_dam']].mean(axis=1)
horses2['quality'].corr(horses2['gparents_quality'])


# ## Remove horses with high variance in quality through their bloodline ##
# 
# Another small improvement is seen here, but the correlation is still very low.

# In[ ]:


horses_low_var = horses2[horses2[['quality_sire',
         'quality_dam',
         'quality_sire_sire',
         'quality_sire_dam',
         'quality_dam_sire',
         'quality_dam_dam']].var(axis=1) < 10] #horses2

horses_low_var['gparents_quality'] = horses_low_var[['quality_sire',
                                                     'quality_dam',
                                                     'quality_sire_sire',
                                                     'quality_sire_dam',
                                                     'quality_dam_sire',
                                                     'quality_dam_dam']].mean(axis=1)
horses_low_var['quality'].corr(horses_low_var['gparents_quality'])


# In[ ]:





# In[ ]:




