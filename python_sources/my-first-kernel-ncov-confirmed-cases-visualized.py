#!/usr/bin/env python
# coding: utf-8

# ### Simple analysis and visualization of 2019-nCoV cases
# 
# #### This is my first kaggle kernel!
# Hope that i could improve not only this kernel countinuously and can give some insights on 2019-nCoV outbreak, but also produce meaningful kernel that gives insight or solutions to various kaggle dataset and competitions.
# 
# I also hope that 2019-nCoV outbreak could be successfully handled with least casualty as possible.

# Import Packages

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Loading dataset and checking dataset size

# In[ ]:


nCov_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")    #nCov dataframe


# In[ ]:


print('DataFrame Size : ', nCov_df.shape)    #checking dataset size
nCov_df.info()  #checking dataset infos. Province/State column has NA values


# In[ ]:


nCov_df['Province/State'] = nCov_df['Province/State'].fillna('Province Unknown')    #Filling missing datas


# Getting latest datas(latest observations), and splitting those into china/non-china cases

# In[ ]:


nCov_df_sorted = nCov_df.sort_values(by=['Date'], ascending=False)
cond_latest = nCov_df_sorted['Date'] == nCov_df_sorted.iloc[0, 1]
cond_china = nCov_df_sorted['Country'] == 'Mainland China'
cond_nonchina = nCov_df_sorted['Country'] != 'Mainland China'
nCov_df_latest_china = nCov_df_sorted[cond_latest & cond_china]
nCov_df_latest_nonChina = nCov_df_sorted[cond_latest & cond_nonchina]


# In[ ]:


plot = sns.catplot(y = 'Province/State', x = 'Confirmed', kind = 'bar', data = nCov_df_latest_china, height = 6, aspect = 2, orient = "h")
plot.set_xticklabels(rotation=45)
plt.subplots_adjust(top=0.9)
plot.fig.suptitle('Confirmed case in China')


# In[ ]:


plot = sns.catplot(x = 'Country', y = 'Confirmed', kind = 'bar', data = nCov_df_latest_nonChina, height = 5, aspect = 3)
plot.set_xticklabels(rotation=45)
plt.subplots_adjust(top=0.9)
plot.fig.suptitle('Confirmed case in other countries')

