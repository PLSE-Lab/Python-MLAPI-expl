#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import seaborn as sns
from fbprophet import Prophet
import plotly.express as px


# In[ ]:


df_orgl = pd.read_csv("../input/world-economic-indiactors-by-oecd/Stats oecd Nov 2019.csv", index_col = 'Time', parse_dates = True)
df_orgl.head()


# In[ ]:


df_orgl.describe()
df_orgl.info()
print(df_orgl.columns)


# **Indicators that are included in this data**

# In[ ]:


df = df_orgl[['Country', 'Variable', 'Value']]
df['Variable'].unique()


# In[ ]:


#looking at the Current account balance in USD
df_cc = df.loc[df['Variable']==('Current account balance in USD')]
df_cc['Country'].unique()


# **Boxplot**

# In[ ]:


# rot = rotates labels at the bottom
# fontsize = labels size
# grid False or True

df_cc.boxplot('Value', 'Country', rot = 80, figsize = (16,10), fontsize = '12',grid = True);


# NEED HELP!!! 
# I want to plot 2019 data and 2020 data on the same graph back to back. Could you please advise me here how to do that?

# In[ ]:


df_cc_2019 = df_cc['2019-01-01':'2019-12-31']
df_cc_2019.head()
df_cc_2020 = df_cc['2020-01-01':'2020-12-31']
df_cc_2019.head()


# In[ ]:


sns.barplot(x = 'Country', y = 'Value', data = df_cc_2019)

sns.barplot(x = 'Country', y = 'Value', data = df_cc_2020)


# In[ ]:


df_2019 = df_cc.loc['2019-01':'2019-02'].plot.bar(x='Country', figsize=(18,12), title='Current Account in USD in 2019', color = 'r', label='2019', grid=True, rot = 80)
df_2020 = df_cc.loc['2019-02':'2020-01'].plot.bar(x='Country', figsize=(18,12), title='Current Account in USD in 2020', color = 'g', label='2020', grid=True, rot = 80)


# In the graph below, I got it stuckable 2019 and 2020. How do I make them next to each other side by side? 

# In[ ]:


year2019 = df_cc.loc[df_cc.index == ('2019-01-01')]
year2020 = df_cc.loc[df_cc.index == ('2020-01-01')]
fig, ax = plt.subplots(figsize=(16,12))
ax.bar(year2019['Country'], year2019['Value'], color = 'g')
ax.bar(year2020['Country'], year2020['Value'], color = 'r')
ax.set_title('Current Account Balance in 2019 and in 2020', fontsize = 20, color = 'grey')
ax.grid(color = "grey", linestyle = '-', linewidth = 0.15)
plt.xticks(rotation=90)
ax.legend('best', labels = ('2019', '2020'));

