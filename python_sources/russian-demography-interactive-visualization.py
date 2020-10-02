#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


# import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()


# ### Data cleaning ###

# In[ ]:


data = pd.read_csv('../input/russian-demography/russian_demography.csv')


# In[ ]:


data.isnull().sum()


# In[ ]:


data = data.dropna()
data.isnull().any()


# ### Data analysis ###

# In[ ]:


sns.heatmap(data.corr(), square=True, annot=True, cbar=False);


# In[ ]:


data.iloc[:, 1:].describe()


# From this table we can see that in Russia a permanent decline in the population (-2.34 by 1000 people) an average of 27 years.

# In[ ]:


df_gb = data.groupby('year')[['birth_rate', 'death_rate']].mean()


# In[ ]:


df_gb


# In[ ]:


df_gb.iplot(mode='lines+markers', xTitle='Year', yTitle='Average',
    title='Yearly Average birth_rate and death_rate')


# In[ ]:


df_ = data.groupby('region')[['birth_rate', 'death_rate']].mean()
df_.iplot(kind='bar', xTitle='Region', yTitle='Average',
    title='Average birth_rate and death_rate in regions')


# In[ ]:


df_npg = data.groupby('region')[['npg']].mean()
df_npg.iplot(kind='bar', xTitle='Natural population growth by 1000 people', yTitle='Average',
    title='Average natural population growth by 1000 people in regions')


# Natural population growth is observed in the republics of the Caucasus with traditional cultural and in the some regions of the North.
# The most population decline in the central Russia.

# In[ ]:


plt.scatter(data['year'],  data['gdw'], label=None,
            c=data['urbanization'], cmap='viridis',
            linewidth=0, alpha=0.5)
plt.xlabel('year')
plt.ylabel('general demographic weight')
plt.colorbar(label='% of urban population');


# In regions with hight percent of urban population less general demographic weight than in the regions with a small indicator. In big cities families have less children than in towns and villages.
# Also we can see in first 10 years of the new century the general demographic weight was significantly low than other time. This can be associated with a low birth rate and high mortality at this time.

# In[ ]:


data_capital = pd.DataFrame()
data_capital = data[(data['region']=='Moscow') | (data['region']=='Saint Petersburg') | (data['region']=='Leningrad Oblast') | (data['region']=='Moscow Oblast')]
data_capital = data_capital.drop(columns=['npg', 'birth_rate', 'death_rate', 'gdw'])


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
ax.plot(data_capital['year'][(data['region']=='Moscow')], 
        data_capital['urbanization'][(data['region']=='Moscow')], ':b', label='Moscow')
ax.plot(data_capital['year'][(data['region']=='Moscow Oblast')], 
        data_capital['urbanization'][(data['region']=='Moscow Oblast')], '-g', label='Moscow Oblast');
ax.plot(data_capital['year'][(data['region']=='Saint Petersburg')], 
        data_capital['urbanization'][(data['region']=='Saint Petersburg')], 'o', label='Saint Petersburg')
ax.plot(data_capital['year'][(data['region']=='Leningrad Oblast')], 
        data_capital['urbanization'][(data['region']=='Leningrad Oblast')], label='Leningrad Oblast')
plt.legend();


# From this table we see that urbanization of Moscow become less and urbanization of Moscow oblast become more.
# This was due to official expansion of the administrative borders of Moscow at the territory of the Moscow region in 2012.
# The rural population has increased in Leningrad Oblast and the number of urban population has decreased.

# ### Happy coding ###
