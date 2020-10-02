#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[ ]:


nobel = pd.read_csv('../input/nobel-laureates/archive.csv')
nobel.head(n=6)


# ## First Nobel Prize Winner

# In[ ]:


display(len(nobel))
display(nobel['Sex'].value_counts())
nobel['Birth Country'].value_counts().head(10)


# ## Dominance of USA Nobel Winner

# In[ ]:


nobel['usa_born_winner'] = nobel['Birth Country'] == 'United States of America'

nobel['decade'] = (np.floor(nobel['Year'] / 10) * 10).astype(int)

prop_usa_winners = nobel.groupby('decade', as_index=False)['usa_born_winner'].mean()
prop_usa_winners


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sns.set()
plt.rcParams['figure.figsize'] = [11, 7]
ax = sns.lineplot(x='decade', y='usa_born_winner', data=prop_usa_winners)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))


# ## Gender Proportion of Nobel Prize Winner

# In[ ]:


# Calculating the proportion of female laureates per decade
nobel['female_winner'] = nobel['Sex'] == 'Female'
prop_female_winners = nobel.groupby(['decade', 'Category'], as_index=False)['female_winner'].mean()

# Plotting USA born winners with % winners on the y-axis
ax = sns.lineplot(x='decade', y='female_winner', hue='Category', data=prop_female_winners)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))


# ## First woman to win the Nobel Prize

# In[ ]:


nobel[nobel.Sex == 'Female'].nsmallest(1, 'Year')


# ## Laurettes with multiple nobel awards

# In[ ]:


nobel.groupby('Full Name').filter(lambda group: len(group) >= 2)


# ## Age of the winner when they were awarded

# In[ ]:


nobel['Birth Date'] = pd.to_datetime(nobel['Birth Date'],errors='coerce')

nobel['Age'] = nobel['Year'] - nobel['Birth Date'].dt.year
sns.lmplot(x='Year', y='Age', data=nobel, lowess=True, 
           aspect=2, line_kws={'Color' : 'black'})


# ## Age Differences between prize categories

# In[ ]:


sns.lmplot(x='Year', y='Age', row='Category', data=nobel, lowess=True, 
           aspect=2, line_kws={'color' : 'black'})


# ## Oldest and youngest winners

# In[ ]:


display(nobel.nlargest(1, 'Age'))

nobel.nsmallest(1, 'Age')


# In[ ]:





# In[ ]:





# In[ ]:




