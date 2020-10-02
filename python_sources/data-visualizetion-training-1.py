#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print('Setup Complete')


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input/uncover/UNCOVER/New_York_Times'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


NYT_c_data = pd.read_csv('/kaggle/input/uncover/UNCOVER/New_York_Times/covid-19-county-level-data.csv', index_col='date')
NYT_c_data


# In[ ]:


NYT_s_data = pd.read_csv('/kaggle/input/uncover/UNCOVER/New_York_Times/covid-19-state-level-data.csv', index_col='date')
NYT_s_data


# In[ ]:


NYT_s_data.info()


# In[ ]:


NYT_s_data.index.value_counts()


# In[ ]:


deaths_each_state = NYT_s_data.groupby(['date', 'state'])['deaths'].max().unstack()
deaths_each_state


# In[ ]:


deaths_each_state.index


# In[ ]:


deaths_each_state.columns


# In[ ]:


plt.figure(figsize=(20, 8))
plt.title('amount of death in each states in america')
g = sns.lineplot(data=deaths_each_state['Alabama'], label='Alabama')
g = sns.lineplot(data=deaths_each_state['Texas'], label='Texas')
g = sns.lineplot(data=deaths_each_state['Vermont'], label='Vermont')
g = sns.lineplot(data=deaths_each_state['Nevada'], label='Nebada')
g.set_xticklabels(labels=deaths_each_state.index, rotation=90)
plt.legend()
plt.ylabel('peaple who dead in covid-19')
plt.show()


# In[ ]:


deaths_each_state.Texas


# In[ ]:




