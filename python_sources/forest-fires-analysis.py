#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')


# In[ ]:


data.head()


# In[ ]:


data.drop(['date'] , axis=1 , inplace=True)


# In[ ]:


sns.heatmap(data.isnull() , yticklabels = False , cbar = False , cmap='viridis')


# There is no NULL values in the dataset

# In[ ]:


data.info()


# In[ ]:


df = data.groupby('state')['number'].sum().sort_values(ascending=False)

plt.figure(figsize = (15,7))
df.plot.bar()
plt.title("Number of the Forest Fires Related To States")
plt.ylabel('Avg of Fires')
plt.xlabel('States')


# *Mato Grosso is the city with the most Forest Fires. The reason may be that it has more forests or has more population than the other cities.*

# In[ ]:


df = data.groupby('month')['number'].sum().sort_values(ascending = False)

plt.figure(figsize = (15,7))
df.plot.bar(color='purple')
plt.title("Number of the Forest Fires Related To Months")
plt.ylabel('Avg of Fires')
plt.xlabel('Months')


# *Julho (July) is the month with the most Forest Fires. The reason may be that weather is the hot.*

# In[ ]:


df = data.groupby('year')['number'].sum().reset_index()

plt.figure(figsize=(18,6))
gr = sns.lineplot( x = 'year', y = 'number',data = df, color = 'blue', lw = 3)
gr.xaxis.set_major_locator(plt.MaxNLocator(19)) # 19 values between 1998 - 2017
gr.set_xlim(1998, 2017) # set to x
sns.set()

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Number of Fires per Year in Brazil',fontsize=18)
plt.xlabel('Year', fontsize = 18)
plt.ylabel('Number of Fires', fontsize = 18)


# As we see , year of 2003 has the most forest fires between 1998-2017. It increases from 1998 at average.

# In[ ]:


df = pd.DataFrame(data[data['state'] == 'Amazonas'])
new_df = df.groupby('year')['number'].sum().reset_index()

plt.figure(figsize=(17,8))
gr = sns.lineplot( x = 'year', y = 'number',data = new_df, color = 'red', lw = 2.5)
gr.xaxis.set_major_locator(plt.MaxNLocator(19)) # 19 values between 1998 - 2017
gr.set_xlim(1998, 2017) # set to x
sns.set()

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Number of Fires per Year In Amazon ',fontsize=18)
plt.xlabel('Year', fontsize = 18)
plt.ylabel('Number of Fires', fontsize = 18)

