#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import ExcelWriter
from pandas import ExcelFile
import io
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns



get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output.


# In[ ]:


plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize']=(10,10)


# In[ ]:


data=pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',encoding='latin1')
data.info()


# In[ ]:


data=pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',encoding='latin1')
data.info()


# In[ ]:


data.head()


# In[ ]:


#using 'sum' function will show us how many nulls are found in each column in dataset
data.isnull().sum()


# In[ ]:


#ok no null values


# In[ ]:


#examining the unique values of n_group as this column will appear very handy for later analysis
data.state.unique()


# In[ ]:


#coming back to our dataset we can confirm our fidnings with already existing column called 'calculated_host_listings_count'
top_fire_state=data.number.max()
top_fire_state


# In[ ]:


#Lets see wich state contains more fire occourrences

ax = data.groupby(['state']).sum()['number'].sort_values(ascending=False).plot(kind='bar')

# Change the y axis label to Arial
ax.set_ylabel('Total', fontname='Arial',fontsize=12)

# Set the title to comic Sans
ax.set_title('states with the most fire occourrences',fontname='Comic Sans MS', fontsize=18)


# Ok, Mato Grosso is the Braziian State with most fire occourrences.

# In[ ]:


#Lets see wich month contains more fire occourrences

plt.figure(figsize=(15,7))
sns.boxplot(x='month',y='number',data=data[['month','number']])


# In[ ]:


# Number of fires per Year
plt.figure(figsize=(20,7))
sns.boxplot(x='year',y='number',data=data[['year','number']])


# **The Result**
# 
# As we can see, Mato Grosso is where Brazil has more fire occourrences. July, Octuber and Noveber are the months with the most cases of fire in Brazil which shows that the drier the weather, more fire occourrences hapen.

# In[ ]:




