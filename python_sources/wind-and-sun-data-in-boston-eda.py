#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input/"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)


# In[2]:


df = pd.read_csv('../input/tmy3.csv')
df.head()


# In[5]:


df.shape


# In[3]:


dg = pd.read_csv('../input/TMY3_StationsMeta.csv')
dg.head()


# In[10]:


sns.lmplot(y='Latitude', x='Longitude', hue='Pool', 
           data=dg, 
           fit_reg=False, scatter_kws={'alpha':0.1})


# In[9]:


dg.dtypes


# In[6]:


dg.shape


# In[8]:


dg.Pool.value_counts().plot.bar()


# In[11]:


dg.State.value_counts().plot.bar()


# In[12]:


dg[dg.State == 'MA']
# Let me choose MARTHAS VINEYARD and PROVINCETOWN (AWOS), BOSTON LOGAN INT'L ARPT


# In[16]:


db = df[df['station_number'].isin([725066, 725073, 725090])]


# In[20]:


db.columns


# In[18]:


db.groupby('station_number')['Pressure (mbar)'].mean()


# In[19]:


db.groupby('station_number')['Wspd (m/s)'].mean()


# In[24]:


db['date_parsed'] = pd.to_datetime(db['Date (MM/DD/YYYY)'], infer_datetime_format = True)


# In[ ]:





# In[26]:


dgroup = db.groupby(['date_parsed' , 'station_number'])['DNI (W/m^2)'].mean()


# In[27]:


dgroup.plot()


# In[30]:


dgroup = dgroup.reset_index().head()


# In[31]:


dgroup


# In[32]:


dgroup = dgroup.set_index('date_parsed')


# In[33]:


dgroup['DNI (W/m^2)'].plot.line()


# In[ ]:




