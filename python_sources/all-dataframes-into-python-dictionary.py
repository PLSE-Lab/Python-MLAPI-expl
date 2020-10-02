#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


filenames = filenames.split('\n')


# In[3]:


dfs = dict()
for f in  filenames:
    dfs[f[:-4]] = pd.read_csv('../input/'+ f)
    


# In[5]:


dfs['Donors'].sample(10)


# In[11]:


dfs['Donors']['Donor Is Teacher'].value_counts().plot.bar()


# In[6]:


dfs['Donations'].sample(10)


# In[9]:


dfs['Donations']['Donation Amount'].hist(bins = 40)


# In[13]:


dfs['Projects'].sample(5)


# In[15]:


dfs['Projects']['Project Type'].value_counts().plot.bar()


# In[16]:


dfs['Resources'].sample(5)


# In[17]:


dfs['Resources']['Resource Unit Price'].hist()


# In[ ]:





# In[18]:


dfs['Schools'].sample(5)


# In[20]:


dfs['Schools'].groupby('School State')['School ID'].count().plot.bar()


# In[ ]:





# In[21]:


dfs['Teachers'].sample(5)


# In[22]:


dfs['Teachers']['Teacher Prefix'].value_counts().plot.bar()


# In[ ]:




