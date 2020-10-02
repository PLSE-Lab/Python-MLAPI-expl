#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import stats

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))


# In[ ]:


pd.set_option('display.max_columns',None)


# ## Loading the Data

# In[ ]:


games = pd.read_csv('../input/videogamesales/vgsales.csv')


# In[ ]:


games.head()


# In[ ]:


games.shape


# In[ ]:


games.info()


# # Checking The Null Values

# In[ ]:


games.isnull().sum()


# ## Checking The Unique values

# In[ ]:


games.nunique()


# # Data Cleaning

# In[ ]:


games[(games.Year.isnull()) & (games.Publisher.isnull())].head()


# In[ ]:


games[games.Publisher.isnull()]['Genre'].value_counts()


# In[ ]:


games[(games.Genre == 'Action') & (games.Publisher.isnull())]


# In[ ]:


games[games.Publisher == 'Unknown'].shape[0]


# In[ ]:


(games[(games.Publisher == 'Unknown') | (games.Publisher.isnull())]['Publisher'].count()/games.shape[0])*100


# In[ ]:


games.Publisher.fillna('Unknown',inplace=True)


# In[ ]:


games.Publisher.isnull().sum()


# In[ ]:


games.head()


# In[ ]:


games[games.Platform == 'Wii']['Year'].mode()[0]


# In[ ]:


games.Genre.nunique()


# In[ ]:


med=pd.DataFrame(games.groupby(by=["Platform","Genre"])["Year"].median().reset_index())
med


# In[ ]:


bnm=pd.DataFrame(games[games[["Platform","Genre","Year"]]["Year"].isna()][["Platform","Genre","Year"]]).reset_index()
bnm


# In[ ]:


for i in range(0,len(med)):
    for j in range(0,len(bnm)):
        if (med["Platform"][i]== bnm["Platform"][j]) & (bnm["Genre"][j]  ==  med["Genre"][i]) :
            bnm["Year"][j]=med["Year"][i]


# In[ ]:


for i in range(0,len(games)):
    for j in range(0,len(bnm)):
        if (games["Platform"][i]== bnm["Platform"][j]) & (bnm["Genre"][j]  ==  games["Genre"][i]) :
            games["Year"][i]=bnm["Year"][j]


# In[ ]:


games.isnull().sum()


# In[ ]:


games.head()


# ## Maximum Number of Games Released Per Year

# In[ ]:


plt.figure(figsize=(15,10))
plt.xticks(rotation = 70, color='white', size=10)
sns.countplot(x='Year',data=games)
plt.show()


# ## Maximum Number of Games of specific Genre

# In[ ]:


plt.figure(figsize=(15,10))
plt.xticks(rotation = 70, color='white', size=10)
sns.countplot(x='Genre',data=games)
plt.show()


# ## Total Number of Global Sales Per Year

# In[ ]:


games.groupby(['Year'])['Global_Sales'].sum()


# ## Which platform has the highest sales overall ?

# In[ ]:


games.groupby('Platform')['Global_Sales'].sum().reset_index().sort_values('Global_Sales',ascending=False).head(1)


# ## Which year has the highest sales globally for each platform ?

# In[ ]:


games.groupby(['Platform','Year'])['Global_Sales'].sum().reset_index().groupby(['Platform'])['Global_Sales','Year'].max()


# In[ ]:




