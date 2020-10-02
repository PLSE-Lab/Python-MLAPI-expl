#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.


# In[ ]:


file = '../input/master.csv'
data = pd.read_csv(file)


# In[ ]:


data.head()


# In[ ]:


data.isnull().head()


# In[ ]:


plt.figure(figsize = (7,7))
sns.heatmap(data.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')


# In[ ]:


data.info()


# In[ ]:


data.drop(['HDI for year'],axis = 1,inplace = True)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


sns.heatmap(data.corr(),annot = True,cmap = 'coolwarm',lw = 0.4,linecolor = 'white',alpha = 0.8,square = True)


# In[ ]:


data.columns


# In[ ]:


data=data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides/100kPop','country-year':'CountryYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


sns.distplot(data['Suicides/100kPop'],kde = False,bins = 30,color = 'red') 


# In[ ]:


sns.pairplot(data,hue = 'Generation',height = 4)


# In[ ]:




