#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print(__version__) # requires version >= 1.9.0
# For Notebooks
init_notebook_mode(connected=True)
# For offline use


# In[71]:


# Handy code which I prefer to enlarge the plots
plt.rcParams["figure.figsize"] = (18,9)


# In[72]:


train = pd.read_csv('../input/train_2016_v2.csv')
properties = pd.read_csv('../input/properties_2016.csv', low_memory=False)
dictionary = pd.read_excel('../input/zillow_data_dictionary.xlsx')


# ## **The Overview**
# Lets check the heads of datasets along with the datatypes of various columns respectively

# In[73]:


train.head()


# In[74]:


properties.head()


# Looks like we have lot of NaN values in the properties. Let's explore there count...

# ###  Percentage of Values Missing from the Properties DataFrame

# 
# \begin{equation*}
# \text{Percentage of null} = 
#  \frac{ \text{MissingValue Counts} }{\text{Total Values}}  * 100
# \end{equation*}
# 

# In[75]:


null = pd.DataFrame(data= properties.isnull().sum()/len(properties)*100, 
                    columns=['Percentage of Values Missing'],
                    index=properties.columns
                   ).reset_index()


# In[76]:


null['Percentage of Values Missing'].mean()


# In[77]:


plt.rcParams["figure.figsize"] = (13,10)
sns.barplot(x= 'Percentage of Values Missing', 
            y='index', 
            data= null.sort_values(by='Percentage of Values Missing', ascending=False),
            color = '#ff004f') 


# Let's take this mean as the reference, to list out top features where there are 50%  values are NaN and see if they can be safely dropped or **not**

# In[78]:


## Caution - Only 50% percentile missing values are taken. There are 29 MORE!!!
Notorious_null = null[null['Percentage of Values Missing'] > null['Percentage of Values Missing'].mean()]


# In[79]:


Notorious_null.sort_values(by='Percentage of Values Missing', ascending=False).head(10)


# In[80]:


plt.rcParams["figure.figsize"] = (13,10)
sns.barplot(x= 'Percentage of Values Missing', 
            y='index', 
            data= Notorious_null,
            color = '#ff004f') 


# In[81]:


len(null) - len(Notorious_null)


# ## Correlation between various features

# Correlation is usually seen with the target variable, in this case **logerror**. But, the given dataset are seperated by two different csv of properties and train respectively. Lets merge them

# In[82]:


alldata = pd.merge(train, properties, how='inner', on='parcelid')


# In[83]:


alldata.head()


# In[84]:


# sns.heatmap(alldata.corr(), cmap='viridis', vmax=0.8, vmin=0)


# In[85]:


alldata.head(10)


# ![](http://)**Dropping all the NaN values above 90%**

# In[86]:


null_drop = null[null['Percentage of Values Missing'] > 85]


# In[87]:


col_to_drop = []
for cols in list(null_drop['index'].values):
    col_to_drop.append(cols)


# In[88]:


alldata.drop(col_to_drop, axis=1, inplace=True)


# In[89]:


alldata.head()


# In[90]:


nullv2 = pd.DataFrame(data= alldata.isnull().sum()/len(alldata)*100, 
                    columns=['Percentage of Values Missing'],
                    index=alldata.columns
                   ).reset_index()


# In[91]:


nullv2.sort_values(by='Percentage of Values Missing', ascending=False)


# ## Cleaning the Data for ML Crunching!

# In[92]:


alldata.fillna(value=0, inplace=True)


# In[93]:


alldata.head(8)


# In[94]:


sns.heatmap(alldata.corr().head(500), cmap='viridis', vmax=0.8, vmin=0)


# In[95]:


alldata.describe()


# In[96]:


from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=20)


# In[97]:


X = alldata.drop(['parcelid','logerror', axis=1)


# In[ ]:




