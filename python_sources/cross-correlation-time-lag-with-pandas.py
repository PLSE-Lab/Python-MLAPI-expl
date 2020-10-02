#!/usr/bin/env python
# coding: utf-8

# <h1>Deriving features and cross-correlation (time-lag) </h1>

# In some machine learning projects, also referred to as experiments, often have to work with time series. Sometimes mean that it is quite helpful to have subject matter knowledge in the area under investigation to aid in selecting meaningful features to investigate paired with a thoughtful assumption of likely patterns in data.
#  
#  Sometimes to correlate - or rather, cross-correlate - with each other, to find out at which time lag the correlation factor is the greatest.
#  
# In next sections I will share an approach that I have been applying in some projects. I had answered a similar question in the stackoverflow some time ago, [here you can see the question]( https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas. 
#  
# 

# <h2>Using a weather dataset as a example</h2>

# In[2]:


import pandas as pd


# In[34]:


df = pd.read_csv("../input/sudeste.csv")


# I will choose a especif weather station as an example.  **RIO DE JANEIRO** (weather station id: 384) and some **more recents records**

# In[41]:


dfx = df[(df['wsid']==384) & (df['mdct'] > '2015-01-01 00:00:00')  ]
dfx.head(5)


# <h2>Cross-correlation (time-lag) using pandas</h2></h2>

# Let's get focus in some features: 
# * temp (temperature) 
# * hmdy (relative humidity)

# In[68]:


fields = ['mdct','temp','hmdy'] # mdct is datetime 
x = dfx[fields]
x.head(10)


# Imagine that you need to **correlate the temp** in t with t-1 (1 hour ago), t-2 (2 hours ago), ... t-n(n hours ago).  A good approach is **create a function** that shifted your dataframe first before calling the corr(). Let us break down what we hope to accomplish, and then translate that into code. For each hour (row) and for a given feature (column) I would like to find the value for that feature N hours prior. For each value of N (1-6 in our case) I want to make a new column for that feature representing the Nth prior hour's measurement.

# In[95]:


def df_derived_by_shift(df,lag=0,NON_DER=[]):
    df = df.copy()
    if not lag:
        return df
    cols ={}
    for i in range(1,lag+1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k,v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)    
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i+=1
        df = pd.concat([df, dfn], axis=1, join_axes=[df.index])
    return df


# Suposing that you need to correlate 6 hours. 

# In[98]:


NON_DER = ['mdct',]
df_new = df_derived_by_shift(x, 6, NON_DER)


# In[99]:


df_new.head(10)


# You will probably remember that I have** intentionally introduced missing values for the first six hours **of the data collected by deriving features representing the prior six hours of measurements. It is not until the sixth hour in that we can start deriving those features, so clearly I will want to exclude those first sixth hours from the data set.

# In[100]:


df_new = df_new.dropna()


# In[101]:


df_new.head(10)


# <h2>Cross-correlation: Finally </h2></h2>

# In[103]:


df_new.corr()


# In[108]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
colormap = plt.cm.RdBu
plt.figure(figsize=(15,10))
plt.title(u'6 hours', y=1.05, size=16)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# Now I can't say that I have significant knowledge of meteorology or weather prediction models, but I can say that the prior 3 hours of temperature have a good correlation for the following measurements.
