#!/usr/bin/env python
# coding: utf-8

# Just a start for now....

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import bokeh
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/price.csv')
df_sqft = pd.read_csv('../input/pricepersqft.csv')


# In[ ]:


print(df.shape)
print(df_sqft.shape)
print(df['City'].nunique())
print(df.columns)


# In[ ]:


##print(df.head())


# In[ ]:


##print(df_sqft.head())


# In[ ]:


df = df.set_index('Population Rank')
df_sqft = df_sqft.set_index('Population Rank')


# In[ ]:


df['Total Change'] = df.loc[:,'January 2017'] - df.loc[:,'November 2010']
df['Percent Change'] = (df.loc[:,'January 2017'] - df.loc[:,'November 2010']) / df.loc[:,'November 2010']
df['Year Change'] = df.loc[:,'January 2017'] - df.loc[:,'January 2016']
df['Year Percent Change'] = (df.loc[:,'January 2017'] - df.loc[:,'January 2016']) / df.loc[:,'January 2016']

##df.head()


# In[ ]:


print('Mean rent (Nov, 2010): ' + str(df['November 2010'].mean()))
print('Standard Deviation of rent (Nov, 2010): ' + str(df['November 2010'].std()))
print('Mean rent (Jan, 2017): ' + str(df['January 2017'].mean()))
print('Standard Deviation of rent (Jan, 2017): ' + str(df['January 2017'].std()))
print('------------------------------------------------------')
print('Mean total change: ' + str(df['Total Change'].mean()))
print('Standard Deviation total change: ' + str(df['Total Change'].std()))
print('Mean percent change: ' + str(df['Percent Change'].mean()))
print('Standard Deviation percent change: ' + str(df['Percent Change'].std()))
print('------------------------------------------------------')
print('Mean 1 year change: ' + str(df['Year Change'].mean()))
print('Standard Deviation 1 year change: ' + str(df['Year Change'].std()))
print('Mean 1 year percent change: ' + str(df['Year Percent Change'].mean()))
print('Standard Deviation 1 year percent change: ' + str(df['Year Percent Change'].std()))


# In[ ]:


df_sqft['Total Change'] = df_sqft.loc[:,'January 2017'] - df_sqft.loc[:,'November 2010']
df_sqft['Percent Change'] = (df_sqft.loc[:,'January 2017'] - df_sqft.loc[:,'November 2010']) / df_sqft.loc[:,'November 2010']
df_sqft['Year Change'] = df_sqft.loc[:,'January 2017'] - df_sqft.loc[:,'January 2016']
df['Year Percent Change'] = (df_sqft.loc[:,'January 2017'] - df_sqft.loc[:,'January 2016']) / df_sqft.loc[:,'January 2016']
##df_sqft.head()


# In[ ]:


kb = df.groupby('Metro')['Total Change'].mean()
plot = sns.kdeplot(kb)
plt.show()
plt.clf()


# In[ ]:


jan = df.groupby('State', as_index=True)['Total Change'].mean()
plot = sns.kdeplot(jan)
plt.show()
plt.clf()


# In[ ]:


## Locate the highest Total change value (investigate skew)

print('Minimum total change: ' + str(df['Total Change'].min()))
print(df.loc[df['Total Change'].idxmin(),:].head(5))
print('-------------------------------')
print('Maximum total change: ' + str(df['Total Change'].max()))
print(df.loc[df['Total Change'].idxmax(),:].head(5))


# In[ ]:




print(df.loc[df['January 2017'].idxmax(),:].head())
print(df['January 2017'].max())
print(df.loc[df['January 2017'].idxmax(),'November 2010':'January 2017'].mean())
print(df.loc[df['January 2017'].idxmax(), 'Total Change'])


# In[ ]:


print(df.groupby('State')['Total Change'].mean().sort_values(ascending=False).head(5))
print('---------------------------------')
print(df.groupby('State')['Total Change'].mean().sort_values(ascending=True).head(5))


# In[ ]:




