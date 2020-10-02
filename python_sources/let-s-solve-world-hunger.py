#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


a = pd.read_csv("../input/Indicators.csv")


# In[ ]:


a.tail()


# In[ ]:


countries = pd.unique(a.CountryCode)
countries


# In[ ]:


years = pd.unique(a.Year)
years


# In[ ]:


cols = pd.unique(a.IndicatorCode)
cols


# In[ ]:


descs = pd.unique(a.IndicatorName)
descs


# In[ ]:


a = a.set_index(['IndicatorCode','CountryCode','Year'],drop=True)
del a['CountryName']
del a['IndicatorName']


# In[ ]:


a.tail()


# In[ ]:


def get_indicators(indicator):
    result = []
    for country in countries:
        for year in years:
            try:
                get_value = a.get_value((indicator,country,year),'Value')
                result.append(get_value)
            except:
                result.append(np.nan)
    return np.array(result)


# In[ ]:


base = get_indicators("NY.GDP.PCAP.CD")


# In[ ]:


from scipy import stats
def get_r_squared(indicator):
    data = get_indicators(indicator)
    mask = (~np.isnan(base)) & (~np.isnan(data))
    if np.any(mask):
        slope, intercept, r_value, p_value, std_err = stats.linregress(base[mask],data[mask])
        return r_value**2 * (r_value >= 0 and 1 or -1)
    else:
        return np.nan


# In[ ]:


result = []
i = 0
for c in cols:
    i += 1
    result.append(get_r_squared(c))
    if i % 10 == 0:
        print(i)


# In[ ]:


df = pd.DataFrame()
df['cols'] = cols
df['descs'] = descs
df['vals'] = result
df['mag'] = np.abs(result)
df = df.sort_values("mag",ascending=False)
df

