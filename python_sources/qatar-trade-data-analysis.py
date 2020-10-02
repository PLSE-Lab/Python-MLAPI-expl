#!/usr/bin/env python
# coding: utf-8

# The dataset contains about about 20M records in 12 columns (7 numerical 5 categorical)
# No null value recorded.
# 
# **OPEN ISSUES**
# 
# * calculate through linear reg in months above 30 
# * calculate diffence between expected and actual
# * see substitution
# 
# **CLOSED ISSUES**
# * Monhs 1-11 seem to have all the exact same count (i.e. 1787064). Why is that?
# -the series is balanced (thanks Duc)
# * The 3 ports all seem to have the exact same count (i.e. 6999334). Why is that?
# -the series is balanced (thanks Duc)
# * What are HS2, HS4, HS6 and HS8 and what is their relation (HS4 is a sub group of HS2, HS6 is sub groups of HS4 etc). This hierarchy do not seem to be linked to country, weight value or anything else)
# -codes describing the type of item imported
# * 12267 descriptions are null (no match between code and hs2)
# - issue with HS2 77 not existing
# * time values seem not to be correct (max 12, min -35)
# - fixed formula

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import zipfile

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## **Load data**
# The dataset has been converted in CSV (UTF-8) and compressed to zip to ease handling and loading.
# Additionally an xlsx file with the HS2 classification has been loaded.
# 
# The two files are linked. Additional columns with further details on classifications are dropped.

# In[ ]:


df = pd.read_csv("/kaggle/input/qatartrade1518/EVERPROD.PAN.csv")
dfcodes = pd.read_csv("/kaggle/input/qatartrade15-18/HS2code.csv",delimiter='\t')

dfcodes.set_index('code', inplace=True)

df = df.join(dfcodes, on='hs2')
#df1 = df
df = df.drop(['Unnamed: 0', 'hs8','hs6','hs4'], axis=1)

df = df.astype({"hs2": object, "year": int })


# A column with the month number starting from Jan 2015 is created, and the year and month columns are dropped.
# A column indicating the the month of the embargo is added. (1 = during embargo)

# In[ ]:


df['time'] = (((df['year']-2015) * 12) + df['month'])

df.loc[df['time'] < 30, 'emb'] = False
df.loc[df['time'] >= 30, 'emb'] = True

df = df.drop(['year','month'], axis = 1)


# ## Explore Data

# In[ ]:


df.sample(5)


# In[ ]:


df.shape


# In[ ]:


df.head(10)


# In[ ]:


df.info()


# In[ ]:


df.describe().round()


# In[ ]:


df.isnull().sum()


# In[ ]:


ccol = ['iso3c','port','continent','region','hs2', 'emb','description','category']
ncol = ['time','weight','import_value']


# In[ ]:


for c in ccol:
    print(df[c].value_counts())
    print("-")


# In[ ]:


country_list = df.iso3c.unique()
time_list = df.time.unique()


# In[ ]:


country_list_sh=['USA','CHN','DEU','JPN','ARE','GBR','IND','ITA','FRA','SAU','CHE','TUR','KOR','ESP','THA','NLD','OMN','AUS','MYS','VNM','EGY']


# Top 21 countries by import value are selected

# In[ ]:


df1 = pd.DataFrame(df[df.iso3c == 'TLS'].groupby(['time'])['import_value'].sum())
df1.rename(columns={'import_value': 'TLS'}, inplace = True)


# In[ ]:


df1.drop("TLS", axis='columns',inplace = True)


# In[ ]:


for  c in country_list_sh:
    df1 = df1.join(pd.DataFrame(df[df.iso3c == c].groupby(['time'])['import_value'].sum()))
    df1.rename(columns={'import_value': c}, inplace = True)


# In[ ]:



corr = df1.corr().round(2)
corr.style.background_gradient(cmap='PiYG')
#YlGn RdYlGn
# 'RdBu_r' & 'BrBG' are other good diverging colormaps


# It's clear that the embargo imposed by SAU, ARE, EGY has quite clear inverse relationship with OMN (-.73, -.69, -.39), expecially the first two. When import decreases in the first 2, we can see a rise in OMN import. 
# 
# Also top inverse relation for SAU are IND-.72, TUR -.79, OMN -.73. Indicating, possibly that whatever was not inported from SAU, probably was substituted by imports from these contruies (further research is needed)
# 
# Top inverse for ARE are the same countries (similar values).
# 
# Top inverse for EGY are is TUR (-.73) and to lesser extent IND and OMN (-.39)

# In[ ]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA


# In[ ]:


fig, axs = plt.subplots(21,figsize=(40,250), dpi = 50, sharex=True)
i=0
x = df1.index
plt.style.use('fivethirtyeight')

for c in country_list_sh:
    y = df1[c].values
    axs[i].plot(x, y, label = c)
    axs[i].tick_params(axis='both', which='major', labelsize=30)
    axs[i].title.set_text(c)
    #axs[i].title.set_size(40)
    axs[i].axvline(x=29)
    i=i+1


# As all the series are (quite stationary) for the periods 0 - 29 (i.e. pre-embargo), we can apply ARIMA model to predict the expected values for each country that could have been expected if the embargo would have not occured. 

# In[ ]:


plt.style.use('fivethirtyeight')

for c in country_list_sh:
    autocorrelation_plot(df1[c][:29].values)
    plt.title(c)
    plt.show()
    
#     axs[i].plot(x, y, label = c)
#     axs[i].tick_params(axis='both', which='major', labelsize=30)
#     axs[i].title.set_text(c)
    #axs[i].title.set_size(40)
#     axs[i].axvline(x=29)
#     i=i+1


# Autoregression does not seem to be significant the model will be be p = 0 hence MA

# In[ ]:


model = ARIMA(df1['DEU'][:29].values, order=(0,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[ ]:


model = ARIMA(df1['ITA'][:29].values, order=(0,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[ ]:


residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# In[ ]:


itaforcast = model_fit.forecast(steps=18)


# In[ ]:


forca = itaforcast[0]


# In[ ]:


forcastgraph = np.append( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],forca)


# In[ ]:


forcastgraph


# In[ ]:


(itaforcast[1])


# In[ ]:


plt.plot(df1[30:].index, df1['DEU'][30:].values)
plt.plot(df1[30:].index, forcastgraph[30:])

