#!/usr/bin/env python
# coding: utf-8

# # Predicting Amazonic Forest Burning Outbreaks, is it Growing?
# 
# ### **Brun**
# #### 02/10/2019
# 
# * **1 Introduction**
#     * 1.1 Analyzing the data
# * **2 Plotting the burnings outbreaks**
#     * 2.1 Comparing other years with 2019
# * **3 Predicting future outbreaks**

# In[ ]:


#For our Image
from IPython.display import Image
from IPython.core.display import HTML 

Image(url= "https://media1.s-nbcnews.com/j/newscms/2019_35/2985041/190826-amazon-brazil-fire-cs-834a_eef05addd0d4dc77d766710fa90413d8.fit-760w.jpg", width=950, height=900)


# ## 1 **Introduction**
# 
# These days the discussion about the amazon forest fire turns out in to a big thing, since the last couple of months, many journals and websites published a lot of material and opinions about it, my mains goal here is not to do the same, I'm here to take a look at some data public published in the INPE (Instituto Nacional de Pesquisas Espaciais) [website](http://queimadas.dgi.inpe.br/queimadas/portal-static/estatisticas_paises/), and try to see if there's something unusual.
# 
# Lately the subject about the environment got a huge attention, the president of France Emannuel Macron comunicated to the international comunity to 

# ###  **1.1 Analyzing the data**

# In[ ]:


# Loading our packages
import pandas as pd
import matplotlib.pyplot as plt #For our beautiful plots!
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import warnings
import numpy as np


# In[ ]:


# Gathering our data
fires_mo = pd.read_excel('../input/inpe-queimadas/fires_mo.xlsx')


# Let's take a fast look at our data and see with what we are dealing

# In[ ]:


fires_mo


# Great! Pretty much data! We can do many things with it, maybe the first one is to take a look and see if we can find something in some visual data.

# ## **2 Plotting burning outbreaks**

# In[ ]:


#Treating the data

#
#Getting from 1999 to the front
fires_mo_c = fires_mo.iloc[1:,:13]

#Removing 3 last rows
fires_mo_c = fires_mo_c.iloc[:21]

time = fires_mo_c.Ano.loc[0:21]
months = []

for i in range(fires_mo_c.iloc[:,1:].shape[0]):
    months += [fires_mo_c.iloc[i,1:]]    

#Concatenating the lists    
long_ts = pd.DataFrame(pd.concat(months, axis=0))


#Generating the dates again
long_ts = long_ts.set_index(pd.date_range('1999-01','2019-12',freq='MS').strftime("%Y-%b"))
long_ts.columns = ['Fires']

#Dealing with non-existing values in series
long_ts[long_ts.Fires=='-']=0


#Turning into numerical    
long_ts = long_ts.astype(int)

#Eliminating two last zeros and october because it hasn't ended yet
long_ts = long_ts[:-3]

    
# Building the timeseries
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(long_ts.index,
        long_ts['Fires'],
        color='purple')
ax.xaxis.set_major_locator(ticker.MultipleLocator(24))



# The time series look pretty seasonalized, indicating that there are moments in which the number of forest burning grows. It's harder to see with all this data, but it seems that it's concentrated in the middle of each year.
# 
# Well, based on this graph, the only thing that it can be seen is that the burnings are higher in the years before 2019, like in the 2003 and 2008 period it was way higher. But, we can't be focusing too much on a seasonalized series, it's harder to find a growth pattern and can lead us to a dubious analysis, let's take a look at a not seasonalized series, maybe it can take us to another look.

# In[ ]:



#Ignore it, please
warnings.simplefilter('ignore')

# #Importing some R functions, I didn't find some cool TS treatment ones in python
# import rpy2.robjects as robjects

# # import rpy2's package module
# import rpy2.robjects.packages as rpackages

# # R vector of strings
# from rpy2.robjects.vectors import StrVector

# package_names = ('base', 'seasonal', 'stats')
# if all(rpackages.isinstalled(x) for x in package_names):
#     have_package = True
# else:
#     have_package = False    
# if not have_package:    
#     utils = rpackages.importr('utils')
#     utils.chooseCRANmirror(ind=1)    
#     packnames_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
#     if len(packnames_to_install) > 0:
#         utils.install_packages(StrVector(packnames_to_install))

# base = rpackages.importr('base')
# seasonal = rpackages.importr('seasonal')
# stats = rpackages.importr('stats')

# from rpy2.robjects import pandas2ri

# pandas2ri.activate()

# r_dataframe = pandas2ri.py2ri(long_ts.Fires)

# fires_ts_r = stats.ts(r_dataframe, start=2003, freq=12)

# fires_des_r = seasonal.final(seasonal.seas(fires_ts_r))

# import rpy2.robjects as ro

# fires_des_r_df = ro.DataFrame(base.list(fires_des_r))

# fires_des_pd_df = pandas2ri.ri2py(fires_des_r_df)

# #Plot it in python

# #Making the ts
# fires_des_ts = pd.concat([fires_des_pd_df,pd.DataFrame(long_ts.index)], axis=1)
# fires_des_ts.columns = ['des','tim']


fires_des_ts = pd.read_excel('../input/treated-inpe-series/fires_des_ts.xlsx')


# Building the timeseries
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(fires_des_ts['tim'],
        fires_des_ts['des'],
        color='purple')
ax.xaxis.set_major_locator(ticker.MultipleLocator(24))


# Now it looks better, the increase from 2002 to 2004 is huge! From that high spike on time, it seems that it's going down through time since than.
# 
# But one thing is looking clearer, the 2019 year is being higher than the previous years! Maybe if we take a look at the total values of burning outbreaks this tendency can be easily seen.

# In[ ]:


#Ignore it, please
warnings.simplefilter('ignore')


#Getting from 1999 to 2019 total fires outbreaks
fires_mo_tot = pd.DataFrame(fires_mo.Total[1:22])

#Setting the index
fires_mo_tot = fires_mo_tot.set_index(time)

# Plotting timeseries
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(fires_mo_tot.index.astype(int),
        fires_mo_tot['Total'],
        color='purple')
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))


# Ok, as it seems, from the past years it was way higher than now, but that not an excuse for any fire outbreak, even more because 2019 already is higher than 2018! We still have some months to end this year and my biggest fear is that it turns to be much higher than 2018. Which months does the burning outbreaks occur with more frequency?

# In[ ]:




#plotting the max, mean and minimum

fires_stats = np.transpose(fires_mo.iloc[-3:,1:13]).astype('int')
fires_stats.columns = ['Maximum', 'Mean', 'Minimum']

#Generating timestamps
fires_stats = fires_stats.set_index(pd.date_range('1999-01','1999-12',freq='MS').strftime("%b"))


fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(fires_stats.index,
        fires_stats['Maximum'],
        fires_stats.index,
        fires_stats['Mean'],
        fires_stats.index,
        fires_stats['Minimum'], label='sine')
ax.legend(('Maximum','Mean', 'Minimum'))


# It looks that the months between july and october the outbreaks tends to get higher, that's exactly when the news and world leaders start talking about the situation in the forest.

# ### **2.1 Comparing other years with 2019**

# If we compare the months from other years with 2019, does it demonstrate a increase in the fires outbreaks? Let's take a look at it!

# In[ ]:


#How does the 2019 outbreaks are doing compared to the other years?

fires_comps = np.transpose(fires_mo.iloc[2:23,1:11]).astype('int')
fires_comps.columns = fires_mo.iloc[2:23,0]


list_comps = []
#Looping to see the compared years with 2019
for i in range(fires_comps.shape[1]):
    list_comps += [round((fires_comps.iloc[:,20]/fires_comps.iloc[:,i])*100,2)]
    
df_comps = pd.concat(list_comps, axis=1)
df_comps.columns = fires_mo.iloc[2:23,0]

#Generating timestamps
df_comps = df_comps.set_index(pd.date_range('1999-01','1999-10',freq='MS').strftime("%b")).iloc[:,:-1]

df_comps


# ## 3 **Predicting future outbreaks**

# What if we try to predict this years output based on the past observations? For that we can analyse different types of time series estimators and see which one can get the closest in our 2019 series, after that predict the next two months output and in some months we can compare the results!

# In[ ]:




