#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('ls ../input/')


# ### Explore the data

# In[ ]:


#Explore sample_submission_1.csv
sample = pd.read_csv('../input/web-traffic-time-series-forecasting/sample_submission_1.csv')
print('sample shape',sample.shape)
sample.head()


# In[ ]:


#Explore key_1.csv
key = pd.read_csv('../input/web-traffic-time-series-forecasting/key_1.csv',index_col='Page')
print('key shape',key.shape)
key.head()


# ### Data Exploration
# read data, set page as index then use unstack flatten the data.

# In[ ]:


data = pd.read_csv('../input/web-traffic-time-series-forecasting/train_1.csv',
                   index_col='Page').T.unstack().reset_index().rename(
    columns={0:'Visits','level_1':'Date'}).dropna(subset=['Visits'])
data.head()


# ## Make sense of the data
# Read pages for exploratory purposes

# In[ ]:


pages = pd.read_csv('../input/web-traffic-time-series-forecasting/train_1.csv',usecols=['Page'],squeeze=True)
print(len(pages))
pages.head()


# In[ ]:


# current_palette = sns.color_palette()
pal = sns.cubehelix_palette(1, start=2, rot=-10, dark=.7, light=.95,as_cmap=True)

#plot dist
labels = ['wikipedia.org','Not wikipedia.org']
fig,ax = plt.subplots(1,figsize=(6,6))
(pages.str.lower().str.contains('wikipedia.org').value_counts().rename(
    index=lambda x: ['Not wikipedia.org','wikipedia.org'] )/len(pages)).plot(
    kind='pie',labels=labels,autopct='%1.0f%%', pctdistance=.8, labeldistance=.3,ax=ax,cmap=pal);
ax.set_title('Pages');
ax.set_ylabel('');


# In[ ]:


from matplotlib import cm
dates = get_ipython().getoutput('head -1 ../input/web-traffic-time-series-forecasting/train_1.csv')
dates = dates[0][8:-1].split('","')

dates_av = pd.to_datetime(pd.Series(dates))
#no repeated date
print('No repeated date',dates_av.shape==dates_av.unique().shape)
#repeated month-day?
md = dates_av.apply(lambda d: d.replace(year=2016))
daysinmonth =dict(zip(md.dt.strftime('%b'),md.dt.days_in_month))
ser = md.map(lambda x: x.strftime('%b')).value_counts()
colors = ser/ser.index.map(lambda x: daysinmonth[x])
print('repeated days',md.unique().shape[0]/md.shape[0])
bars = plt.bar(range(len(ser.index)),colors,width=.9);
for bar in bars:
    bar.set_facecolor([cm.coolwarm_r(.6),pal(.6)][bar.get_height()<2])
plt.gca().set_xticks(range(len(ser.index)));
plt.gca().set_yticks([0,1,2])
plt.gca().set_xticklabels(ser.index);
plt.gca().set_ylabel('Days/days in month');
plt.gca().set_xlabel('Months');
plt.gca().set_title('Data distribution regarding month-day');


# ## Feature Extraction
# using extract and assigning the results to columns in train
# #### Page related features

# In[ ]:


fnames = ['Name','Language','Access','Agent']
fnamedict = dict(zip(range(len(fnames)),fnames))
pFeatures= pages.str.extract(
    '(.+)_(\w{2})\.wiki.+_(.+)_(.+)',expand=True).rename(columns=fnamedict)
pFeatures['Page'] = pages
pFeatures.set_index('Page',inplace=True)
pFeatures.head()


# #### Explore page features values

# #### Find language correspondance
# Using read html and [List of languages](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) 
# ###### I had to create a dataset to access this

# In[ ]:


# url= 'https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes'
# df_l = pd.read_html(url,header=0)[1].dropna(
#     how='all',axis=1).dropna(
#     how='all').loc[:,['639-1',
#                       'ISO language name']].set_index('639-1')
df_l = pd.read_csv('../input/wikipedia-language-iso639/lang.csv',index_col=0)
df_l.head()


# ### Group and plot language data

# In[ ]:


fig,ax = plt.subplots(1,figsize=(8,8))
res = pFeatures.groupby('Language')['Language'].count().rename(
    index=df_l.iloc[:,0].to_dict())                            
patch,labels,prec = ax.pie(res,labels= res.index,
                                        autopct='%1.0f%%', 
                                        pctdistance=.8, 
                                        labeldistance=.6);
#position labels
for t in labels:
    t.set_horizontalalignment('center')
    t.set_rotation(-9)
#change pie patches colors
for p in patch:
    p.set_color(pal(np.random.rand(1)[0]))


#     ### group and plot access data

# In[ ]:


fig,ax = plt.subplots(1,figsize=(6,6))
pFeatures.groupby('Access')['Access'].count().plot(kind='pie',
                                                       cmap=pal,
                                                       autopct='%1.0f%%', 
                                                       pctdistance=.8, 
                                                       labeldistance=.4,
                                                       ax=ax);


# ### group and plot Agent data

# In[ ]:


fig,ax = plt.subplots(1,figsize=(6,6))
pFeatures.groupby('Agent')['Agent'].count().plot(kind='pie',
                                                       cmap=pal,
                                                       autopct='%1.0f%%', 
                                                       pctdistance=.8, 
                                                       labeldistance=.4,
                                                       ax=ax);


# #### Overall stats

# In[ ]:


data.describe()


# ###### information 

# In[ ]:


data.info()


# #### Convert the `Date` column to TimeDate

# In[ ]:


data.Date = pd.to_datetime(data.Date)


# #### Check if converting `Visits` to integers makes a difference to memory usage

# In[ ]:


data.Visits=data.Visits.astype(int)
data.info()


# #### Sort the dataframe `data` on the `Date` column

# In[ ]:


data.sort_values('Date',inplace=True)


# ##### Rough plot of the overall picture

# In[ ]:


import matplotlib.dates as mdates
months = mdates.MonthLocator()  # every month
monthsFmt = mdates.DateFormatter('%b')
fig,ax = plt.subplots(1,figsize=(12,6))

ax.plot(data['Date'].values,data['Visits'].values,color=pal(.6))

# format the ticks
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)

ax.set_xlim(dates_av.min(), dates_av.max())
# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()
plt.show()


# 
# ## Analysis of Time Series Data

# In[ ]:


import pandas as pd


# In[ ]:


data_s = pd.read_csv('../input/web-traffic-time-series-forecasting/train_1.csv',
                     index_col='Page').rename(columns=pd.to_datetime)
data_s.info()


# In[ ]:


# from sklearn.preprocessing import LabelEncoder


# In[ ]:


##### Time idependant  learing model


# In[ ]:




