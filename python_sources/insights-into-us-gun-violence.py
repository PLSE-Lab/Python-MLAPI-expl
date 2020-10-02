#!/usr/bin/env python
# coding: utf-8

# Hi! I'm a beginner and this is only my second kernel in Kaggle.
# This data set discusses the growing gun violence which has been occuring in US for the past some year. I'll trying to get a few meaningful insights from the data provided here.
# So let us start....
# ![Gun](http://www.powherny.org/wp-content/uploads/2017/10/courtesy-harvard.edu_.gif)

# **Imports**
# 
# Let's start with importing the common libraries that we will need in this project

# In[31]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[32]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# **Loading the data**
# 
# Now I'll load the data set. I have further divided the date column into days, month and year. It will help us to get a better understanting of the data.

# In[33]:


path = "../input/gun-violence-data_01-2013_03-2018.csv"
df1 = pd.read_csv(path)


# In[34]:


df1.head()


# In[35]:


df1["date"]=pd.to_datetime(df1["date"],format="%Y-%m-%d")
df1["Year"]=df1["date"].apply(lambda time:time.year)
df1["Month"]=df1["date"].apply(lambda time:time.month)
df1["Day"]=df1["date"].apply(lambda time:time.day)
df1["Day_of_Week"]=df1["date"].apply(lambda time:time.dayofweek)
dmap1 = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}#We're Using this Dictionary to Map our column
df1["Day_of_Week"]=df1["Day_of_Week"].map(dmap1)
df1.head(2)


# Let's see what we are working with

# In[36]:


df1.info()


# In[37]:


df1.describe()


# Let us see the most gruesome incident of the last few years

# In[9]:


print ('The incident that killed most people')

df2=df1.loc[df1['n_killed'].idxmax()]
df2[['date','n_killed','state','city_or_county']]


# **Number of Incidents in Each State**
# 
# 
# Let's see how many incidents each state faced and the national average

# In[10]:


plot = df1.state.value_counts().plot(kind='bar', title="Number of Incidents in Each State",                              figsize=(18,10), colormap='rainbow')
plot.set_xlabel('State')
plot.set_ylabel('Number of incidents')
mean_line = plot.axhline(df1.state.value_counts().mean(), color='r',                         label='Average Number of incidents')
plt.legend()


# We see Illinois, California, Florida and Texas are the worst affected states. We also see the national mean comes around 5000 (which is a pretty high number!).

# **Loss of lives**
# 
# Let's see which states have lost most lives due to gun violence. 

# In[11]:


df3=df1.groupby('state').agg({'n_killed':['sum']}).plot.bar(figsize=(18,10), colormap='summer')


# California, Texas, Florida has suffered the most followed closely by Illinois. Not surprising since these states had the most number of incidents as we found out from the previous visualization.

# **Worst affected cities**
# 
# Now let us find the top 10 worst affected citites with gun violence in US.

# In[12]:


x=df1['city_or_county'].value_counts()
x=x.sort_values(ascending=False)
x=x.iloc[0:10]
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.ylabel('# Incidents', fontsize=15)
plt.xlabel('City', fontsize=12)



#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# Looks like Chicago leads the way with a fair margin. 

# **Year wise damage**
# 
# Now we will see the yearwise damage that gun violence has done

# In[16]:


df1.groupby('Year').agg({'n_killed':['sum']}).plot.line(figsize=(8,8), colormap='summer')


# In[14]:


df1.groupby('Year').agg({'incident_id':['count']}).plot.line(figsize=(8,8), colormap='rainbow')


# We see gun violence has been increasing steadily from 2014 to 2017, which is not a good news!

# **Most dangerous days**

# In[20]:


plt.style.use('ggplot')
list8=df1.groupby(["Month","Day_of_Week"]).count()["incident_id"].unstack()
plt.figure(figsize=(16,10))
sns.heatmap(list8,cmap='viridis')


# As we clearly see from the heatmap, most incidents seem to occur on Sunday. Also January seems to be the most vicious month followd by March also we see a huge rise of incidents in Sundays of July and August 

# In[23]:


x=df1['Day_of_Week'].value_counts()
x=x.sort_values(ascending=False)
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Worst affected days",fontsize=25)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('# incidents', fontsize=12)
plt.xlabel('Days', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# In[29]:


x=df1['Month'].value_counts()
x=x.sort_values(ascending=False)
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Worst affected months",fontsize=25)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('# incidents', fontsize=12)
plt.xlabel('Mon', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# The above two graphs clearly proves the claims we inferred from the heatmap.

# **Incidents all round the year of states**
# 
# Now I'll find the worst affected month of different states

# In[25]:


plt.style.use('ggplot')
df2=df1.groupby(["Month","state"]).aggregate({'state':['count']}).unstack()
plt.figure(figsize=(20,10))
sns.heatmap(df2,cmap='coolwarm')
plt.title('Incidents all round the year of states')


# We get some valuable insights from the heatmap. This data can be used by security.

# **Map Visualization**
# 
# 

# In[27]:


import imageio
import folium
import folium.plugins as plugins
from mpl_toolkits.basemap import Basemap

# Sample it down to only the North America region 
lon_min, lon_max = -132.714844, -59.589844
lat_min, lat_max = 13.976715,56.395664

#create the selector
idx_NA = (df1["longitude"]>lon_min) &            (df1["longitude"]<lon_max) &            (df1["latitude"]>lat_min) &            (df1["latitude"]<lat_max)
#apply the selector to subset
shootings=df1[idx_NA]

#initiate the figure
plt.figure(figsize=(20,10))
m2 = Basemap(projection='merc',
             llcrnrlat=lat_min,
             urcrnrlat=lat_max,
             llcrnrlon=lon_min,
             urcrnrlon=lon_max,
             lat_ts=35,
             resolution='c')

m2.fillcontinents(color='#191919',lake_color='#000000') 
m2.drawmapboundary(fill_color='#000000')    
m2.drawcountries(linewidth=0.1, color="w")              

# Plot the data
mxy = m2(shootings["longitude"].tolist(), shootings["latitude"].tolist())
m2.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.05, zorder=5)

plt.title("Shootings in USA")


# Let us chekout the worst affected cities 
# 
# *P.S. Increase your brightness to get a clearer picture.*

# In[45]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))

#a random point inside chicago city
lat = 41.8781
lon = -87.6298
#some adjustments to get the right pic
lon_min, lon_max = lon-0.3,lon+0.5
lat_min, lat_max = lat-0.4,lat+0.4
#subset for vegas
data_vegas=df1[(df1["longitude"]>lon_min) &                    (df1["longitude"]<lon_max) &                    (df1["latitude"]>lat_min) &                    (df1["latitude"]<lat_max)]

#Facet scatter plot
data_vegas.plot(kind='scatter', x='longitude', y='latitude',
                color='yellow', 
                s=.02, alpha=.6, subplots=True, ax=ax1)
ax1.set_title("Chicago")
ax1.set_facecolor('black')

#a random point inside baltimore
lat = 39.2904
lon = -76.6122
#some adjustments to get the right pic
lon_min, lon_max = lon-0.3,lon+0.5
lat_min, lat_max = lat-0.4,lat+0.3
#subset for pheonix
data_baltimore=df1[(df1["longitude"]>lon_min) &                    (df1["longitude"]<lon_max) &                    (df1["latitude"]>lat_min) &                    (df1["latitude"]<lat_max)]
#plot pheonix
data_baltimore.plot(kind='scatter', x='longitude', y='latitude',
                color='yellow', 
                s=.02, alpha=.6, subplots=True, ax=ax2)
ax2.set_title("Baltimore")
ax2.set_facecolor('black')
f.show()


# **Wordcloud**
# 
# 

# In[28]:


from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
import matplotlib as mpl

mpl.rcParams['figure.figsize']=(20,15)    
mpl.rcParams['font.size']=12                
mpl.rcParams['savefig.dpi']=100             
mpl.rcParams['figure.subplot.bottom']=.1 


stopwords = set(STOPWORDS)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=250,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df1['incident_characteristics']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=1100)


# **Final insights...**
# 
# * Illinois, California, Florida and Texas are the worst affected states.
# * Most incidents occured in Sunday.
# * January and March are the most deadly months.
# * We see a sharp rise of inicednts in Sundays of July and August.
# 

# In[ ]:




