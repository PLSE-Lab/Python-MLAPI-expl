#!/usr/bin/env python
# coding: utf-8

# # Analyzing Terrorist Activities Around The World
# 
# Thanks for stopping by my kernel.
# 
# ### This work is inspired by the work of many kagglers who have worked on this dataset. If you like my findings on the dataset, please, leave an upvote. Feedback on my findings is highly appreciated! 
# 
# I will be using Basemap for plotting geographical maps. There is a issue with basemap and python 3.6, I have resolved the issue. Feel free to ask me.
# 
# #### Note:
# This kernel is a work in progress. Be patience to see interactive maps, animations and word clouds.
# 
# #### Definition of terrorism: 
# 
# Terrorism is, in the broadest sense, the use of intentionally indiscriminate violence as a means to create terror among masses of people; or fear to achieve a financial, political, religious or ideological aim.
# 
# #### About the dataset:
# 
# The dataset contains information of more than 180,000 terrorist attacks.
# 
# The Global Terrorism Database (GTD) is an open source database including information on terrorist attacks around the world from 1970 through 2017, except 1993. The GTD includes systematic data on deomestic as well as international terrorist incidents that have occurred during this time. It includes more than 180,000 attacks. [More Information on dataset](http://start.umd.edu/gtd/)
# 
# 
# ## Content
# 
# There are 135 different columns in this dataset. But, I have selected some of them.
# 
# - year  -  in which year the attack happend
# - month	- in which month of the year the attack happened
# - day   - on which day of the month the attack happened 
# - country - in which country the attacked happened
# - region  - which region the country lies in	
# - city	  - the city in which the attack happened
# - latitude - of the location of the attack
# - longitude	- of the location of attack
# - attackType - 	type of attack
# - target - who was the target
# - targetType - type of target
# - groupName	- name of the group who executed the attack
# - weaponType - which type of weapon is used in the attack
# - success - whether the attack was successful or not
# - killed - how many people were killed in the attack
# - wounded - how many people were wounded in the attack
# - summary - summary of the attack happened
# - motive - motive of the attack
# 
# 
# 
# 
# ## Changelog 
# - added world terrorist attack animation (10/2/2018)
# - added word cloud of most used word during terrorist attack (10/2/2018)
# 
# ## Index of content
# 
# 1. Import the necessary libraries 
# 2. Import data
# 3. Peek in the data and simply it
# 4. Analyze, Analyze and Analyze
#  - 4.1 Number of terrorist activities every year
#  - 4.2 Top 10 terrorism affected countries
#  - 4.3 Top 10 terrorism affected regions
#  - 4.4 Attack methods used by terrorist
#  - 4.5 Favourite targets of terrorist
# 5. Map projections
# 6. Trends
#  - 6.1 Attack trend Vs Regions
#  - 6.2 GroupName Vs Activity Trend
# 7. Attack Vs Killed
# 8. Most Notorious Groups
# 9. Terrorist Attacks on the World (Animation)
# 10. Most frequently used word in terrorist attacks (word Cloud)

# # 1. Import the necessary libraries

# In[ ]:


# data analysis and wrangling
import numpy as np
import pandas as pd
import nltk
from wordcloud import WordCloud, STOPWORDS
from PIL import Image


# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from matplotlib import animation, rc
get_ipython().run_line_magic('matplotlib', 'notebook')
 
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import os
import io
import base64
from IPython.display import HTML, display
from scipy.misc import imread
import codecs
from subprocess import check_output
print(os.listdir("../input/"))
print(check_output(["ls", "../input"]).decode("utf8"))


# # 2. Import data

# In[ ]:


data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')


# # 3. Peek in the data and simplify it

# In[ ]:


data.head()


# As you can see that there are 135 different columns for this dataset. We will take only those columns which we need for our analysis. Also, The names of the columns are pretty weired. I will simplify them.
# 
# I have decided to take the following parameter in the analysis: 
# 
# |old column name | new column name|
# |----------------|----------------|
# |iyear            | year          |
# |imonth           | month |
# |iday       | day|
# |country_txt | country |
# |region_txt | region |
# |city | city|
# |latitude | latitude |
# |longitude | longitude |
# |attacktype1_txt | attackType |
# |target1 | target |
# |targtype1_txt | targetType |
# |gname | groupName |
# |weaptype1_txt | weaponType |
# |success | success|
# |nkill | killed |
# |nwound | wounded |
# |summary | summary|
# |motive | motive|
# 

# In[ ]:


data = pd.DataFrame(data[['iyear', 'imonth', 'iday', 'country_txt', 'region_txt', 'city', 'latitude', 'longitude', 'attacktype1_txt',
            'target1', 'targtype1_txt', 'gname', 'weaptype1_txt', 'success', 'nkill', 'nwound', 'summary', 'motive']])


# In[ ]:


data.head()


# In[ ]:


data.rename(columns={'iyear': 'year', 'imonth': 'month', 'iday': 'day', 'country_txt': 'country', 'region_txt': 'region',
                    'attacktype1_txt': 'attackType', 'target1': 'target', 'targtype1_txt': 'targetType', 'gname': 'groupName',
                    'weaptype1_txt': 'weaponType', 'nkill': 'killed', 'nwound': 'wounded'}, inplace=True)


# In[ ]:


data.head()


# # 4. Analyze, Analyze and Analyze
# 
# Let's look at the number of attack every year.

# #### 4.1 Number of terrorist activities every year

# In[ ]:


plt.figure(figsize=(14,7))
plt.tight_layout()
sns.countplot('year', data=data)
plt.xticks(rotation=90)
plt.title('Number of Terrorist Activities Each Year')
plt.show()


# It seems like the number of attack incresed every year. And, The attacks boomed from the year 2012

# #### 4.2 Top 10 terrorism affected countries

# In[ ]:


print('Top 10 terrorism affected countries:')
print(data['country'].value_counts().head(10))


# In[ ]:


plt.figure(figsize=(14,7))
sns.barplot(x= data['country'].value_counts()[:10].index, y=data['country'].value_counts()[:10].values, palette='rocket')
plt.title('Top 10  Terrorism Affected Countries')
plt.show()


# #### 4.3 Top 10 terrorism affected regions

# In[ ]:


print('Top regions in terms of terrorism:')
data['region'].value_counts().head(10)


# In[ ]:


plt.figure(figsize=(14,7))
sns.countplot(x= 'region', data=data, palette='rocket', order=data['region'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Top 10 Terrorism Affected Regions')
plt.show()


# #### 4.4 Attack methods used by terrorist

# In[ ]:


plt.figure(figsize=(14,7))
plt.tight_layout()
sns.countplot('attackType', data=data, order=data['attackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attack methods used by terrorists')
plt.show()


# #### 4.5 Favourite targets of terrorist

# In[ ]:


plt.figure(figsize=(17, 7))
plt.tight_layout()
sns.countplot(data['targetType'], order=data['targetType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Favourite targets of terrorist')
plt.show()


# In[ ]:


data.head()


# # 5. Map projections
# 
# #### Terrorism activities around the world.

# In[ ]:


#extract the data we are interested in

lat = data['latitude'].values
long = data['longitude'].values


# In[ ]:


fig = plt.figure(figsize=(20,10))
m = Basemap(projection='mill', resolution='c', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)

m.shadedrelief()
m.drawcoastlines()
m.drawcountries()
m.scatter(long, lat, latlon=True, c='r', alpha=0.4, s=3)
plt.show()


# #### Most active groups in particular regions

# In[ ]:


fig = plt.figure(figsize=(20,10))
m1 = Basemap(projection='mill', resolution='c', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20,
           lat_0=True, lat_1=True)

m1.shadedrelief()
m1.drawcoastlines()
m1.drawcountries()

color_list = ['#f97704','#324732', '#cb9d1d', '#cd3333','#1a1a1a','#ff753b','#490c66','#ea1d75','#00bbb3','#8b0000',
             '#440567','#043145','#da785b','#e00062']

group_list = data[data['groupName'].isin(data['groupName'].value_counts()[:14].index)]
regional_wise_active = list(group_list['groupName'].unique())

def draw_map(groupName, color, label):
    lat_g = list(group_list[group_list['groupName']==groupName].latitude)
    lon_g = list(group_list[group_list['groupName']==groupName].longitude)
    x, y = m1(lon_g, lat_g)
    m1.scatter(x, y, c=j , label=i, alpha=0.8, s=3)
    
for i, j in zip(regional_wise_active, color_list):
    draw_map(i, j, i)
    
leg = plt.legend(loc='lower left', frameon=True, prop={'size':10})
frame = leg.get_frame()
frame.set_facecolor('white')
plt.title('Most Active Group in Particular Regions')
plt.show()


# # 6. Trends

# #### 6.1 Attack trend Vs Regions

# In[ ]:


data_region = pd.crosstab(data.year, data.region)
data_region.plot(figsize=(18,6))
plt.show()


# #### 6.2 GroupName Vs Activity Trend

# In[ ]:


top10Groups = data[data['groupName'].isin(data['groupName'].value_counts()[1:15].index)]
pd.crosstab(top10Groups.year, top10Groups.groupName).plot(figsize=(18, 6))


# # 7. Type of attack Vs Region

# In[ ]:


region_attack = pd.crosstab(data.region, data.attackType)
region_attack.plot.bar(stacked=True, colormap='magma', figsize=(16, 8))
plt.show()


# # 7. Attack Vs Killed

# In[ ]:


num_of_attacks = data['country'].value_counts()[:10].to_frame()
num_of_attacks.columns=['attacks']
people_killed = data.groupby('country')['killed'].sum().to_frame()
num_of_attacks.merge(people_killed, left_index=True, right_index=True, how='left').plot.bar(width=0.8, figsize=(18,6))
plt.show()


# # 8. Most Notorious Groups 

# In[ ]:


fig = plt.figure(figsize=(14, 8))
sns.barplot(data['groupName'].value_counts()[1:15].values, data['groupName'].value_counts()[1:15].index, palette=('hot'))
plt.xticks(rotation=90)
plt.title('Most Notorious Groups')
plt.show()


# # 9. Terrorist Attacks on the world (Animation)
# 
# Let's also add the casualties[killed + wounded] figure in the dataset

# In[ ]:


data['casualties'] = data['killed'] + data['wounded']


# In[ ]:


fig = plt.figure(figsize=(19,10))
def animate(year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('Terrorism in world' + '\n' + str(year))
    m2 = Basemap(projection='mill', resolution='c', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)
    lat_ani = list(data[data['year']==year].latitude)
    long_ani = list(data[data['year']==year].longitude)
    x_ani, y_ani = m2(long_ani, lat_ani)
    m2.scatter(x_ani, y_ani, s=[i for i in data[data['year']==year].casualties], c= 'r')
    m2.drawcoastlines()
    m2.drawcountries()
    
ani = animation.FuncAnimation(fig,animate,list(data.year.unique()), interval = 1500)

ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif; base64, {0}" type="gif"  />'''.format(encoded.decode('ascii')))


# # 10. Most frequently used word in terrorist attacks
# 
# Let's make a word cloud to see it.

# In[ ]:


most_used_word = data['motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(most_used_word)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(i for i in words if i not in stopwords)
wc = WordCloud(stopwords=STOPWORDS, background_color='black').generate(" ".join(words_except_stop_dist))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




