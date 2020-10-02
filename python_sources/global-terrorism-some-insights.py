#!/usr/bin/env python
# coding: utf-8

# Terrorism is the major problem faced worldwide today. Even there are various reasons behind a terrorist attack, the lives a innocent people are at stake. It is those innocent people who die first who dont have anything to do with the motive of the terrorists. 
# 
# Terrorism should be erradicated completely from the society at all levels. Even terrorists have their version of story, it should be told at the right place. Killing innocent people will bring no good to them.
# 
# Lets analyse the dataset that contains the terroist attacks that happened worldwide.

# In[ ]:


# importing all the necessary packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# importing the dataset
terror=pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')


# In[ ]:


# Preview the raw dataset
terror.head ()


# Most of the column names are confusing. Also there are various unnecessary features. Lets remove everything and rename the useful features into more meanigfull names : 

# In[ ]:


# Renaming the features
terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)


# In[ ]:


# Retaining only usefull features
terror=terror[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]


# In[ ]:


# Preview the cleaned dataset
terror.head ()


# Feature Engineering :
# Creating new features from the existing features.

# In[ ]:


# Creating new feature "Casualities" by adding "Killed" and "Wounded" features
terror['casualities']=terror['Killed']+terror['Wounded']


# In[ ]:


terror.head (3)


# One of the important thing to know about your dataset is to know how many null values are there in every features of the

# In[ ]:


terror.isnull().sum()


# Here, motive is the feature that contains most of the null values. This is obvious that the terrorist group which involved in the attack have to confess the motive. Untill that nobody know the motive of the attack.

# In[ ]:


terror['Group'].value_counts().head (5)


# Here, 78306 attacks are unknown attacks and there is no data on which group did them. Also the motive for 121764 attacks remain unknown. So we know who did 43458 attacks but those groups didn't revealed the motive.

# Some obvious insights :

# In[ ]:


print('Country with Highest Terrorist Attacks:',terror['Country'].value_counts().index[0])
print('Regions with Highest Terrorist Attacks:',terror['Region'].value_counts().index[0])
print('Maximum people killed in an attack are:',terror['Killed'].max(),'that took place in',terror.loc[terror['Killed'].idxmax()].Country)
print('Maximum casualties of', terror['casualities'].max(), 'happened in a single attack in', terror.loc[terror['casualities'].idxmax()].Country)


# Plotting the global terrorist activites trend on a time scale :

# In[ ]:


plt.subplots(figsize=(15,5))
sns.countplot('Year',data=terror)
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# From the above graph, its very clear that the global terrorist activities hit a lowest point at the end of 19th century. But all of a sudden, it hiked to a wooping 18,000 mark in 2014.
# 
# We need to ask why terrorism raised after 2000. There may be various reasons for this that may not be available in the dataset.

# Which regions are affected by terrorism the most :

# In[ ]:


plt.subplots(figsize=(15,5))
sns.countplot('Region',data=terror,order=terror['Region'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Region')
plt.show()


# What are the usual targets of the terrorists :

# In[ ]:


plt.subplots(figsize=(15,5))
sns.countplot('Target_type',data=terror,order=terror['Target_type'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Target victims')
plt.show()


# Irony here is that innocent people are the catagory that get affected the most. The reason may be that the normal people are the easiest targets while political leaders will always be in the highest security.

# Plotting the locations where the terrorist attacks claimed lesser than 100 casualities :

# In[ ]:


m3 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180)
lat_100=list(terror[terror['casualities']<100].latitude)
long_100=list(terror[terror['casualities']<100].longitude)
x_100,y_100=m3(long_100, lat_100)
m3.drawcoastlines()
m3.drawcountries()
m3.plot(x_100, y_100,'go',markersize=0.5,color = 'g')
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.title("Terroist attacks with lesser than 100 casualities")


# From the above map, India, Middle east and European countries are the favorite targets of the terrorist groups.

# Plotting the locations where the terrorist attacks claimed more than 100 casualities :

# In[ ]:


m3 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180)
lat_100=list(terror[terror['casualities']>=100].latitude)
long_100=list(terror[terror['casualities']>=100].longitude)
x_100,y_100=m3(long_100, lat_100)
m3.drawcoastlines()
m3.drawcountries()
m3.plot(x_100, y_100,'go',markersize=5,color = 'r')
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.title("Terroist attacks with more than 100 casualities")


# Most of the major attacks were also happened in Middle east and contries like Pakistan, Afganisthan etc. 

# Which regions are facing more terrorist attacks worldwide :  

# In[ ]:


terror_region=pd.crosstab(terror.Year,terror.Region)
terror_region.plot(color=sns.color_palette('Set2',12))
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()


# Middle east, North Africa, South Asia are the top most affected regions by terrorism. In this chart also we can see a complete drop in the global terrorism rate at the end of 19th century. 
# 
# But after that, there was a sudden hike in the terrorism worldwide. So its very clear that something that happened around this time led to this hike. What was that ?
# 
# **9/11 attack ?**

# Lets analyse which type of attack is famous in every region :

# In[ ]:


terror_type = pd.crosstab(terror.Region,terror.AttackType)
terror_type


# In[ ]:


terror_type.plot.barh(stacked=True, width=1)
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()


# Frrom the chart above, it is clear that bombing and explosion is the favourite attack of terrorist groups in every region.
# 
# This may be the reason why most number of civilians are killed in the attacks as a single explosion claims lots of lives.

# What are the top most affected countries from terrorism worl wide :

# In[ ]:


# Top 20 countries affected by terrorism
coun_terror=terror['Country'].value_counts()[:20].to_frame() # to_frame() function will generate a dataframe out of the results. 
coun_terror.columns=['Attacks']
coun_terror


# In[ ]:


coun_terror.plot.bar()
fig=plt.gcf()
fig.set_size_inches(18,6)


# Lets mix and match the features and see if we get any insights out of it :

# We are going to plot number of attacks and number of casualities in a bar chat for every significant country :

# In[ ]:


# This will give the number of people killed in every country collectively
coun_kill=terror.groupby('Country')['Killed'].sum().to_frame() 
coun_kill.head ()


# In[ ]:


# This will merge the coun_terror and coun_kill datasets 
# and give top 20 countries with no of attacks and no of people killed 
attack_kill = coun_terror.merge(coun_kill,left_index=True,right_index=True,how='left')
attack_kill


# Lets plot both of these datasets in a single barchat :

# In[ ]:


# Plotting the same on a bar chart
attack_kill.plot.bar()
fig=plt.gcf()
fig.set_size_inches(18,6)


# From the above chart, we can come to some obvious conclusions :
# * In some of the Middle east contries like Iraq, number of casualities is more than twice the number of attacks. High density of population may be the reason. Poor prevention and security may also be a reason for this.
# * In developed contries like UK, Spain, France, no of attacks is more than the number of casulaities. This means that these contries are better is safety and they are good in prevention before a terror attack happens. Low population density may also be a reason for this.

# Which terror groups are highly active ?

# In[ ]:


# To find which terrorist group is most active
coun_group=terror['Group'].value_counts()[:20].to_frame()
coun_group


# In[ ]:


coun_group[1:20].plot.bar()
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.title ("Most active terrorist groups")


# Plotting th activities of top 10 groups in a time series plot :

# In[ ]:


top_groups10=terror[terror['Group'].isin(terror['Group'].value_counts()[1:11].index)]
noto_gro = pd.crosstab(top_groups10.Year,top_groups10.Group)
noto_gro.plot(color=sns.color_palette('Paired',10))
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()


# The above chart clearly shows that all the terror groups are active for some particular time and stopped their activities after some time before 2000. But after 2000, we can see lot of groups emerging.
# 
# We can ask questions like what happened after the year 2000 ? What is the moto for these groups ? Is somebody creating these groups and funding them for their own goodwill ?
# 
# Most notorious groups like Taliban, ISIS emerged after 2000.

# What is the favourite attacking style of terrorists ?

# In[ ]:


# Favorite attacking style world wide
sns.countplot(terror['AttackType'], order = terror['AttackType'].value_counts().index)
plt.xticks(rotation=90)


# Bombing and explosions plays a major role in terrorist attacks.

# In[ ]:





# Lets plot the terrorist attcks in India :

# In[ ]:


terror_india=terror[terror['Country']=='India']


# In[ ]:


terror_india.head ()


# Attacks with lesser than 100 casualities :

# In[ ]:


m3 = Basemap(projection='mill', llcrnrlat=5,urcrnrlat=37,llcrnrlon=67,urcrnrlon=99)
lat_100=list(terror[terror['casualities']<100].latitude)
long_100=list(terror[terror['casualities']<100].longitude)
x_100,y_100=m3(long_100, lat_100)
m3.drawcoastlines()
m3.drawcountries()
m3.plot(x_100, y_100,'go',markersize=1,color = 'g')
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.title("Terroist attacks with lesser than 100 casualities")


# Its clear from the above map that states like Andhra, Bihar, Orissa and J&K are the most affected states from terrorism. Presence of Naxals is one of the main reason for this.

# Attacks with more than 100 casualities :

# In[ ]:


m3 = Basemap(projection='mill', llcrnrlat=5,urcrnrlat=37,llcrnrlon=67,urcrnrlon=99)
lat_100=list(terror[terror['casualities']>=100].latitude)
long_100=list(terror[terror['casualities']>=100].longitude)
x_100,y_100=m3(long_100, lat_100)
m3.drawcoastlines()
m3.drawcountries()
m3.plot(x_100, y_100,'go',markersize=5,color = 'r')
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.title("Terroist attacks with lesser than 100 casualities")


# Lets explore more on this topic in upcoming versions
