#!/usr/bin/env python
# coding: utf-8

# Bangladesh is a developing country. During this lockdown situation, we are struggling to continue our surveys. The data was taken from BUET: a renowned enginnering university in Bangladesh. The survey was done to check the feasibility of online classes

# In[ ]:


## importing file
import pandas as pd
import numpy as np

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns; sns.set()
import plotly.express as px
plt.style.use('seaborn-whitegrid')
import missingno as msn

#map
import geopandas
from shapely.geometry import Point
import plotly.graph_objects as go


# In[ ]:


df = pd.read_csv("../input/online-classes-survey-bangladesh/EEE17BUET.csv")


# # Feature analysis

# ## District
# Where ae they living currently

# In[ ]:


for i in range(len(df)):
    listi = df.iloc[i]['District'].split()
    new_value = listi[0]
    df["District"]= df["District"].str.replace(df.iloc[i]['District'], new_value, case = True) 
print(df.District.value_counts())


# In[ ]:


# Getting to know GEOJSON file:
country = geopandas.read_file("../input/geometry/bangladesh.json")

#check the data
country.head()
print(type(country))
type(country.geometry)
type(country.geometry[0])

#plot the map
fig = country.plot(figsize = (200, 50))

#create an empty dataframe to store the lat,long, name, and cases
myDF = pd.DataFrame()
myDF['Lat'] = country.geometry.centroid.x
myDF['Long'] = country.geometry.centroid.y
myDF['Zilla'] = country['NAME_2']

#write the name of all zillas in a list
allZilla = '\n'.join(zillas for zillas in list(myDF.iloc[:,2]))


# In[ ]:


for i in range(len(df)):
    if((df.iloc[i]["District"] in allZilla) == False): print(df.iloc[i]["District"])


# In[ ]:


df["District"]= df["District"].str.replace("Chapai", "Nawabganj", case = True)
df["District"]= df["District"].str.replace("Netrokona", "Netrakona", case = True)
df["District"]= df["District"].str.replace("Moulvibazar", "Maulvibazar", case = True)
df["District"]= df["District"].str.replace("Brahmanbaria", "Brahamanbaria", case = True)


# In[ ]:


for i in range(len(df)):
    if((df.iloc[i]["District"] in allZilla) == False): print(df.iloc[i]["District"])


# In[ ]:


new = pd.DataFrame()
new = df['District']
caselist = []
listi = new.value_counts().reset_index().values.tolist()

onlyNames = []
for i in range(len(listi)):
    onlyNames.append(listi[i][0])
print(onlyNames)

for i in range(len(country['NAME_2'])):
    if((country.iloc[i]['NAME_2'] in onlyNames) == False): caselist.append(0)
    else: 
        ind = onlyNames.index(country.iloc[i]['NAME_2'])
        caselist.append(listi[ind][1])

print(caselist)


# In[ ]:


country.head()


# In[ ]:


country['cases'] = caselist


# In[ ]:


country.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
myPlt = country.plot(column = 'cases',cmap='PRGn',ax=ax,edgecolor='k')
ax.set_axis_off()
ax.set(title='People staying in respective areas')

fig = myPlt.get_figure()
fig.savefig("output_country.png")


# ## Required Device
# Do they have the required device to continue online classes

# In[ ]:


print(df.Required_device.value_counts())
splot = sns.barplot(df.Required_device.value_counts().index, df['Required_device'].value_counts().values, alpha=0.8)
splot.set_title('Have Required Device')
for p in splot.patches:
    splot.annotate(format(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

fig = splot.get_figure()
fig.savefig("required_device.png")


# ##  Broadband
# Do people have proper broadband connections

# In[ ]:


print(df.Broadband.value_counts())
splot = sns.barplot(df.Required_device.value_counts().index, df['Broadband'].value_counts().values, alpha=0.8)
splot.set_title('Have Broadband connection')
for p in splot.patches:
    splot.annotate(format(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    
fig = splot.get_figure()
fig.savefig("Broadband.png")


# ## two_month_net
# If they can affort buying mobile data for 2 months straight

# In[ ]:


print(df.two_month_net.value_counts())
splot = sns.barplot(df.Required_device.value_counts().index, df['two_month_net'].value_counts().values, alpha=0.8)
splot.set_title('Support two month net connecton')
for p in splot.patches:
    splot.annotate(format(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

fig = splot.get_figure()
fig.savefig("two_month_net.png")


# ## Cellular net type

# In[ ]:


print(df.Cellular.value_counts())

figu = df.Cellular.value_counts().plot(kind='pie', 
                                    figsize = (5,5), 
                                    autopct = lambda p : '{:,.0f}'.format(p * df.Cellular.count()/100) , 
                                    subplots = True,
                                    colormap = "plasma_r", 
                                    title = 'Cellular Network Type', 
                                    fontsize = 15)
fig.savefig("cellular_network.png")


# ## Net speed
# Here I just saw people's net connection

# In[ ]:


print(df.Net_speed.value_counts())


# ## Hall
# Hall is synonymous to dorm, 'Resident' means the people who stay in the 'Hall', attached means the people who stay at home

# In[ ]:


print(df.Hall.value_counts())

df.Hall.value_counts().plot(kind='pie', 
                                    figsize = (5,5), 
                                    autopct = lambda p : '{:,.0f}'.format(p * df.Hall.count()/100) , 
                                    subplots = False,
                                    colormap = 'Accent', 
                                    title = 'Resident/Attached', 
                                    legend= True, 
                                    fontsize = 15)


# ## Where left the books
# As this was a case of emergency, many people left their books in the Hall

# In[ ]:


print(df.Books.value_counts())

df.Books.value_counts().plot(kind='pie', 
                                    figsize = (5,5), 
                                    autopct = lambda p : '{:,.0f}'.format(p * df.Books.count()/100) , 
                                    subplots = False,
                                    colormap = 'coolwarm_r', 
                                    title = 'Where our books are', 
                                    legend= True, 
                                    fontsize = 15)


# ## Class System
# //this needs to be checked if dataset is changed
# People were given 4 options, people could choose any of them, multiple selection was allowed too. So manual processing was demanded

# In[ ]:


print(df.Class_System.value_counts())

#process the datas manually
recorded = []
recorded.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Uploaded Lecture Notes (PPT, Word or PDF)']))
recorded.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials']))
recorded.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Online Conference Platforms (like Zoom), Uploaded Lecture Notes (PPT, Word or PDF)']))
recorded.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Online Conference Platforms (like Zoom), Facebook Live, Uploaded Lecture Notes (PPT, Word or PDF)']))
recorded.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Facebook Live, Uploaded Lecture Notes (PPT, Word or PDF)']))
recorded.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Online Conference Platforms (like Zoom)']))

uploaded = []
uploaded.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Uploaded Lecture Notes (PPT, Word or PDF)']))
uploaded.append(len(df.loc[df['Class_System'] == 'Uploaded Lecture Notes (PPT, Word or PDF)']))
uploaded.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Online Conference Platforms (like Zoom), Uploaded Lecture Notes (PPT, Word or PDF)']))
uploaded.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Online Conference Platforms (like Zoom), Facebook Live, Uploaded Lecture Notes (PPT, Word or PDF)']))
uploaded.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Facebook Live, Uploaded Lecture Notes (PPT, Word or PDF)']))
uploaded.append(len(df.loc[df['Class_System'] == 'Online Conference Platforms (like Zoom), Facebook Live, Uploaded Lecture Notes (PPT, Word or PDF)']))
uploaded.append(len(df.loc[df['Class_System'] == 'Online Conference Platforms (like Zoom), Uploaded Lecture Notes (PPT, Word or PDF)']))

conference = []
conference.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Online Conference Platforms (like Zoom), Uploaded Lecture Notes (PPT, Word or PDF)']))
conference.append(len(df.loc[df['Class_System'] == 'Online Conference Platforms (like Zoom)']))
conference.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Online Conference Platforms (like Zoom), Facebook Live, Uploaded Lecture Notes (PPT, Word or PDF)']))
conference.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Facebook Live, Uploaded Lecture Notes (PPT, Word or PDF)']))
conference.append(len(df.loc[df['Class_System'] == 'Online Conference Platforms (like Zoom), Facebook Live, Uploaded Lecture Notes (PPT, Word or PDF)']))
conference.append(len(df.loc[df['Class_System'] == 'Online Conference Platforms (like Zoom), Uploaded Lecture Notes (PPT, Word or PDF)']))

facebook = []
facebook.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Online Conference Platforms (like Zoom), Facebook Live, Uploaded Lecture Notes (PPT, Word or PDF)']))
facebook.append(len(df.loc[df['Class_System'] == 'Recorded Video Tutorials, Facebook Live, Uploaded Lecture Notes (PPT, Word or PDF)']))
facebook.append(len(df.loc[df['Class_System'] == 'Online Conference Platforms (like Zoom), Facebook Live, Uploaded Lecture Notes (PPT, Word or PDF)']))
facebook.append(len(df.loc[df['Class_System'] == 'Facebook Live']))


# In[ ]:


labels = ['Recorded Video', 'Online Conference Platforms', 'Uploaded Lecture Notes (PPT, Word or PDF)', 'Facebook Live']
sizes = [sum(recorded), sum(uploaded), sum(conference), sum(facebook)]

import plotly.graph_objects as go

# Use textposition='auto' for direct text
fig = go.Figure(data=[go.Bar(
            x=labels, y=sizes,
            text=sizes,
            textposition='auto',
        )])

fig.update_traces(marker_color='rgb(158,100,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=3)
fig.update_layout(title_text='Feasible system for classtaking (one selected multiple option)')
fig.show()


# ## Prefered assessment methods
# The survey was made to see how they prefer their daily assessments should be taken in this situation
# 

# In[ ]:


print(df.Ct.value_counts())

fig = plt.figure(figsize= (16,4))
splot = sns.barplot(df.Ct.value_counts().index, df['Ct'].value_counts().values, alpha=0.8, palette = sns.color_palette('spring'))
splot.set_title('Preferred assessment methods')
for p in splot.patches:
    splot.annotate(format(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
fig.show()

fig = splot.get_figure()
fig.savefig("preferred_exam.png")


# ## Comments
# People gave their opinion about online class, so I had to manually go through them and write the end result

# In[ ]:


print(df.Comments.value_counts())
Comments = ["feasible for EVERYBODY(NOT MAJORITY)",
        "internet slow",
        "no broadband",
        "location can't provide live stream for zoom",
        "lives in Dhaka and has broadband connection, still often disconnected for bad weather",
        "go outside for good connection",
        "recorded class ok",
        "loadshedding due to bad weather",
        "online class to lessen pressure of offline class",
        "must be recorded",
        "overwhelmed by the current epidemic, can't concentrate",
        "need internet cost fund",
        "power cut can harm attendence",
        "broadband connection not cost effective",
        "pdf ok",
        "don't have all the books",
        "impossible if long term",
        "class test replaced by assignment",
        "taking attendance postponed",
        "make zoom free for students and teachers"]
for i in range(len(Comments)):
    print(Comments[i])

with open("comments.txt", "w") as outfile:
    outfile.write(Comments)


# In[ ]:


print(len(df))

