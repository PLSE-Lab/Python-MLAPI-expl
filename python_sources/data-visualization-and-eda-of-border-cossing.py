#!/usr/bin/env python
# coding: utf-8

# ## Import important libraries and load the data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv')
df


# ## Check shape and view basic statistical details using describe()

# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.head()


# ## Pre-process the data by converting Border values to country names

# In[ ]:


df["Border"] = ["Mexico" if i == "US-Mexico Border" else "Canada" for i in df.Border]

df
# ## Find unique values in each column

# In[ ]:


print("Unique port name: ",len(df["Port Name"].unique()))
print("Unique States: ",len(df["State"].unique()))
print("Unique Port Codes: ",len(df["Port Code"].unique()))
print("Unique Borders: ",len(df["Border"].unique()))
print("Unique measure of entries: ",len(df["Measure"].unique()))


# ## Check for Null values

# In[ ]:


df.isnull().any()


# ## import folium to visualize the data on a map

# In[ ]:


import folium
world_map =folium.Map(location=[37.09, -95.71],zoom_start=4)
world_map


# ## We need to format the Location column so that we can use it to plot the location on the map

# In[ ]:


#temp = df['Location']
df['Location']


# ##  First we strip POINT infront of our data

# In[ ]:


df['Location'] = df['Location'].str.lstrip('POINT')
df['Location']


# ## Strip parenthsis from both sides

# In[ ]:


'''df['Location'] = df['Location'].map(lambda x: x.lstrip('('))
df['Location'].str.lstrip(')')
df['Location']'''
df['Location']= df['Location'].str.replace(r'\)', '')
df['Location']= df['Location'].str.replace(r'\(', '')
df['Location']


# In[ ]:


df['Location'].shape


# ## Convert into two different columns

# In[ ]:


temp = df['Location']
temp.to_frame()
temp[0]
temp.shape
temp = pd.DataFrame(df['Location'].str.split(' ',2).tolist(),columns = ['z','Y','X'])
temp
temp.X.shape


# ## Insert into the dataframe

# In[ ]:


df.insert(8, "X", temp.X, True) 
df.insert(9,"Y",temp.Y,True)


# In[ ]:


df


# In[ ]:


df.head()


# ## Plot 1st 1000 onto the map

# In[ ]:


df =df.iloc[:1000,:]
df.shape
df['Port Name']


# In[ ]:


#Create entry_map for plotting
entry_map =folium.Map(location=[37.09, -95.71],zoom_start=4)
entry_map


# ## Add point of entries on the map

# In[ ]:


entries = folium.map.FeatureGroup()
for lat,lon in zip(df.X,df.Y):
    entries.add_child(
    folium.CircleMarker(
    [lat,lon],
    radius=5,
    color='yellow',
    fill=True,
    fill_color='blue',
    fill_opacity=0.6,
        )
    )
entry_map.add_child(entries)


# ## We can also add marker on the map with labels

# In[ ]:


latitudes = list(df.X)
longitudes = list(df.Y)
labels = list(df['Port Name'])

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(entry_map)    
    
# add incidents to map
entry_map.add_child(entries)


# ## If the markers look more congested we can remove them and add labels on points

# In[ ]:


entry_map = folium.Map(location=[37.09, -95.71], zoom_start=4)

# loop through the 100 crimes and add each to the map
for lat, lng, label in zip(df.X, df.Y, df.Measure):
    folium.CircleMarker(
        [lat, lng],
        radius=5, # define how big you want the circle markers to be
        color='yellow',
        fill=True,
        popup=label,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(entry_map)

# show map
entry_map


# ## We can make cluster maps. Whenever we zoon the clusters will split

# In[ ]:


from folium import plugins

cluster_map = folium.Map(location = [37.09, -95.71], zoom_start = 4)

entries = plugins.MarkerCluster().add_to(cluster_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(df.X, df.Y, df['Port Name']):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(entries)

# display map
cluster_map


# In[ ]:





# ## Violinplots of State Vs Value (no of people crossing the border)

# In[ ]:


plt.figure(figsize=(18,5))
ax = sns.violinplot(x="State", y="Value",
                    data=df[df.Value < 30000],
                    scale="width", palette="Set3")


# ## Pie-chars of different states. Limit the states to Top 6 states 

# In[ ]:


df_state =df.groupby('State',axis=0).sum()
df_state=df_state.drop(columns=['Port Code'])
#Top 5 States
df_state = df_state.sort_values(['Value'], ascending=[0])
df_state
limit = 6
df_state = df_state.iloc[:limit,:]
df_state


# In[ ]:


explode_list = [0, 0, 0, 0.1,0.1,0.1]

df_state['Value'].plot(kind='pie',figsize=(15,6), autopct='%1.1f%%',startangle=90,shadow=True,pctdistance=1.12,explode=explode_list,labels=None)
plt.title('Border crossed based on the States',y=1.12)
plt.legend(labels=df_state.index, loc='upper left')
plt.axis('equal') 
plt.show()


# ## Pie-chars of different Measures of entry. Limit the Measures to Top 6. 

# In[ ]:


df_measure=df.groupby('Measure',axis=0).sum()
df_measure=df_measure.drop(columns=['Port Code'])
df_measure=df_measure.sort_values(['Value'],ascending=[0])
limit=6
df_measure=df_measure.iloc[:limit,:]
df_measure


# In[ ]:


explode_list = [0, 0,0,0.1,0.1,0.1]

df_measure['Value'].plot(kind='pie',figsize=(15,6), autopct='%1.1f%%',startangle=90,shadow=True,pctdistance=1.12,explode=explode_list,labels=None)
plt.title('Border crossed based on measures of crossing',y=1.12)
plt.legend(labels=df_measure.index, loc='upper left')
plt.axis('equal') 
plt.show()


# ## Pie-Chart of number of entries from Mexico and Canada Border

# In[ ]:


df_country=df.groupby('Border').sum()
df_country=df_country.drop(columns=['Port Code'])
df_country


# In[ ]:


df_country['Value'].plot(kind='pie',figsize=(15,6),autopct='%1.1f%%',startangle=90,shadow=True,pctdistance=1.12,labels=None)
plt.title('Border crossed based on country',y=1.12)
plt.legend(labels=df_country.index, loc='upper left')
plt.axis('equal') 
plt.show()


# ## Pie-chart of entries from different Ports. Limit upto 10 Ports

# In[ ]:


df_port=df.groupby('Port Name').sum()
df_port=df_port.drop(columns=['Port Code'])
df_port=df_port.sort_values(['Value'],ascending=[0])
df_port.head(10)
df_port = df_port.iloc[:10,:]
df_port


# In[ ]:


exp_list = [0, 0,0,0,0,0,0.1,0.1,0.1,0.1]

df_port['Value'].plot(kind='pie',figsize=(15,6), autopct='%1.1f%%',startangle=90,shadow=True,pctdistance=1.12,explode=exp_list,labels=None)
plt.title('Border crossed based on measures of crossing',y=1.12)
plt.legend(labels=df_port.index, loc='upper left')
plt.axis('equal') 
plt.show()


# In[ ]:


df


# ## Bar Chart of States Vs no of entries(Values) . We can see that california has the highest no fo entries

# In[ ]:


df_state =df.groupby('State',axis=0).sum()
df_state=df_state.drop(columns=['Port Code'])
#Top 5 States
df_state = df_state.sort_values(['Value'], ascending=[0])
df_state


# In[ ]:


df_state.plot(kind='bar', figsize=(10, 6), rot=90) 


# ## Bar Chart of Measure of Entry Vs No of entries(Values) . We can see that no of entries are more by personal entry passengers.

# In[ ]:


df_measure=df.groupby('Measure',axis=0).sum()
df_measure=df_measure.drop(columns=['Port Code'])
df_measure=df_measure.sort_values(['Value'],ascending=[0])
df_measure


# In[ ]:


df_measure.plot(kind='bar', figsize=(10, 6), rot=90) 


# ## Bar chart of No of entries from Mexico Vs Canada. We can see that entries from Mexico are more.

# In[ ]:


df_country.plot(kind='bar',figsize=(10,6),rot=90)


# ## Bar chart of Ports vs no of entries from that port

# In[ ]:


df_port.plot(kind='bar',figsize=(10,6),rot=90)


# ## Bar chart of State Vs no fo entries using Seaborn.

# In[ ]:


#Using Seaborn
import seaborn as sns

plt.figure(figsize=(18,7))
sns.barplot(x = df_state.index, y = "Value", data = df_state)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


plt.figure(figsize=(18,7))
sns.barplot(x = "State", y = "Value", data = df)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


plt.figure(figsize=(18,7))
sns.barplot(x = "State", y = "Value",hue='Border', data = df)
plt.xticks(rotation=45)
plt.show()


# ## Bar Chart of Measure of entry vs no of entries using that Measure using Seaborn.

# In[ ]:


plt.figure(figsize=(18,7))
sns.barplot(x = df_measure.index, y = "Value", data = df_measure)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


plt.figure(figsize=(18,7))
sns.barplot(x = "Measure", y = "Value", data = df)
plt.xticks(rotation=45)
plt.show()


# ## Bar Chart of Port vs No of entries from that port

# In[ ]:


plt.figure(figsize=(18,7))
sns.barplot(x = df_port.index, y = "Value", data = df_port)
plt.xticks(rotation=45)
plt.show()


# ## Boxenplot of Mexico VS Canada 

# In[ ]:


sns.boxenplot(x="Border", y="Value",
              color="b",
              scale="linear", data=df)
plt.show()


# In[ ]:


df


# # Let's make a word cloud of all the ports. Convert all values in Port Names into text.

# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
text=" ".join(str(port) for port in df['Port Name'])
text


# ## Wordcloud with max 100 words 

# In[ ]:


word_cloud = WordCloud(max_words=100,background_color='white').generate(text)
plt.figure(figsize=(15,10))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Count plots of Measure vs Border

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(15,5))
chart1=sns.countplot(df['Measure'],hue='Border',data=df,ax=ax[1])
chart1.set_xticklabels(chart1.get_xticklabels(), rotation=90)
chart2=sns.countplot(df['Measure'],data=df,ax=ax[0])
chart2.set_xticklabels(chart2.get_xticklabels(), rotation=90)
ax[0].title.set_text("Measure:data")
ax[1].title.set_text("Measure:Border")


# ## Count plots of State vs Border

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(15,5))
chart1=sns.countplot(df['State'],hue='Border',data=df,ax=ax[1])
chart1.set_xticklabels(chart1.get_xticklabels(), rotation=45)
chart2=sns.countplot(df['State'],data=df,ax=ax[0])
chart2.set_xticklabels(chart2.get_xticklabels(), rotation=90)
ax[0].title.set_text("State:data")
ax[1].title.set_text("State:Border")


# ## Let's  plot line chart. We read the dataset into a new dataframe df_date

# In[ ]:


df_date = pd.read_csv('../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv')
df_date


# ## Convert the Data columns into pandas readable format and set it as index

# In[ ]:


df_date["DateAsDateObj"] = pd.to_datetime(df_date.Date)
df_date = df_date.set_index("DateAsDateObj")
df_date


# ## Line chart of no of entries(Value) Vs years 

# In[ ]:


dataForPlot = df_date.resample("M").mean()
dataForPlot.loc[:,["Value"]].plot()


# ## Line chart of no of entries(Value) Vs months

# In[ ]:


dataForPlot = df_date.resample("M").mean()
dataForPlot.loc[:,["Value"]].plot()


# ## Line chart of all the Measures of entries Vs no of entries over the years

# In[ ]:


##Plot  by year
dataForPlot = df_date.loc[:,["Measure","Value"]]
dataForPlot = dataForPlot.groupby("Measure").resample("Y").mean()
dataForPlot.reset_index().pivot(index="DateAsDateObj",columns="Measure", values="Value").plot(subplots=True, figsize=(8,14))


# ## Line chart of all the states vs no of entries over the years

# In[ ]:


dataForPlot = df_date.loc[:,["State","Value"]]
dataForPlot = dataForPlot.groupby("State").resample("Y").mean()
dataForPlot.reset_index().pivot(index="DateAsDateObj",columns="State", values="Value").plot(subplots=True, figsize=(8,14))


# ## Line chart of all the measures of entries Vs no of entries over months

# In[ ]:


dataForPlot = df_date.loc[:,["Measure","Value"]]
dataForPlot = dataForPlot.groupby("Measure").resample("M").mean()
dataForPlot.reset_index().pivot(index="DateAsDateObj",columns="Measure", values="Value").plot(subplots=True, figsize=(8,14))


# ## Line Chart of all the States Vs no of entries over months
# 

# In[ ]:


dataForPlot = df_date.loc[:,["State","Value"]]
dataForPlot = dataForPlot.groupby("State").resample("M").mean()
dataForPlot.reset_index().pivot(index="DateAsDateObj",columns="State", values="Value").plot(subplots=True, figsize=(8,14))


# In[ ]:




