#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import folium
import matplotlib.pyplot as plt
import seaborn as sns
from folium import plugins
import branca.colormap as cm
import missingno as msno


# In[ ]:


data=pd.read_csv("/kaggle/input/airbnb-istanbul-dataset/AirbnbIstanbul.csv")
df=data.copy()


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes.to_frame()


# In[ ]:


df.last_review=pd.to_datetime(df.last_review)


# In[ ]:


df[df.duplicated(subset=["host_name","host_id","neighbourhood","name","id"])]


# In[ ]:


df.name.nunique()


# In[ ]:


pd.DataFrame(df.isnull().sum()/len(df),columns=["missing_rate"])


# In[ ]:


msno.matrix(df)


# In[ ]:


df.drop("neighbourhood_group",axis=1,inplace=True)


# In[ ]:


msno.bar(df)


# In[ ]:


msno.heatmap(df)


# In[ ]:


df.eq(0).sum().to_frame()


# In[ ]:


m = folium.Map([42 ,29], zoom_start=5,width="%100",height="%100")
locations = list(zip(df.latitude, df.longitude))
#icons = [folium.Icon(icon="airbnb", prefix="fa") for i in range(len(locations))]

cluster = plugins.MarkerCluster(locations=locations,popups=df["neighbourhood"].tolist())
m.add_child(cluster)
m


# In[ ]:


m = folium.Map(location=[41,29],width="%100",height="%100")
for i in range(len(locations)):
    folium.CircleMarker(location=locations[i],radius=1).add_to(m)
m


# In[ ]:


airbnb=df[["latitude","longitude","price"]]
min_price=df["price"].min()
max_price=df["price"].max()
min_price,max_price


# In[ ]:


df.price.describe().to_frame()


# In[ ]:


m = folium.Map(location=[41,29],width="%100",height="%100")
#colormap = cm.LinearColormap(['green', 'yellow', 'red'],vmin=min_price, vmax=max_price)
colormap = cm.StepColormap(colors=['green','yellow','orange','red'] ,index=[min_price,105,190,327,max_price],vmin= min_price,vmax=max_price)
#cm.LinearColormap.to_step

for loc, p in zip(zip(airbnb["latitude"],airbnb["longitude"]),airbnb["price"]):
    folium.Circle(
        location=loc,
        radius=2,
        fill=True,
        color=colormap(p),
        #popup=
        #fill_opacity=0.7
    ).add_to(m)
#colormap.caption = 'Colormap Caption'
#m.add_child(colormap)

m


# In[ ]:


plt.figure(figsize=(15,6))
sns.barplot(x="neighbourhood",y="price",data=df)
plt.xticks(rotation=90);
plt.grid();


# In[ ]:


plt.figure(figsize=(15,6))
df.groupby("neighbourhood")["price"].mean().sort_values(ascending=False).plot.bar(color="purple");


# In[ ]:


plt.figure(figsize=(15,6))
df.groupby("neighbourhood")["price"].mean().nlargest(10).plot.bar(color="r")


# In[ ]:


plt.figure(figsize=(15,6))
df.groupby("neighbourhood")["price"].mean().nsmallest(10).plot.bar(color="g")


# In[ ]:


df.loc[df.price==0,:]


# In[ ]:


df[df.neighbourhood=="Beyoglu"]["price"].describe()


# In[ ]:



sns.catplot(x="price",y="neighbourhood",hue="room_type",data=df,kind="bar",height=30).set_yticklabels(fontsize=20).set_xticklabels(fontsize=20).set_xlabels(fontsize=20).set_ylabels(fontsize=20)
plt.xticks(rotation=90);


# In[ ]:


sns.catplot(x="room_type",y="price",data=df,kind="bar",aspect=2);


# In[ ]:


plt.subplot(121)
plt.pie(df["room_type"].value_counts(),
        labels=df.room_type.value_counts().index,
        shadow=True,
        autopct='%1.1f%%',
        radius=2,
        startangle=140);
plt.title("Room Types", 
          bbox={'facecolor':'0.8', 'pad':2},loc="center");
plt.show();


# In[ ]:



plt.pie(df["neighbourhood"].value_counts().nlargest(15),
        labels=df.neighbourhood.value_counts().index[:15],
        shadow=True,
        autopct='%1.1f%%',
        radius=2,
        startangle=140);
plt.title("Number of Airbnb According to Neighbourhoods", 
          bbox={'facecolor':'0.8', 'pad':2},loc="center");
plt.show();


# In[ ]:


corr=df[["minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365","price"]].corr().abs()
corr


# In[ ]:


print("Unique Host Name Number :",df.host_name.nunique())


# In[ ]:


plt.figure(figsize=(15,6))
df.groupby("host_name")["number_of_reviews"].sum().sort_values(ascending=False).nlargest(15).plot.bar(color="b");
plt.title("Number of reviews according to the host names");


# In[ ]:


plt.figure(figsize=(15,6))
df.groupby("host_name")["price"].mean().sort_values(ascending=False).nlargest(15).plot.bar(color="c");
plt.title("Price Mean According to Hosts");


# In[ ]:


plt.figure(figsize=(15,6))
df.groupby("host_name")["price"].median().sort_values(ascending=False).nlargest(15).plot.bar(color="c");
plt.title("Price Median According to Hosts");


# In[ ]:


sns.heatmap(corr,annot=True,cmap="coolwarm");

