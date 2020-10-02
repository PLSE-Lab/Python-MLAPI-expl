#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Firstly having a brief information about dataset will help us understand better

# In[ ]:


df = pd.read_csv("../input/airbnb-istanbul-dataset/AirbnbIstanbul.csv")
df.info()


# In[ ]:


df.isnull().sum()


# As you can from output above, the column 'neighbourhood_group', 'last_review','reviews_per_month' columns doesn't have sufficient data. So, dropping them will be helpful;

# In[ ]:


df.drop(columns="neighbourhood_group", inplace = True)
df.drop(columns="last_review", inplace = True)
df.drop(columns="reviews_per_month",inplace=True)


# For having more sense approach column names can be changed;

# In[ ]:


df.rename(columns={
    'number_of_reviews':'Sum_of_reviews',
    'calculated_host_listings_count':'Booking_Count',
    'availability_365':'Availability'},inplace = True)


# For having neater dataset, Null values should've filled with 'NaN'

# In[ ]:


df["name"].fillna('NaN', inplace=True)
df["host_name"].fillna('NaN',inplace=True)


# Hence, from raw dataset we did a simple cleaning process with the help of pandas module

# In[ ]:


df.isnull().sum()


# After basic cleaning, we can dive into our Airbnb dataset for having clear knowledge. For having better understanding about statistics of the dataset,

# In[ ]:


df.describe()


# As you can see from the quantile statistics and max values of the columns, every integer column we have except Availability column has anomalies. For those will do classification, this can cause great losses in accuracy and for prevent that we will filter these datas taking percentile values into account.

# In[ ]:


df = df[df["price"] < 1000]
df = df[df["Sum_of_reviews"]< 10]
df = df[df["Booking_Count"]<5]
df = df[df["minimum_nights"]< 5 ]


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# Finally we can dive into data analyse

# **Distribution of Room types**

# In[ ]:


types = df["room_type"].value_counts().reset_index()

explode=(0.1, 0.0, 0) 

fig, ax = plt.subplots(figsize=(8,4))
ax.pie(types["room_type"], explode=explode, labels=types["index"], autopct='%1.2f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title("Airbnb Istanbul Room Types",fontsize= 20)

plt.show()


# **Top 5 hosts according to Airbnb house counts**

# In[ ]:


category= df["host_name"].value_counts().reset_index().sort_values("host_name", ascending=False).head(5)
sns.barplot(x = 'index',
            y = 'host_name',
            data = category,
            palette="Blues_d"
           
           )



            



# **Top 10 Airbnb Neighbourhoods**

# In[ ]:


neighbourhoods = df["neighbourhood"].value_counts().reset_index()[:10]
b=neighbourhoods["neighbourhood"].tolist()[:10]
a = neighbourhoods["index"].tolist()[:10]

fig, ax = plt.subplots(figsize=(18, 9), subplot_kw=dict(aspect="equal"))


wedges, texts = ax.pie(b, wedgeprops=dict(width=0.5), startangle=-10)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(a[i], xy=(x, y), xytext=(1.1*np.sign(x), 1.1*y),
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("Top 10 Airbnb Neighbourhoods")

plt.show()

