#!/usr/bin/env python
# coding: utf-8

# * On this kernel we will try to understand App Store dataset. Although it looks complicated, under some data science skills we will make it easier. So let's start..

# ![](http://i.hurimg.com/i/hurriyet/75/750x422/5b973f3c0f25431164667ba6.jpg)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.plotly as py
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../input/AppleStore.csv")
data.head()


# **Meaning of columns:**
# 
# * "size_bytes": Size (in Bytes)
# * "rating_count_tot": User Rating counts (for all version)
# * "rating_count_ver": User Rating counts (for current version)
# * "user_rating" : Average User Rating value (for all version)
# * "user_rating_ver": Average User Rating value (for current version)
# * "cont_rating": Content Rating
# * "sup_devices.num": Number of supporting devices
# * "ipadSc_urls.num": Number of screenshots showed for display
# * "lang.num": Number of supported languages
# * "vpp_lic": Vpp Device Based Licensing Enabled

# **DATA CLEANING**

# We will not use all the columns, let's remove some which we don't need

# In[ ]:


data.drop(["Unnamed: 0","id","currency","cont_rating","vpp_lic","ipadSc_urls.num",
           "ver"],axis=1,inplace=True)
data.dropna(inplace=True)


# Let's convert byte to megabyte

# In[ ]:


data.size_bytes=data.size_bytes/(1024*1024)
data.size_bytes=data.size_bytes.astype(int)
data.rename({"size_bytes":"size_mb"},axis=1,inplace=True)
data.head()


# Here we cleaned our dataset, let's check some statistical informatin

# In[ ]:


data.describe()


# **Basic EDA**

# In[ ]:


#Correlation Maps
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,linewidth=0.5,linecolor="red",fmt=".1f",ax=ax)
plt.show()


# In[ ]:


df_free=data.price
df_free=["free" if i==0 else "paid" for i in df_free]
data_pair=data.loc[:,["size_mb","price","rating_count_tot","user_rating","prime_genre"]]
data_pair["free_paid"]=df_free
sns.pairplot(data_pair,palette="dark",hue="free_paid")
plt.show()


# We can see relationship between datas. Also we grouped our apps as "free" and "paid"

# In[ ]:


sns.countplot(data_pair.free_paid)
plt.title("Count of paid & free apps",fontsize=15)


# Amount of free apps are more than paid apps

# Let's check our datas sorted by groups

# In[ ]:


genre_list=data.prime_genre.value_counts()
labels=genre_list.index
values=genre_list.values
trace={"values":values,"labels":labels,"domain":{"x":[0,0.95]},"name":"number of genre",
       "hoverinfo":"label+percent+value","hole":0.3,"type":"pie"}
layout={"autosize":False,"width":800,"height":800,
        "title":"Pie Chart of Number of Genre","annotations":[{"font":{"size":16},"showarrow":False,
                                                              "text":"Number of apps","x":0.11,"y":0.95}]}
data1=[trace]
fig=go.Figure(data=data1,layout=layout)
iplot(fig)


# We can see game sector is dominated app store market place. More than half of them are "Game" applicaions. "Entertainment" and "Education" are following to game with around 7 percent.

# Let's check size of genres

# In[ ]:


genre_size=data.loc[:,["prime_genre","size_mb"]]
genre_size=genre_size.groupby("prime_genre").mean()
new_index=(genre_size["size_mb"].sort_values(ascending=False)).index.values
genre_size=genre_size.reindex(new_index)
genre_size.head()


# We calculated mean size of each genres and sorted

# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x=genre_size.index,y=genre_size.size_mb,palette="dark")
plt.xticks(rotation=75)
plt.xlabel("Genre of apps")
plt.ylabel("Average sizes")
plt.title("Average size of App Genres")
plt.show()


# It is obviously seen that "Medical" and "Games" apps have greater size than others. 
# 
# Keep going with reviews

# In[ ]:


trace=go.Histogram(x=data.user_rating,name="User Ratings",marker=dict(color="rgb(165,70,225)"),text="apps")
data1=[trace]
layout=go.Layout(autosize=False,width=800,height=500,title="Histogram of User Ratings",xaxis=dict(title="Rating"),yaxis=dict(title="Count"))
fig=go.Figure(data=data1,layout=layout)
print("Average rating: ",np.mean(data.user_rating))
iplot(fig)


# Although the users mostly gave 4.5 point, average app rating is 3.5
# 
# Let's go deeper and **check all genres!!**

# In[ ]:


groups = data.groupby('prime_genre').filter(lambda x: len(x) > 178)
groups['user_rating'].hist(by=groups['prime_genre'], sharex=True, figsize=(16,12),color="orange")
plt.show()


# We can have a look closer to user rating of each genre (highest six genre)
# 
# **Keep going with comparing of each average ratings**

# In[ ]:


data1=[]
for i in data.prime_genre.value_counts().index[:12]:
    trace=go.Box(y=data.loc[data.prime_genre==i]["user_rating"],name=i)
    data1.append(trace)
iplot(data1)


# Now we have more statistical information about our data. It is over? No keep going to discover more details..
# 
# Let's try to find out how they are updated. We will try a basic method to undertand it is updated or not. But how? 
# 
# **When_Updated=(Latest_Version_Rating) / (Total_Rating)**-----> If the result is bigger than others, that means that our app is recently updated.

# In[ ]:


updated = data[data["rating_count_tot"]>180000]
updated=updated.loc[:,["track_name","rating_count_tot","rating_count_ver"]]
updated["when_update"]=(updated.rating_count_ver/updated.rating_count_tot)
updated["when_update"]=(updated.when_update-np.min(updated.when_update))/(np.max(updated.when_update)-np.min(updated.when_update))
updated.index = np.arange(1,len(updated)+1)

trace=go.Scatter(x=updated.index,y=updated.when_update,mode="markers",text=updated.track_name,
                marker=dict(color="rgba(0,55,170,0.8)"))
data1=[trace]
layout=dict(title="How Updated Applications",xaxis=dict(title="Application List",zeroline=False,ticklen=5),
           yaxis=dict(title="Updating dates",ticklen=5,zeroline=False))
fig=dict(data=data1,layout=layout)
iplot(fig)


# As we can see that which app is more close to top of the graph, it is more updated than others. With that basic codes we found out which app is updated.
# 
# **Infinity Blade and Real Basketball are most recent updated apps.** Lets remove outliers and check again.

# In[ ]:


drop_=updated[(updated.track_name=="Infinity Blade")|(updated.track_name=="Real Basketball")|
              (updated.track_name=="WhatsApp Messenger")|
              (updated.track_name=="Zillow Real Estate - Homes for Sale & for Rent")].index
updated.drop(drop_,inplace=True)

trace=go.Scatter(x=updated.index,y=updated.when_update,mode="markers",text=updated.track_name,
                marker=dict(color="rgba(255,0,10,0.9)"))
data1=[trace]
layout=dict(title="How Updated Applications",xaxis=dict(title="Application List",zeroline=False,ticklen=5),
           yaxis=dict(title="Updating dates",ticklen=5,zeroline=False))
fig=dict(data=data1,layout=layout)
iplot(fig)


# Now our graph is more clear, the updated apps could be seen
# 
# 

# **Price vs Rating**

# In[ ]:


g=sns.jointplot(data["price"],data["user_rating"],color="brown")
sns.set(context='notebook',style='darkgrid')


# Let's remove outlier prices and check again

# In[ ]:


data_=data[data.price<200]
g=sns.jointplot(data_["price"],data_["user_rating"],color="brown")
sns.set(context='notebook',style='darkgrid')


# We can see that prices of most rated apps are between 0 and 20 USD
# 
# Also most installed apps are free or 1 USD.
# 
# Don't stop! Keep going to check price of genres.

# In[ ]:


groups = data.groupby('prime_genre').filter(lambda x: len(x) > 178)
fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
p = sns.stripplot(x="price", y="prime_genre", data=groups, jitter=True, linewidth=1)


# Oppsss we have outliers again. Let's remove them!

# In[ ]:


groups=groups[groups.price<100]
fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
p = sns.stripplot(x="price", y="prime_genre", data=groups, jitter=True, linewidth=1)


# Yes, now we have a better view. We can see that game applications are more expensive than others. The scale of game apps are between 0-10 USD. But the others are between 0-5 USD.
# 
# **Let's close this kernel with an interesting chart!**

# In[ ]:


import plotly
plotly.tools.set_credentials_file(username='EmreTatbak', api_key='fauFOQaAjN8EoOG8b6Vx')

genre_price=data.loc[:,["prime_genre","price"]]
genre_price=genre_price.groupby("prime_genre").mean()
new_index=(genre_price["price"].sort_values(ascending=False)).index.values
genre_price=genre_price.reindex(new_index)
genre_price.index[1:11]
genre_price.values[1:11]

data1 = [go.Scatterpolar(
  r = genre_price.values[:10],
  theta = genre_price.index[:10],
  fill = 'toself'
)]

layout = go.Layout(autosize=False,
    width=750,
    height=750,
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, np.max(genre_price.values[:10])]
    )
  ),
  showlegend = False
)

fig = go.Figure(data=data1, layout=layout)
py.iplot(fig,)


# 

# On the radar chart we can see distribution of mean prices of first 10 apps

# * Conclusion
# 
# Here we analysed our dataset in order to get some clear results. We used seaborn and pyplot libraries to show our dataset with graphs and some charts. We cleared and prepared dataset for machine learning algorithms. Our dataset is purely clear that we can make some prediction for the future. 
# 
# * If you have any question or advise feel free to ask me. I am waiting for your comments :)
# 
# Special thanks to Lavanya Gupta for giving me inspiration.
# 
# Special thanks to DATAI Team for the tutorials.
