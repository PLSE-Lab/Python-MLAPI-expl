#!/usr/bin/env python
# coding: utf-8

# ## Loading libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import squarify
import seaborn as sns


from plotly.offline import init_notebook_mode, iplot,plot
init_notebook_mode(connected=True)
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.graph_objs as go

from wordcloud import WordCloud,STOPWORDS
from os import path
from PIL import Image
import plotly.plotly as py


# ## Loading datasets

# In[ ]:


app=pd.read_csv("../input/app-store-apple-data-set-10k-apps/AppleStore.csv")
appdis=pd.read_csv("../input/app-store-apple-data-set-10k-apps/appleStore_description.csv")


# ## basic description about data

# In[ ]:




app.describe()


# In[ ]:


app.info()


# ## percentage of missing values
# 

# In[ ]:


pd.DataFrame(app.isnull().sum()/app.isnull().count(),columns=["percent of null"])


# In[ ]:


app.head()


# ## Types of Apps

# In[ ]:



types=app.groupby("prime_genre")["id"].count().sort_values(ascending=False).reset_index()
plt.figure(figsize=(14,7))
sns.barplot(y=types["prime_genre"],x=types["id"],data=types)
plt.gca().set_xlabel("count")
plt.show()


# ## Free vs paid apps

# In[ ]:


free=app[(app["price"]==0)]["id"].count()
paid=app[(app["price"]!=0)]["id"].count()
types=pd.DataFrame({"count":[free,paid],"type":["free","paid"]})
plt.figure()
sns.barplot(x=types["type"],y=types['count'])
plt.gca().set_xlabel("Types")
plt.show()


# In[ ]:


labels=["free","paid"]
values=[free,paid]
trace=go.Pie(labels=labels,values=values)
iplot([trace],filename='pie.html',validate=False)


# In[ ]:





# ## List of most expensive apps

# In[ ]:


app[(app['price']!=0)][["track_name","price"]].sort_values(by="price",ascending=False).reset_index()[:10]


# ## price of apps vs Average user ratings count 

# In[ ]:


price=app.groupby('price')['rating_count_tot'].agg("mean").reset_index()
trace=go.Scatter(x=price['price'],y=price["rating_count_tot"],mode='markers',)
layout = dict(title = ' price of apps vs Average user ratings count ',
              xaxis = dict(title = 'price in USD'),
              yaxis = dict(title = 'Average user ratings count'),
              )
data=[trace]
fig=dict(data=data,layout=layout)
iplot(fig,validate=False)


# ## Correlation matrix

# In[ ]:



plt.figure(figsize=(9,5))
df=app.iloc[:,[3,5,6,7,8,9]]
sns.heatmap(df.corr(),linewidths=.5,cmap="YlGnBu")
plt.show()


# ## Size of apps vs average user ratings count

# In[ ]:


size=app.groupby('size_bytes')['rating_count_tot'].agg("mean").reset_index()
trace=go.Scatter(x=size['size_bytes'],y=size["rating_count_tot"],mode='lines',)
layout = dict(title = 'Size of apps vs average user ratings count',
              xaxis = dict(title = 'size of app in bytes'),
              yaxis = dict(title = 'Average user ratings count'),
              )
data=[trace]
fig=dict(data=data,layout=layout)
iplot(fig,validate=False)


# 

# 
# ## square plot 

# In[ ]:


plt.figure(figsize=(9,7))
prime_genre=app.groupby(['prime_genre'])["id"].agg('count')
squarify.plot(sizes=prime_genre,label=prime_genre.index)
plt.show()


# ## App Store wordcloud

# In[ ]:


stopwords=set(STOPWORDS).union("going","want")
alice_mask = np.array(Image.open("../input/ios-logo/apple-logo.jpg"))
names = appdis["app_desc"]
#print(names)
wordcloud = WordCloud(max_words=150,stopwords=stopwords,max_font_size=70, width=500, height=300,mask=alice_mask,background_color ="white").generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud,interpolation="bilinear")
plt.title("App Store", fontsize=45)
plt.axis("off")
plt.show()
plt.savefig("app_store.png")


# ## App store category wise visualization

# In[ ]:


apps=app.groupby(['prime_genre'])[['price','rating_count_tot','size_bytes','user_rating']].agg("mean")

data= [
    {
        "x":apps['size_bytes']/1000000,
        'y':apps["rating_count_tot"],
        'text':apps.index,
        'mode':'markers',
        'marker':{
            'size':apps['price']*10,
            'color':apps["user_rating"],
            'showscale':True,
            'colorscale':'portland'
            
        }
        
    }
]

layout= go.Layout(
    title= 'App store category wise visualization',
    xaxis= dict(
        title= 'size of app in gb'
    ),
    yaxis=dict(
        title='mean total ratings '
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels',validate=False)
py.image.save_as(fig,'plot.png')


# In[ ]:





# In[ ]:




