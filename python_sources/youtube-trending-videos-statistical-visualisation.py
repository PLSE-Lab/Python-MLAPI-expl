#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-size:50px"><b>The data science process</b></h1>
# <section style="background-color:#aacbff;">
# <span style="color:purple; "> 1. Ask questions or define problems</span><br>
# <span style="color:purple;"> 2. Get the data related to the problem</span><br>
# <span style="color:purple;"> 3. Prepare and Describe the data</span><br>
# <span style="color:purple;"> 4. **Explore the data** - notice any **Patterns**? Any **anomalies**?</span><br>
# <span style="color:purple;"> 5. <strike>Model the data</strike></span><br>
# <span style="color:purple;"> 5. <strike>Communicate & Visualise the data</strike></span>
# </section>
# 
# <section style="color:blue; font-size:12px;"><b>This notebook will only focus on the first 4 steps listed above.</b></section>
# 

# **We start by importing the essential python data science modules!**

# In[6]:


# Data manipulation libraries
import numpy as np
import pandas as pd


# In[7]:


# Data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[44]:


# Initial visualisation styles settings
sns.set(style='darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# <h1>1. Initial Questions / Problem statements(s) </h1>

# <span><u>These are some of the questions you might ask:</u></span>
# * What are the similarities between the most watched video categories?
# * When should you post to increase your chances of getting likes?
# * How long does it take for any video to trend on YouTube?
# .......................
# 
# **Note that you can, infer new questions once you start visualising the data!********
# 

# <h1> 2. Get the data </h1>

# In[11]:


us_videos = pd.read_csv("../input/USvideos.csv")
# Verify that data has been indeed imported
us_videos.head(1)


# <h1> 3. Prepare the data </h1>

# <section style="color:blue;"><b><i>
# Next up, we want to replace the categories_id colum with the actual category titles
# </b></i>
# </section>

# In[17]:


# Take a quick pick to viee contents of the json file
pd.read_json('../input/US_category_id.json').head(2)['items'][0]


# Notice that the **items** key contains file dictionary elemts. One of these elemts contain the **title key **which gives the **category titles **that we are looking for.
# 
# We extract this information as follows:

# In[18]:


# IMPORT THE JSON CATEGORIES FILE USING THE BUILT-IN json module
import json

# Initialise an empty Categories list as Cat_list
Cat_list = {}
with open("../input/US_category_id.json","r") as jsonData:
    extracted_data = json.load(jsonData)  # Dictionary holding the entire contents of the json file
    
    for x in extracted_data["items"]:
        # id : title 
        Cat_list [ x["id"] ] = x["snippet"]["title"]


# In[34]:


Cat_list['1']


# <section style="color:blue;"><b><i>
# Next up, we want to map our category id's to their corresponding cartegory titles
# </b></i>
# </section>

# In[37]:


us_videos['category_id'] = us_videos['category_id'].apply(lambda y: str(y))


# In[38]:


us_videos['Category Title'] = us_videos['category_id'].map(Cat_list)

# Check the last column to verify that we added a new column : 'Category Title
us_videos.head(1)


# <h2>3.1. Formatting the data</h2>
# 
# Data fomarting is an essential step to ensure that our data is in a correct format or all the entries in are represent with the correct data type(s)
# 
# ** e.g. We want our dates to be represented as Datetime objects and not as strings**

# In[39]:


us_videos['trending_date'] = pd.to_datetime( us_videos['trending_date'], format="%y.%d.%m" )
us_videos['publish_time'] = pd.to_datetime( us_videos['publish_time'] )


# In[41]:


type( us_videos['publish_time'][0] )


# <h2> 3.2. Check for missing values </h2>
# 
# It is generally assumed that if a column has less than 50% of the total entries it must be droped from the dataset. 

# In[43]:


us_videos.info()


# <h2> 3.3. Add any new features that you think will be useful based on the questions asked above</h2>

# In[54]:


# It might be interesting to add analyse views distribution for different weekdays, months, hours etc.
us_videos['Trending_Year'] = us_videos['trending_date'].apply(lambda x: x.year)
us_videos['Trending_Month'] = us_videos['trending_date'].apply(lambda x: x.month)
us_videos['Trending_Day'] = us_videos['trending_date'].apply(lambda x: x.day)
us_videos['Trending_Day_Of_Week'] = us_videos['trending_date'].apply(lambda x: x.dayofweek)

us_videos["Publish_Year"]=us_videos["publish_time"].apply(lambda y:y.year)
us_videos["Publish_Month"]=us_videos["publish_time"].apply(lambda y:y.month)
us_videos["Publish_Day"]=us_videos["publish_time"].apply(lambda y:y.dayofweek)
us_videos["Publish_Hour"]=us_videos["publish_time"].apply(lambda y:y.hour)


# In[55]:


us_videos.head(1)


# Notice that the weeks days and months are given in terms in numercial form. **Lets change this!!**

# In[57]:


us_videos['Trending_Day_Of_Week'] [0]


# In[58]:


days_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
months_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

us_videos['Trending_Day_Of_Week'] = us_videos['Trending_Day_Of_Week'].map(days_map)
us_videos['Publish_Day'] = us_videos['Publish_Day'].map(days_map)

us_videos['Trending_Month'] = us_videos['Trending_Month'].map(months_map)
us_videos['Publish_Month'] = us_videos['Publish_Month'].map(months_map)


# In[59]:


us_videos.head(2)


# <h1>4. EDA - Exploratory Data Analysis </h1>
# 
# * In this section, we'll use **statistical visualisations** to explore the **distributions of our data**.
# * Any **interesting patterns** can be explored **individually,** in order to draw **actionable insights.**

# <h2>4.1. Category views and likes </h2>
# Where necessary, we'll make use of interactuve plots using the **plotly and cufflinks ** libraries

# In[63]:


# 

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf


# In[64]:


init_notebook_mode(connected=True)
cf.go_offline()


# In[69]:


us_videos.iplot(kind='bar', x='Category Title', y='views', title='Views Per Category',mode='markers',size=10)


# In[75]:


us_videos.iplot(kind='bar', x='Category Title', y=['likes','dislikes'], title='Number of likes Per Category',mode='markers',size=10, colors=['blue','green'],bargap=0.1)


# > > <h2>4.2. Field Correlations</h2>******

# In[77]:


plt.figure(figsize=(20,8))
us_videos_numerical = us_videos[['views','likes','dislikes' ,'comment_count','Category Title']]
sns.pairplot( us_videos_numerical, hue='Category Title')


# In[85]:


sns.heatmap( us_videos_numerical.corr(), cmap='rainbow', annot=True)


# In[ ]:





# <h2>4.3. Publish days vs  number of views / likes</h2>

# In[86]:


plt.figure(figsize=(26,10))
sns.barplot(x='Publish_Day', y='views', data=us_videos, palette='viridis')


# In[87]:


plt.figure(figsize=(26,10))
sns.barplot(x='Publish_Day', y='views', data=us_videos[ us_videos['Publish_Day'] == 'Fri' ], hue='Category Title')


# In[88]:


plt.figure(figsize=(26,10))
sns.barplot(x='Publish_Day', y='views', data=us_videos[ us_videos['Publish_Day'] == 'Sat' ], hue='Category Title')


# <h2>4.3. How many days does it take for a video to trend on average</h2>

# In[89]:


# Create a new column to show the dime delta between publish time and trending time
def day_to_trend(x):
    timeDelta = x['trending_date'] - x['publish_time']
    return timeDelta.seconds/3600


# In[90]:


us_videos['Days_to_trend'] = us_videos.apply( day_to_trend, axis=1 ) 


# In[92]:


sns.distplot(us_videos['Days_to_trend'], bins = 10, color='orange')


# <h2>4.3. Other questions to explore</h2>

# **Number of Videos Uploaded on Particular Time******

# In[94]:


us_videos.head(1)


# In[95]:


sns.distplot(us_videos['Publish_Hour'], bins = 10, color='purple')


# ****Most Watched Videos**

# In[150]:


top_10 = us_videos.sort_values('views',ascending=False)[['views','title']]
top_10 = top_10.head(10)
top_10


# In[151]:


top_10.iplot(kind='bar', x='title', y='views', title='Top 10 videos',mode='markers')
#sns.barplot(x=top_10.title, y=top_10.views)


# # That's it for now....Happy data mining!!

# In[ ]:




