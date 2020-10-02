#!/usr/bin/env python
# coding: utf-8

# **We all know NYC is the best city in the world, right???
#    Here, we get started with pulling the data through BigQuery and SQL, and then getting a sense for the structure of the data. [](http://)
# **

# The first thing we have to do is load our packages. Packages are complex compilation of functions created by other users,and open to the community to use. You can technically recreate them yourself in the code, but who has time for that?! Not New Yorkers, I promise you that.
# 
# Below we import some of the required packages for the project. Note: Pandas/Numpy are essential for data science, and I strongly encourage you to check out seperate tutorials for those packages. 

# In[21]:


#import libraries 
import numpy as np #computing
import pandas as pd #calculations/Data wrangling
import bq_helper
from bq_helper import BigQueryHelper
import seaborn as sns #visualization package
import matplotlib.pyplot as plt


# The next step is actually loading our data. We're using a public dataset on BigQuery, and will be pulling using basic SQL. Nothing terribly complex here, we could devote an entire series into SQL! 

# In[4]:


#Get Data
nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="new_york") #Set up connection
bq_assistant = BigQueryHelper("bigquery-public-data", "new_york") #Set up helper
bq_assistant.list_tables() #Show list of available tables 


# In[5]:


#Preview of Dataset 
nyc.head('citibike_trips') #Sneak a peek at data 


# In[7]:


#Query Data, eliminate nulls for total_amount
query = """select *
          from `bigquery-public-data.new_york.citibike_trips`  
          limit 10000
          """


# Awesome. Now that we've pulled our data through our first SQL Query, we're going to proceed to storing it for easy usage. 

# In[8]:



data = nyc.query_to_pandas_safe(query, max_gb_scanned=10) 


# It's time to explore the data! Let's get a sense for what the data looks like, how many rows/columns are present,and other simple explorations. 

# In[9]:


data.head(5) #See head of data 


# In[10]:


data.shape  #(1000 rows,23 columns)


# In[ ]:


data.describe() #See summary of quant data 


# In[ ]:


data.columns #See list of columns 


# In[ ]:


data.info() #Check out structure of each column (if integer, string,etc)


# In[ ]:



pd.isna(data).sum() #See number of NAs by column. We see that birth_year has a sizeable amount of nulls- will need to investigate and handle 


# So we have a good, general sense of the data now. It also helps to visualize the data, to see certain trends/stories that we might have otherwise misseed. Let's do a few quick visuals to get us started here. We'll develop more sophisticated visuals later on in the series! 

# Let's look at what the gender split is. 

# In[14]:


sns.countplot(data['gender'])


# Males dominate the demographic- it's interesting to note how unknown is almost equal to female in the data.

# Now let's see the distribution of the start stations. 

# In[26]:


g = sns.countplot(data['start_station_name']) #Create basic plot 


# Yikes. X-axis is looking very cluttered, can't make any sense of the data. Let's tweak that. 

# 

# In[29]:


ax = sns.countplot(x=data['start_station_name']) #Store labels 

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right") #Rotate x-axis
plt.show() #Show visual


# That's much better! Looks like Barrow & Hudson has the most trip starts. Ew. 
# 
# This is a good stopping point for the first stage of our project. Next time, we'll perform some more in-depth analysis of the data, do some grouping/staistical analysis, perform some data cleansing/wrangling, and then perform a bit more in-depth visuals. 
# 
# In the inteirm, feel free to play around with the code, write some of your own,or explore other tutorials.
