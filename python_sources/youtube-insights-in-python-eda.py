#!/usr/bin/env python
# coding: utf-8

# ## Objective is To Analyze Entire Youtube(USA) Data of 2017-18
# 

# **Notes**
# 
# ** I'm Not Droping Any Duplicate Columns**
# 
# ** I Try to Write Codes as Simple as I Can**
# 
# ** I Will Love to Listen From You for Further Enhancement**

# **Importing Useful Libarries**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import json
get_ipython().run_line_magic('matplotlib', 'inline')


# **Reading The Data**

# In[ ]:


df1=pd.read_csv("../input/USvideos.csv")
df1.head(3)


# In[ ]:


#Here We're Importing and Process the Data From Json File To Get Our Categories  
blank_category={}#This will hold entire data extracted from json file
with open("../input/US_category_id.json","r") as d:#by creating a function I automate the task of data collection
    data=json.load(d)
    for category in data["items"]:
        blank_category[category["id"]]=category["snippet"]["title"]#it Stores the category id with category name


# In[ ]:


blank_category


# ** Now Converting Trending Date and Publish_Time into Standard Pandas Data**

# In[ ]:


#The give format in "trending_date" column is not standardized so I need to mention it
df1["trending_date"]=pd.to_datetime(df1["trending_date"],format="%y.%d.%m")
df1["publish_time"]=pd.to_datetime(df1["publish_time"])


# ** Now Creating New Columns For Each Category**

# In[ ]:


#By Creating New columns for each time category we can get insights in much efficient manner
df1["Trending_Year"]=df1["trending_date"].apply(lambda time:time.year)
df1["Trending_Month"]=df1["trending_date"].apply(lambda time:time.month)
df1["Trending_Day"]=df1["trending_date"].apply(lambda time:time.day)
df1["Trending_Day_of_Week"]=df1["trending_date"].apply(lambda time:time.dayofweek)
df1["publish_Year"]=df1["publish_time"].apply(lambda time:time.year)
df1["publish_Month"]=df1["publish_time"].apply(lambda time:time.month)
df1["publish_Day"]=df1["publish_time"].apply(lambda time:time.day)
df1["publish_Day_of_Week"]=df1["publish_time"].apply(lambda time:time.dayofweek)
df1["Publish_Hour"]=df1["publish_time"].apply(lambda time:time.hour)


# In[ ]:


df1.head(2)#New Data Frame Created But day of week in numeric format we need to convert it


# **Now Creating Map Function**

# In[ ]:


dmap1 = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}#We're Using this Dictionary to Map our column
df1["publish_Day_of_Week"]=df1["publish_Day_of_Week"].map(dmap1)
df1["Trending_Day_of_Week"]=df1["Trending_Day_of_Week"].map(dmap1)
df1.head(2)#Checking Result


# **Now Check The Data Set for Missing Values**

# In[ ]:


df1.info()
#This Data Set Has Negligible Missing Values Which Can Be Neglected For our Analysis


# **Now Converting Data Frame Columns into Suitable Format**

# In[ ]:


list1=["views likes dislikes comment_count".split()]
for column in list1:
    df1[column]=df1[column].astype(int)
# We Need To Convert The Category_id into String,because later we're going to map it with data extracted from json file    
list2=["category_id"] 
for column in list2:
    df1[column]=df1[column].astype(str)
df1["Category"]=df1["category_id"].map(blank_category)#We've Created blank_category{} to store value from JSON file    


# **Now Jump Into Visualization**

# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize=(8,8))
list3=df1.groupby("Publish_Hour").count()["Category"].plot.bar()
list3.set_xticklabels(list3.get_xticklabels(),rotation=30)
plt.title("Publish Time of Videos")
sb.set_context(font_scale=1)


# - This Shows The Pattern How the Videos Uploaded on Particular Time

# In[ ]:


list5=df1[df1["Publish_Hour"]==17].groupby(["Category","publish_Day"]).count()["video_id"].unstack()
plt.figure(figsize=(9,9))#You can Arrange The Size As Per Requirement
sb.heatmap(list5)
plt.title("Category v/s Date Published on 17 hours")


# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize=(8,8))
df1.groupby("Category").count()["views"].plot.bar()
plt.title("Category Wise Uploads")


# - "Entertainment" has the Highest Uploads 

# In[ ]:


plt.style.use('ggplot')
sb.set(rc={"figure.figsize":(20,10)})
df1[df1["Category"]=="Entertainment"].groupby(["views","title"]).count()[4108:]["video_id"].reset_index("views").plot.bar()
plt.title("Top 10 videos in Entertainment Category")


# - As You Can See There're Lots of Duplicate Videos Which Trends Across Various Days
# - You Can Analyze Each Video,According To Trending Days

# In[ ]:


list6=sb.jointplot(x="publish_Day",y="Trending_Day",data=df1,size=8,kind="hex")
plt.title("Filter Out The Trending & Non Trending Videos")


# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize=(8,8))
list7=df1["video_id"].value_counts().plot()
list7.set_xticklabels(list7.get_xticklabels(),rotation=90)
plt.title("This Show The The Occurance of Video in term of Id")


# In[ ]:


plt.style.use('ggplot')
list8=df1.groupby(["publish_Month","publish_Day_of_Week"]).count()["video_id"].unstack()
plt.figure(figsize=(12,10))
sb.heatmap(list8,cmap='viridis')


# **As We Can See Here in December Month's "Friday" highest no.of videos uploaded**
# 
# **Same Process Can Be Done To Check the Date and Hour**

# **Let's Find Top 10 Most Watched Video of 2017**

# In[ ]:


list10=df1[["title","views"]].sort_values(by="views",ascending=True)
list10.drop_duplicates("title",keep="last",inplace=True)
list11=list10.sort_values(by="views",ascending=False)
list12=list11.head(10)
list12.set_index("title",inplace=True)
#I'm not eliminating any data from the Mian Data Frame
#Instead I create a sub set for sake of simplicity
plt.style.use('ggplot')
sb.set(rc={"figure.figsize":(10,10)})
list12.plot.barh()
plt.title("Most Watched Video on YouTube")
#Same Technique Can Be Applied to Find Most Commented,Liked,disliked videos


# **Now Let's Check The Insights from "Youtube Rewind Video"**

# In[ ]:


#We Need To Grab The Location of The Video
list13=df1[df1["title"].str.match("YouTube Rewind")]#.str.match will grab the titles have "YouTube Rewind"


# In[ ]:


sb.factorplot(x="video_id",y="likes",hue="Trending_Day",data=list13,size=8,kind="point")
plt.title("Trending Days v/s Like")


# In[ ]:


sb.factorplot(x="Trending_Day",y="views",hue="publish_Day",data=list13,size=8,kind="point")
plt.title("Trending Days and Views Analysis with Respect To Publish Day")


# **We Can Analyze More Data If Trending Time Has Given**

# In[ ]:


sb.factorplot(x="likes",y="dislikes",hue="Trending_Day",data=list13,size=8,kind="bar",palette="gnuplot2")
plt.title("Dislikes Increasing With Respect To Trading Days")


# In[ ]:


sb.factorplot(x="views",y="comment_count",hue="Trending_Day",data=list13,size=8,kind="bar",palette="CMRmap")
plt.title("Comments With Respect To View")


# In[ ]:


sb.factorplot(x="views",y="Trending_Day_of_Week",hue="Trending_Day",data=list13,size=6,kind="bar",palette="Dark2")
plt.title("Views With Respect To Day of Week")


# In[ ]:


plt.style.use('ggplot')
list14=list13.groupby(["views","Trending_Day_of_Week"]).count()["title"].unstack()
plt.figure(figsize=(8,8))
sb.heatmap(list14,cmap='viridis')


# In[ ]:


sb.jointplot(x="Trending_Day",y="dislikes",data=list13,kind="resid",size=8,color="red")
plt.title("Relation Between Trending Day and Dislike")


# **Analyzing Data From Most Disliked Video**

# In[ ]:


list15=df1[["title","dislikes"]].sort_values("dislikes",ascending=True)
list15.drop_duplicates("title",keep="last",inplace=True)
list16=list15.sort_values("dislikes",ascending=False).head(10)
list16.set_index("title",inplace=True)#Data Preparation
sb.set(rc={"figure.figsize":(10,10)})
plt.style.use('ggplot')
list16.plot.bar()#So Sorry is The Most Disliked Video


# In[ ]:


#Let's Dig Little Deeper
list17=df1[df1["title"].str.match("So Sorry")]
sb.factorplot(x="video_id",y="likes",hue="Trending_Day",data=list17,size=6,kind="point")
plt.title("Trending Days v/s Like")


# In[ ]:


plt.style.use('ggplot')
sb.factorplot(x="Trending_Day",y="views",hue="publish_Day",data=list17,size=6,kind="point")
plt.title("Trending Days and Views Analysis with Respect To Publish Day")#Trending Days v/s Views
sb.factorplot(x="Trending_Day",y="likes",hue="publish_Day",data=list17,size=6,kind="point")
plt.title("Trending Days and likes Analysis with Respect To Publish Day")#Trending Days v/s Likes
sb.factorplot(x="likes",y="dislikes",hue="publish_Day",data=list17,size=6,kind="point")
plt.title("likes and dislikes Analysis with Respect To Publish Day")#likes v/s dislikes


# ** We're Getting No Significant Insights From These 3 plots**

# In[ ]:


plt.style.use('ggplot')
sb.jointplot(x="dislikes",y="likes",data=list17,kind="resid")
sb.jointplot(x="dislikes",y="likes",data=list17,kind="kde")


# - It Seem To Have A Curvilinear Realtion In Between Likes and Dislikes

# In[ ]:


plt.style.use('ggplot')
sb.swarmplot(x="likes",y="dislikes",hue="Trending_Day",data=list17,size=8,palette="inferno")
plt.title("Dislikes Increasing With Respect To Trading Days")


# In[ ]:


plt.style.use('ggplot')
sb.factorplot(x="likes",y="comment_count",hue="Trending_Day",data=list17,size=6,kind="point",palette="inferno")
plt.title("Comments With like")
#Initially This Video Floods with comment but there's drop on 6th day


# In[ ]:


plt.style.use('ggplot')
sb.set(rc={"figure.figsize":(10,10)})
list17[["likes","dislikes"]].plot.bar(stacked=True)
plt.title("We're Comparing Likes and Dislikes With Trending Days ")
#These Plots Visualize That initially likes are more but as the time passes the dislikes increases


# # Final Words: -> Coded With Love :)
