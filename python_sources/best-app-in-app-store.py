#!/usr/bin/env python
# coding: utf-8

# Welcome Every one to the analysis on the App store data. To do a good analysis requires to properly understand the data. So lets jump into this directly and see what he have got for this dataset. The data is about the IOS apps hosted in app store. According to the description page:
# 
# This data set contains more than 7000 Apple iOS mobile application details. The data was extracted from the iTunes Search API at the Apple Inc website. R and linux web scraping tools were used for this study.
# 
# Interesting!!!Seems like we can mine a great amount of knowledge with data spanning across 7000 apps. I will start by doing some Exploratory data analysis that will help us get some intuition about what we have in hand and how and in what direction we can direct our analysis.
# 
# **Problem statement** :- To get more people to download your app, you need to make sure they can easily find your app. Mobile app analytics is a great way to understand the existing strategy to drive growth and retention of future user.
# 
# > Let the analysis begin.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# We will start by  loading the data into a dataframe!

# In[ ]:


appstore_df= pd.read_csv("../input/AppleStore.csv", sep=",", index_col="id", header=0)
app_description_df=pd.read_csv("../input/appleStore_description.csv", sep=",", index_col="id", header=0)


# In[ ]:


print("The size of the appstore data is {0}".format(appstore_df.shape))
print("The dimension of the appstore data is {0}".format(appstore_df.ndim))
print("The size of the app description data is {0}".format(app_description_df.shape))
print("The dimension of the appstore data is {0}".format(app_description_df.ndim))


# Perfect the now of rows in both the dataframe are 7197 so instead of analyzing them indivisually why now merge them into one, that way it would be easy for us to understand. 
# 
# So why to wait , lets do that!

# In[ ]:


combined_app_data= pd.merge(appstore_df,app_description_df,left_index=True, right_index=True)
combined_app_data.shape


# Cool, we now have both the data loaded into one dataframe. Lets have a glimse of our data.

# In[ ]:


combined_app_data.head()


# What the heck is this wierd "Unnamed: 0" column doing. I suspect its a row identifier that has been introduced. Lets be sure about it and delete it.
# 
# Also we see that track name and size byte are repeated twice, so will drop that as well.

# In[ ]:


assert len(combined_app_data["Unnamed: 0"].value_counts())==combined_app_data.shape[0]
combined_app_data=combined_app_data.drop("Unnamed: 0", axis=1)
combined_app_data=combined_app_data.drop("track_name_y", axis=1)
combined_app_data=combined_app_data.drop("size_bytes_y", axis=1)
combined_app_data.shape


# Perfect , lets have a look and feel about the kind of data we have , we will have a look at the columns, and there datatypes.

# In[ ]:


columns_list=combined_app_data.columns.tolist()
print("The columns in our data frame are {0}".format(columns_list))


# In[ ]:


combined_app_data.info()


# In[ ]:


combined_app_data.describe()


# Seems like the data does not conatin any null entry as specified by the info output, but lets cross verify this to be doubly sure. If that's the case, we would not have to worry about any data imputaion which will save us a lot of work and time.

# In[ ]:


combined_app_data.isna().sum().to_frame().T


# Thats confirms, hurray!!
# 
# Next we will tabulate the data types.

# In[ ]:


data_type_df=combined_app_data.dtypes.value_counts().reset_index()
data_type_df.columns=["Data type", "count"]
data_type_df


# So we have 11 numerical variable and 7 categorical variable. Lets seggerate the columns and run our analysis seperately. This will also help us validate if some column is mistakely upcasted to object category.

# In[ ]:


categorical_column_list=combined_app_data.loc[:,combined_app_data.dtypes=="object"].columns.tolist()
print("The categorical columns is out data are {0} \n".format(categorical_column_list))
numerical_column_list= [col for col in columns_list if col not in categorical_column_list]
print("The numerical columns is out data are {0} \n".format(numerical_column_list))


# Lets identify the app which had the maximum rating count.

# In[ ]:


combined_app_data.loc[combined_app_data.rating_count_tot.idxmax()].to_frame().T


# Ohh wow we have the facebook on the top of the charts based on maximum no of rating even though the average user rating is 3.5 . This is probably because facebook is used by a very broad set of users. I am now interested is having a look at top 5 such applications based upon rating count.

# In[ ]:


combined_app_data.sort_values(by="rating_count_tot", ascending=False).track_name_x.head(5).reset_index().T


# I have used almost all of them so i can certainly  agree to this. Lets also look at top 5 apps based upon the average rating.

# In[ ]:


combined_app_data.sort_values(by="user_rating", ascending=False).track_name_x.head(5).reset_index().T


# Which creatures are these, i don't know them. I think a better idea woul be to sort by both total rating count and average rating.

# In[ ]:


combined_app_data.sort_values(by=["rating_count_tot","user_rating"], ascending=False).track_name_x.head(5).reset_index().T


# Yeah i get the same , so the judgement based on total rating count was meaningfull and will be used as a fature referenece as target variable.
# 
# Pricing scheme can be very influencial and people tend to dowload and rate apps that are free more often than paid apps. So lets study that first.
# 

# In[ ]:


combined_app_data["isFree"]= np.where(combined_app_data.price==0.00, "Free app", "Paid app")


# In[ ]:


combined_app_data.isFree.value_counts().plot.bar()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(3.7, 6.27)
sns.barplot(x="isFree", y="rating_count_tot", data=combined_app_data,ax=ax)


# There are quite significant no of paid apps. This should have an impact on the popularity  of an app as seen above, and will use this information further in the analysis.
# 
# Lets see how is the rating distributed across all 7000 apps.

# In[ ]:


combined_app_data.groupby("user_rating")['track_name_x'].count().plot.bar()


# Most of the apps have rating around 4.5 , thats impressive. Those on the left hand corner which has low rating might either be very new with less users or they might not have been actively maintained.
# 
# Lets find out what makes an app to get a high rating count , is it the genre, or is it the Number of supporting devices or it could be ,Number of supported languages.

# In[ ]:


combined_app_data["lang.num"].value_counts().sort_index().to_frame().T


# Ohh seems like the range to languages varies from 1 to 75, this will make it difficult to . I will discretize this for a better understanding.

# In[ ]:


combined_app_data["lang.num_descrete"]=pd.cut(combined_app_data["lang.num"], bins=[0,5,10,20, 50,80],labels=["<5","5-10","10-20","20-50", "50-80"])


# In[ ]:


combined_app_data["lang.num_descrete"].value_counts().plot.bar()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(19.7, 8.27)
xx=sns.pointplot(x="lang.num", y="rating_count_tot", data=combined_app_data, hue="lang.num_descrete",ax=ax)
ax.set(xlabel='No of language support', ylabel='Total rating count')
plt.show()


# So it seems like the apps having support around 27-29 had the highest rating count. Although this can be an influencial factor in the app poplularity , but cannot be the only parameter. Because if this would have been the case the app with the makimum language should have been on the top charts.
# 
# Next we will look at the genre!

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(19.7, 8.27)
combined_app_data.prime_genre.value_counts().plot.bar()


# The maximum apps in teh app store falls under the Game category.

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(29.7, 8.27)
sns.barplot(x="prime_genre", y="rating_count_tot", data=combined_app_data,ax=ax)


# This surely tells that the max ratings are being recorded from the Social netwroking apps.
# 
# Next we will have a look at the no of supporting devices.
# 

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(19.7, 8.27)
combined_app_data["sup_devices.num"].value_counts().plot.bar()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(29.7, 8.27)
sns.barplot(x="sup_devices.num", y="rating_count_tot", data=combined_app_data,ax=ax)


# Apps that have support for 12 devices seems to have the maximum rating total, however there does not seem any pattern with the rest of the numbers.
# 
# Next , we will look at the Number of screenshots showed for display parameter!!

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(29.7, 8.27)
sns.barplot(x="ipadSc_urls.num", y="rating_count_tot", data=combined_app_data,ax=ax)


# Based upon the analysis done above , high ratings have been recorded accross apps that have the following charateristics:
# 
# 1.)No of language support = between 27-29
# 
# 2.)Genre=Social networking
# 
# 3.)No of supported devices=12
# 
# 4.) No of screenshots= 1
# 
# 5.) Free app=True
# 
# These are just the observation that has been made based upon the information presented by the data.
# 

# In[ ]:


plt.scatter(x=combined_app_data.size_bytes_x, y=combined_app_data.rating_count_tot)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




