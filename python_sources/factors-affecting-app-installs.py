#!/usr/bin/env python
# coding: utf-8

# **Google Play Store: identify the factors affecting the number of installs of an app**
# 
# Google Play Store has countless apps, about 2.6 million as of December 2018. I am into Mobile Development and this dataset peeked my interest. I needed to identify the factors that lead to higher installs. Isn't that the main goal of any developer? To reach a larger audience? 
# 
# For me, technology is a way to reach people, connect with them, provide them with solutions and make daily life simpler and better. 
# 
# > Marketing without data, is like driving with your eyes closed - Dan Zarella
# 
# This dataset can allow one to understand the data and drive in the right direction, towards the customers. Giving them what they want. 
# 
# **So let's get started!**
# 
# ![Get Started](https://media1.tenor.com/images/7dcc0b5a2c64d741b6edd12a88738cf9/tenor.gif?itemid=4767352)

# In[ ]:


import numpy as np
import pandas as pd 
import os


# **Step 1: Understanding the data **
# 
# Before jumping to conclusions and just randomly making graphs it is crucial to first understand the data. This process involves identifying the columns of the table and the kind of respective entries.

# In[ ]:


data = pd.read_csv("../input/googleplaystore.csv")
data.head()


# In[ ]:


data.describe(include='all')


# The 2 tables above give a brief summary of the dataset. 
# We can see the different columns: category, rating, size, installs, etc. 
# It is also evident that only the rating column is numerical since it is the only one with mean values.  
# The 'top' row in the second table indicates the most common values, which I will dwelve into more later. 
# Overall, the factors that seem important to me in terms of its relation to the number of installs are as follows:
# * Category
# * Rating
# * Type: Free or Paid
# * Content Rating
# * Price

# **Step 2:  Preprocessing - Cleaning the dataset**
# 
# Cleaning the dataset effectively is the most crucial step in understanding the data. 
# Clean data makes analysis more reliable and effective. 
# The first step towards cleaning data is to remove duplicate entries. The summary indicates that the number of unique apps were 9660 while the total entries are 10841. This shows duplication in the dataset which will hinder the reliability of the data. 
# 

# In[ ]:


print('Before removing duplicates',len(data))
data = data.drop_duplicates(subset='App')
print('After removing duplicates',len(data))


# Now that the duplicate entries have been removed, it's time to remove all the empty entries as well. This will give us a clean dataset with ***unique*** and ***complete*** data. 

# In[ ]:


print('Before removing Nan entries',len(data))
data = data.dropna()
print('After removing Nan entries',len(data))


# ![Celebrations](https://media.giphy.com/media/3o6EhYpHUWjYmLN8rK/giphy.gif)
# 
# **Voila!** 
# 
# We have a clean dataset now. 
# What do we do now?
# Let's start with making our dependent variable ready. Why is it not ready you may ask? Well, I'll show you how!

# In[ ]:


print(data.Installs.unique())
print("Data type: ", data.Installs.dtypes)


# Above code displays the unique values in the dataset for the installs column. The things to notice are:
# 1. The data type is **object**. This needs to be changed to int. *Why?* To create charts and graphs, one field needs to be numerical. 
# 2. The factors that make it an object type are:
#      * '+' character at the end of every value
#      *  ',' character in each value, separting every 3 digits
#     
# What do we do now? We remove the extra characters by just replacing them with nothing. Once we do that, we declare the column astype(int) to convert it to integer.  

# In[ ]:


data.Installs = data.Installs.str.replace('+','').str.replace(',','').astype(int)
print(data.Installs.head())


# See? Now we have nicely removed the extra characters and are ready to start understanding other variables and their relation to the number of installs.
# **Are you ready for that?**
# 
# Let's start with understanding the categories.

# In[ ]:


data['Category'].value_counts().plot.bar()
print(data['Category'].value_counts().describe())


# What do we understand from this?
# 
# As indicated in the beginning, the **family** category is the most common category on the app store. 
# 
# This does not indicate that it is most preferred by customers. To identify that, I wanted to check out which category has the most likes and which category has the most average likes. 
# 
# To do that, there were 2 ways. One is well, ineffecient and involves writing more code but it was a step to learning and the second was just grouping!
# 
# So, for the first method:
# 1. Create a DataFrame: Category Name and Total Installs
# 2. Loop through each unique category 
# 3. Filter to get results of just that category
# 4. Calculate the sum of installs for those
# 5. Make a graph. 

# In[ ]:


category_sums = pd.DataFrame(columns=['Category','Total_Installs','Mean_Installs'])
for category in data['Category'].unique():
    sum_install = data.loc[data.Category == category].Installs.sum()
    mean_install = data.loc[data.Category == category].Installs.mean()
    category_sums = category_sums.append({"Category":category, "Total_Installs":sum_install, "Mean_Installs":mean_install}, ignore_index=True)
category_sums.plot.bar(x='Category', y='Total_Installs')


# In[ ]:


category_sums.sort_values(by='Mean_Installs', ascending=False)[:10].plot.bar(x="Category",y="Mean_Installs")


# Well, that was a lot of unnecessary code there. To make it simpler, I tried to just sort the data in groups by Cateogry.  I managed to cut down multiple lines of code to just **one** line.

# In[ ]:


#Another method to find the average install/sum of installs for each category
data.groupby("Category").mean().sort_values(by="Installs",ascending=False)[:10].plot.bar()


# The graph proves 2 things.
# 1. Both the methods work the same and give the same results. 
# 2. It clearly indicates that despite Family being the most common category, the most installs are for **Communication** and **Social**. 

# Moving onto the next factor: **Rating**
# 
# As you would normally expect, higher the ratings, higher the installs. Weirdly the stats show otherwise. It appears that the mean installs are highest of apps with ratings of **4.3** and **4.5**
# 
# To a certain extent this could indicate that a lose corelation between the 2 variables.

# In[ ]:


import seaborn as sns
data.groupby('Rating')['Rating','Installs'].mean().sort_values(by='Installs',ascending=False).head(5).plot.bar()


# Another important factor, is well money. Obviously! Do people pay for the apps? And if they do, then to what extent do they pay? 
# This is possibly an important factor for both: user and developer. It helps to identify how much the developer could earn too. 

# In[ ]:


data.Type.unique()
data.groupby('Type').Installs.mean().plot.bar()
print(data.groupby('Type').Installs.mean().round())


# So, obviously. It is evident that Free apps are more popular than the Paid apps. Despite the ratio being about 1:100. 
# 
# Still, it would be interesting to know how much people tend to pay for the app. Based on the graph, they commonly pay upto **$6.99**. 

# In[ ]:


data.Price.unique()
paid = data.loc[data.Price != '0']
paid.groupby('Price')['Price','Installs'].mean().sort_values(by='Installs',ascending=False).head(5).plot.bar()


# Finally, it's time to understand the Content Rating of the Apps. This indicates the basic user base as well. The graph shows that the most mean installs is for **Teen**, Content Rating. The other top mean installs are for **Everyone 10+** and **Everyone**

# In[ ]:


data['Content Rating'].unique()
data.groupby('Content Rating').Installs.mean().plot.bar()


# **Ta-Da! **
# 
# ![Victory Dance](https://media.giphy.com/media/9wcu6Tr1ecmxa/giphy.gif)
# 
# We have successfully managed to get an *overview* of the factors. 
# 
# Further I want to continue to explore the **correlation** of the factors and possibly **apply machine learning** to predict the installs based on these factors. 
# 
# This has been my first attempt, and I am hoping to better this analysis and improve the charts in the future! 
# 
# 
