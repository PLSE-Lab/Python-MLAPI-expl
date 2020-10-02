#!/usr/bin/env python
# coding: utf-8

# # Visualizing Google Play Store Apps via Seaborn
# 
# Here in this kernel, we will be visualizing the Google Play Store Apps dataset via **seaborn** library. Then we will be able to make some conclusions.
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# Firstly fetch the dataset into `apps` variable.

# In[ ]:


apps = pd.read_csv("../input/googleplaystore.csv")


# And then take a look what do we have in it.

# In[ ]:


apps.info()


# One column (`rating`) is float and the rest of columns consist of string objects, 13 attributes as total. At the first glance, we can see there are some kind of features that are expected to be numeric values (reviews, installs etc.), but they are structured as string type.  To understand the metadata, we have to look at the brief definitions of the features:
# 
# 
# **App:** Application name  
# **Category:** Category the app belongs to  
# **Rating:** Overall user rating of the app (as when scraped)  
# **Reviews:** Number of user reviews for the app (as when scraped)  
# **Size:** Size of the app (as when scraped)  
# **Installs:** Number of user downloads/installs for the app (as when scraped)  
# **Type:** Paid or Free  
# **Price:** Price of the app (as when scraped)  
# **Content Rating:** Age group the app is targeted at - Children / Mature 21+ / Adult  
# **Genres:** An app can belong to multiple genres (apart from its main category). For eg, a musical family game will belong to Music, Game, Family genres.  
# **Last Updated:** Date when the app was last updated on Play Store (as when scraped)  
# **Current Ver:** Current version of the app available on Play Store (as when scraped)  
# **Android Ver:** Min required Android version (as when scraped)  

# Take a look at the first five samples of the dataset, by using `apps.head()` method:

# In[ ]:


apps.head()


# Before stepping into visualizations, we have to check what kind of distribution the `Rating` feature has.

# In[ ]:


apps.Rating.value_counts()


# Ranging from 1.2 to 4.4, most of them are high ratings. But there is an awkward value (`19.0`), which has to be manipulated. See how data looks like:

# In[ ]:


apps[apps.Rating == 19.0]


# Enrty looks messy. Let's try to guess how it should have been:
# 
# 
# **Category:** NaN  
# **Rating:** 1.9  
# **Reviews:** 1,000+  
# **Size:** 3.0M  
# **Installs:** 19  
# **Type:** Free  
# **Price:** 0  
# **Content Rating:** Everyone  
# **Genres:** NaN  
# **Last Updated:** February 11, 2018  
# **Current Ver:** 1.0.19  
# **Android Ver:** 4.0 and up  
# 
# 
# It still has NaN attributes, which means this guy cannot be mended. Remove it!

# In[ ]:


apps = apps.drop(apps.index[10472])
apps.iloc[10471:10475]


# We know there are bunch of entries like this, so we have to get rif of dirty data for the sake of visualization.

# In[ ]:


apps = apps.dropna()


# I wonder if there is a strong relation between the category of application and its rating. But first things first.  
# ### *How many categories are there, and what are them?*

# In[ ]:


apps.Category.nunique()


# In[ ]:


apps.Category.unique()


# We will be using seaborn's boxplot to visualize the correlation. As we bind x-axis as applications grouped by their categories, and y-axis as average ratings of the categories; then we'll be able to see clearly whether there is a strong relation between the category of application and its rating or not.  
# 
# To do so, I created a dataframe which consist of two features (*lists*):  
# `Category` : Name of categories (`category_list`)  
# `Rating` : Average ratings of each categories (`ratings`)

# In[ ]:


category_list = list(apps.Category.unique())
ratings = []

for category in category_list:
    x = apps[apps.Category == category]
    rating_rate = x.Rating.sum()/len(x)
    ratings.append(rating_rate)
data = pd.DataFrame({'Category':category_list, 'Rating':ratings})
new_index = (data['Rating'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

sorted_data


# In[ ]:


plt.figure(figsize=(25,15))
sns.barplot(x=sorted_data.Category, y=sorted_data.Rating)

plt.xticks(rotation = 45)
plt.xlabel('Application Category')
plt.ylabel('Ratings')
plt.title('Average Ratings by Category')
plt.show()


# As it seems, ratings change slightly as category of the application changes.  
# 
# We have another feature named *content rating*, which is the age group the app is targeted at. I wonder how it distributes over the categories, so I will be using seaborn's **horizontal barplot** to visualize it.  
# 
# Let's see what are the `Content Ratings` that we have to deal with.

# In[ ]:


apps["Content Rating"].unique()


# I will be creating lists for each content rating types, and fill them with the count of categories they encompass.

# In[ ]:


# list of categories
cat_list = list(apps.Category.unique())

# content rating lists
everyone = []
teen = []
everyone_10 = []
mature_17 = []
adults_only_18 = []
unrated = []

# the function which fills category's (temp) content rating counts into lists
def insert_counts(everyone, teen, everyone_10, mature_17, adults_only_18, unrated, temp):
    
    # everyone
    try:
        everyone.append(temp.groupby('Content Rating').size()['Everyone'])
    except:
        everyone.append(0)
    
    # teen
    try:
        teen.append(temp.groupby('Content Rating').size()['Teen'])
    except:
        teen.append(0)
    
    # everyone 10+
    try:
        everyone_10.append(temp.groupby('Content Rating').size()['Everyone 10+'])
    except:
        everyone_10.append(0)
        
    # mature 17+
    try:
        mature_17.append(temp.groupby('Content Rating').size()['Mature 17+'])
    except:
        mature_17.append(0)
        
    # adults only 18+
    try:
        adults_only_18.append(temp.groupby('Content Rating').size()['Adults only 18+'])
    except:
        adults_only_18.append(0)
        
    # unrated
    try:
        unrated.append(temp.groupby('Content Rating').size()['Unrated'])
    except:
        unrated.append(0)

# fill lists iteratively via function
for cat in cat_list:
    temp = apps[apps.Category == cat]
    insert_counts(everyone, teen, everyone_10, mature_17, adults_only_18, unrated, temp)
    


# In[ ]:


f,ax = plt.subplots(figsize = (25,25))
sns.barplot(x=everyone,y=cat_list,color='green',alpha = 0.5,label='Everyone')
sns.barplot(x=teen,y=cat_list,color='blue',alpha = 0.7,label='Teen')
sns.barplot(x=everyone_10,y=cat_list,color='pink',alpha = 0.6,label='Everyone 10+')
sns.barplot(x=mature_17,y=cat_list,color='yellow',alpha = 0.6,label='Mature 17+')
sns.barplot(x=adults_only_18,y=cat_list,color='red',alpha = 0.6,label='Adults Only 18+')
sns.barplot(x=unrated,y=cat_list,color='aqua',alpha = 0.6,label='Unrated')

ax.legend(loc='lower right',frameon = True)
ax.set(xlabel='Percentage of Content Ratings', ylabel='Categories',title = "Percentage of Categories According to Content Ratings ")


# One can easily observe that nearly all the applications are targeting 'Everyone', excepting 'Dating' applications.  
# 
# But I want to dive deeper via seaborn's **countplot** and **pieplot**.

# In[ ]:


plt.figure(figsize = (10,8))
sns.countplot(apps['Content Rating'])
plt.show()


# In[ ]:


plt.figure(figsize = (15,15))
plt.pie(everyone, labels = cat_list, autopct = '%.1f%%', rotatelabels = True, startangle = -90.0)
plt.title('Distribution of the Everyone Content across Categories')

plt.show()


# In[ ]:


plt.figure(figsize = (15,15))
plt.pie(mature_17, labels = cat_list, autopct = '%.1f%%', rotatelabels = True)
plt.title('Distribution of the Mature 17+ Content across Categories')
plt.show()


# Now I wonder whether there is a correlation between the minimum supported Android version of the applications and the ratings of them.  
# 
# But we know that our `Android Ver` feature is object-type (string), so we have to cast it to a numerical type; but first let's check if there is any inconvenient type for this kind of conversion.

# In[ ]:


apps2 = apps
apps2['Android Ver'].value_counts()


# 1319 of the entries' Android versions are marked as `Varies with device`. Since most of the applications are supporting `4.1 and up`, I want to assume these guys are `4.1 and up` too.

# In[ ]:


apps2['Android Ver'][apps2['Android Ver'] == 'Varies with device'] = '4.1 and up'

apps2['android_ver_int'] = apps2['Android Ver'].str[0:1].astype(int)

apps2['android_ver_int'].value_counts()


# Now let's keep them in a distinct dataframe, sorted by their `Android Ver`.

# In[ ]:


new_index2 = (apps2['android_ver_int'].sort_values(ascending=False)).index.values
sorted_apps2 = apps2.reindex(new_index2)

sorted_apps2.head(7)


# In[ ]:


f,ax = plt.subplots(figsize = (5,5))
sns.heatmap(sorted_apps2.corr(), annot = True, fmt = '.2f', ax = ax)
plt.show()


# As we can see, there is a weak correlation between *minimum supported android version* and *ratings*.  
# 
# To have a clearer sight on correlation, let's try plotting them on the same graph, with grouping by categories (via seaborn's **pointplot**).

# In[ ]:


new_df = apps2.groupby('Category').mean()
new_df.sort_values('Rating', inplace = True)

new_df.head()


# In[ ]:


new_df['Category'] = new_df.index


# In[ ]:


f,ax2 = plt.subplots(figsize =(20,10))
sns.pointplot(x='Category',y='android_ver_int',data=new_df,color='magenta',alpha=0.8)
sns.pointplot(x='Category',y='Rating',data=new_df,color='aqua',alpha=0.8)
plt.text(x = 18, y = 4.3, s = 'Average Rating', color = 'aqua', fontsize = 17,style = 'italic')
plt.text(x = 18, y = 3.46, s = 'Average Min Supported Android Ver', color='magenta',fontsize = 18,style = 'italic')
plt.xlabel('Categories', fontsize = 15, color = 'black')
plt.ylabel('Ratings', fontsize = 15, color = 'black')
plt.xticks(rotation = 75)
plt.show()


# We can say these two features are irrelevant. Using seaborn's **jointplot** may help us to see where most of the data stay.

# In[ ]:


g = sns.jointplot(new_df['android_ver_int'], new_df['Rating'], kind="kde", height=7, color='aqua')
plt.savefig('graph.png')
plt.show()


# Now I want to test the effect of `Reviews` feature on ratings. Again, we need to cast its type to integer, and add it as a new feature.

# In[ ]:


apps['Reviews_int'] = apps.Reviews.astype(int)


# In[ ]:


f,ax = plt.subplots(figsize = (5,5))
sns.heatmap(apps.corr(), annot = True, fmt = '.2f', ax = ax)
plt.show()


# The correlation of reviews and ratings are as weak as the correlation of android version.  
# 
# There is another feature called `Type`, indicating whether the application is free or paid. Including the content rating, let's examine the effects of them on rating, by using seaborn's **boxplot**.

# In[ ]:


#Content Rating
#Type
#Rating

plt.figure(figsize = (12,7))
sns.boxplot(x='Content Rating', y='Rating', hue='Type', data=apps, palette='PRGn')
plt.show()


# We can come with something like "people are tend to vote higher when they pay for it".
# 
# And also there is no paid applications for adults only (18+) content rating.

# ## Conclusions
# 
# * Ratings change slightly as category of the application changes.
# * Almost all the applications are targeting 'Everyone', excepting 'Dating' applications.
# * There is a weak correlation between *minimum supported android version* and *ratings*.
# * The correlation of reviews and ratings are as weak as the correlation of android version.
# * People are tend to vote higher when they pay for it.
# * There is no paid applications for adults only (18+) content rating.
# 

# In[ ]:




