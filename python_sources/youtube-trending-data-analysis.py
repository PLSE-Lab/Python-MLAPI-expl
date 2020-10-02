#!/usr/bin/env python
# coding: utf-8

# **THIS IS DATA ANALYSIS OF YOUTUBE TRENDING SECTION**

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import datetime as dt
from glob import glob


# We have a Json file and a csv for individual country. We will parse dataframes from csvs and JSONs will be used for mapping of "Category" column in dataframes. 
# We would be using "video_id" for index as it is unique for every video.

# In[43]:


#Creating list of filenames
csv_files = glob('../input/*.csv')
json_files = glob('../input/*.json')

#Loading files into variables
df_list = list(map(lambda z: pd.read_csv(z,index_col='video_id'),
                                                             csv_files))
britain_js, germany_js, canada_js, france_js, usa_js = list(map(lambda a: json.load(open(a,'r')), 
                                                                            json_files))


# We will look into head and info of any one dataframe(Britain) and plan our data cleaning process according to that.
# As we look into the head of dataframe, 'description', 'tags' and 'thumbnail_link' dosen't seem to be relevant for our analysis, so we should drop them.

# In[44]:


df_list[4].head()


# In[45]:


df_list[0].info()


# As 'description', 'tags' and 'thumbnail_link' are not necessary for our analysis we can drop them

# In[46]:


def column_dropper(df):
    new_df = df.drop(columns=['description', 'tags', 'thumbnail_link'])
    return new_df

df_list2 = list(map(column_dropper, df_list)) 
df_list[0].head()


# JSON file included with dataset would be used to make "category" column.
# First we have to make a dictionary with key and value pair of "category_id" and "category".
# Then we map this dictionary on the dataframe, droppping 'category_id' at the end because that column is no more useful

# In[47]:


def category_dict_maker(js):
    items = js['items']
    item_id = []
    item_snippet_title = []
    for item in items:
        item_id.append(item['id']) 
        item_snippet_title.append(str(item['snippet']['title']))
    item_dict = dict(zip(item_id, item_snippet_title))
    return(item_dict)

brit_dict = category_dict_maker(britain_js)

def category_maker(value):
    for key in brit_dict:
        if str(value) == key:
            return (brit_dict[key])
        else:
            continue

def cat_applier(df):
    df['category'] = df.category_id.apply(func=category_maker)
    df.category = df.category.astype('category')
    return df.drop(columns=['category_id'])

df_list3 = list(map(cat_applier, df_list2))    
df_list3[0].head()


# We will change dates('trending_date' and 'publish_time') to datetime format.
# Dataset of France has an invalid month number(41) so we would just "coerce" the errors for now

# In[48]:


def string_convertor(string):
    yy=string[0:2]
    dd=string[3:5]
    mm=string[6:8]
    new_string = str("20"+yy+"-"+mm+"-"+dd)
    return new_string

def datetime_setter(df):
    df.trending_date = pd.to_datetime(df.trending_date.apply(string_convertor), errors='coerce')
    df.publish_time = pd.to_datetime(df.publish_time, errors='coerce')
    return df

df_list4 = list(map(datetime_setter, df_list3)) 
df_list4[0].head()


# As cleaning is complete for now, we can unpack the list into tuples

# In[57]:


france, britain, canada, usa, germany = df_list4


# We will calculate the difference between the publish time and trending date and find the one with minimum differrence, to see which video featured in trending section with least time.
# **We can see two outliers (-1 day and 3657 days).**
# -1 could be perfectly resonable here, due to time zones differences.

# In[58]:


britain['trending_delta'] = britain.trending_date - britain.publish_time
min_time = np.min(britain['trending_delta'])
max_time = np.max(britain['trending_delta'])


print("Fastest to trending:") 
print(britain[['title','trending_delta']].loc[britain['trending_delta']==min_time])
print("\nSlowest to trending:") ,
print(britain[['title','trending_delta']].loc[britain['trending_delta']==max_time],'\n')

print("Mean trending delta:", np.mean(britain['trending_delta']))
print("Median trending delta:", np.median(britain['trending_delta']))


# Comparing British to Canadian, we see completely different plots. Seems like Canadians watch more Music videos compared to British.
# From the countplot of category from British dataframe we can conclude that Entertainment videos have great views to like ratio

# In[59]:


sns.lmplot('views', 'likes', data=britain, hue='category', fit_reg=False);
plt.title('British Youtube Trending Section')
plt.xlabel('Views');
plt.ylabel('Likes');
plt.show()


# In[60]:


sns.lmplot('views', 'likes', data=canada, hue='category', fit_reg=False);
plt.title('Canadian Youtube Trending Section')
plt.xlabel('Views');
plt.ylabel('Likes');
plt.show()


# From the both category count plots we can conclude Entertainment videos have higher count for every country.
# Also, this count plot below, confirms our hypothesis that Candians watch more Music videos than British.

# In[52]:


sns.countplot('category', data=britain)
plt.title('Category count plot for Britain')
plt.xlabel('Category')
plt.ylabel('Video Count')
plt.xticks(rotation=90)
plt.show()


# In[53]:


sns.countplot('category', data=canada)
plt.title('Category count plot for Canada')
plt.xlabel('Category')
plt.ylabel('Video Count')
plt.xticks(rotation=90)
plt.show()


# Hey, this is my first ipynotebook, I would like some contstructive criticism on this 
