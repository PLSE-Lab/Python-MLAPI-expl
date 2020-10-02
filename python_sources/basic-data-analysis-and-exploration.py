#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv("../input/train.csv")

df_songs = pd.read_csv("../input/songs.csv")

df_songs_extra = pd.read_csv("../input/song_extra_info.csv")

df_members = pd.read_csv("../input/members.csv",parse_dates=["registration_init_time","expiration_date"])

df_test = pd.read_csv("../input/test.csv")


# **Common user ids in both training and test sets **

# In[ ]:


print("Number of common users in both the datasets : " ,len(set.intersection(set(df_train['msno']), set(df_test['msno']))))


# In[ ]:


print("Number of Common Songs in both the datasets : ", len(set.intersection(set(df_train['song_id']), set(df_test['song_id']))))

print("No of Unique songs in Training set :", df_train['song_id'].nunique())

print("No of Unique songs in Test set :" ,df_test['song_id'].nunique())


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(df_train['target'])


# In[ ]:


df_songs.head()


# In[ ]:


df_songs_extra.head()


# In[ ]:


df_train =df_train.merge(df_songs,how="left",on="song_id")


# In[ ]:


df_train =df_train.merge(df_songs_extra,how="left",on="song_id")


# In[ ]:


df_train.head()


# In[ ]:


plt.figure(figsize=(12,10))
sns.countplot(df_train['source_system_tab'],hue=df_train['target'])


# In[ ]:


plt.figure(figsize=(12,10))
g = sns.countplot(df_train['source_type'],hue=df_train['target'])
locs, labels = plt.xticks()
g.set_xticklabels(labels,rotation=45)


# Its clear that most of the users prefer playing from their local playlist or local library when they open their app . 
# 
# Its also clear that the my-library and the discover features of the app have the highest count of users from where they play their music .

# In[ ]:


df_train.head()


# **Dropping Null values for song_length and language**

# In[ ]:


df_train.dropna(subset=["song_length"],inplace=True)

df_train.dropna(subset=["language"],inplace=True)


# In[ ]:


df_train['source_system_tab'] = df_train['source_system_tab'].astype("category")
df_train['source_type'] = df_train['source_type'].astype("category")


# In[ ]:


df_train['language'].value_counts()


# In[ ]:


plt.figure(figsize=(12,10))
sns.countplot(df_train['language'],hue=df_train['target'])


# **Language with code 3.0 seems taiwaneese (after a bit of googling )
# 52.0 is of course english 
# 31.0 is Korean .** 
# 
# The above 3 languages make up for most of the data

# In[ ]:


x = df_train['language'].value_counts()


# In[ ]:


df_len = len(df_train)
for lang_id,count in zip(df_train['language'].value_counts().index,df_train['language'].value_counts()) : 
    
    print(lang_id,":",(100*count / df_len))


# **As we can clearly see above , the first 3 languages make up for 90% of the data , languages being Taiwanees ,English and Korean**

# In[ ]:


df_train = df_train.merge(df_members,how="left",on="msno")


# In[ ]:


plt.figure(figsize=(14,12))
df_train['bd'].value_counts(sort=False).plot.bar()

plt.xlim([-10,100])


# **well looks like a  lot of people are yet to be born but have started listening to music ...almost 40% have an age 0 **. 

# In[ ]:


len(df_train.query("bd< 0"))


# **Looks like there are 195 rows of people aged less than 0 . We can delete them since the count is very small .**

# In[ ]:


df_train = df_train.query("bd >= 0")


# In[ ]:


df_train.head()


# In[ ]:


len(df_train.query("bd > 100"))


# **Woaah...6508 people are aged above 100 . Lets see how to deal with them . **

# **lets create a temporary dataframe of genuine ages and do some analysis on them first . **

# In[ ]:


df_train_temp = df_train.query("bd >=5 and bd <80")


# In[ ]:


df_train_temp['bd'].describe()


# In[ ]:


plt.figure(figsize=(15,12))
sns.countplot(df_train_temp['bd'])


# Lets bin the ages into the ranges (5-10,10-18,18-40,40-60,60-80) years

# In[ ]:


df_train_temp['age_range'] = pd.cut(df_train_temp['bd'],bins=[5,10,18,30,45,60,80])


# In[ ]:


plt.figure(figsize=(15,12))
sns.countplot(df_train_temp['age_range'],hue=df_train_temp['target'])


# In[ ]:


df_train_temp['genre_ids'].value_counts().head()


# In[ ]:


plt.figure(figsize=(15,12))
sns.boxplot(df_train_temp['age_range'],df_train_temp["song_length"]/60000,hue=df_train_temp['target'],)
plt.ylabel("Song Length in Minutes")
plt.xlabel("Age Groups")
plt.ylim([0,6])


# In[ ]:


plt.figure(figsize=(14,12))
sns.countplot(df_train_temp['age_range'],hue=df_train_temp["source_type"])
plt.legend(loc="upper right")


# In[ ]:


plt.figure(figsize=(14,12))
sns.countplot(df_train_temp['age_range'],hue=df_train_temp["source_screen_name"])
plt.legend(loc="upper right")


# In[ ]:


plt.figure(figsize=(14,12))
sns.countplot(df_train_temp['age_range'],hue=df_train_temp["source_system_tab"])
plt.legend(loc="upper right")


# In[ ]:


df_train_temp['gender'].value_counts()


# In[ ]:


plt.figure(figsize=(15,12))
df_train_temp.query("gender =='female'")["genre_ids"].value_counts().head(15).plot.bar()
plt.title("Distribution of Genres across Females ")
plt.xlabel("Genre IDs")
plt.ylabel("Count")


# In[ ]:


plt.figure(figsize=(15,12))
df_train_temp.query("gender =='male'")["genre_ids"].value_counts().head(15).plot.bar()
plt.title("Distribution of Genres across Males ")
plt.xlabel("Genre IDs")
plt.ylabel("Count")


# In[ ]:


df_train.drop("composer",axis=1,inplace=True)
df_train_temp.drop("composer",axis=1,inplace=True)


# **Lets check the age outliers and what percent of them exist in our training dataset**

# In[ ]:


100 * len(df_train.query("bd< 0 or bd >80")) / len(df_train)


# They make up for 0.1 % of the training dataset , so it should be safe to delete them . 

# In[ ]:


df_train = df_train.query("bd> 0 and bd <=80")


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_test.drop("composer",axis=1,inplace=True)


# In[ ]:


df_train.info()


# In[ ]:


#df_train[''] = pd.cut(df_train['bd'],bins=[5,10,18,30,45,60,80])
df_train['age_range'] = pd.cut(df_train_temp['bd'],bins=[5,10,18,30,45,60,80])


# In[ ]:


plt.figure(figsize=(14,12))
df_test['bd'].value_counts(sort=False).plot.bar()
plt.xlim([-10,80])


# In[ ]:




