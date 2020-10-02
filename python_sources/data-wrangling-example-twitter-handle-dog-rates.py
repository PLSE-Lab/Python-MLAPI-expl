#!/usr/bin/env python
# coding: utf-8

# **Title : <font color=red>A Data Wrangling example for Twitter @dog_rates</font> <font>(Beginner's Guide)</font>**
# 
# **Author : <font color=darkblue>Soumya Ghosh</font>**

# **Content :**
# 
# > Data Gathering
# 
# >Data Assessing
# - Visual Assessment
# - Programmatic Assessment
# 
# >Data Cleaning
# - Quality issues
# - Tidiness
# 
# >Insights
# - Statistical Analysis
# - Visualization

# ____

# ####### Loading Required Modules for this Project

# In[251]:


import pandas as pd
import numpy as np
from pandas import Series
from pandas import DataFrame


# In[252]:


##import json
import requests
import os
import sys
import datetime


# # Data wrangling

# ## Gathering Data

# **Loading data from 'twitter-archive-enhanced.csv'**

# In[253]:


twitterArchive_df = pd.read_csv("../input/twitter-archive-enhanced.csv")


# **Loading data from 'image-predictions.tsv'**

# In[254]:


imagePredict_df = pd.read_csv('../input/image-predictions.tsv',sep="\t")


# **creating DataFrame from 'tweet_json.txt' file**

# In[255]:


# creating DataFrame using append method with the help of Series

tweetOtherInfo_df = DataFrame()
with open('../input/tweet_json.txt','r') as frd:
    column_names = frd.readline().strip().split(",")
    for line in frd.readlines():
        tweetOtherInfo_df = tweetOtherInfo_df.append(Series({key:value for key,value in zip(column_names,line.strip().split(","))}),ignore_index=True)


# In[256]:


# changing column order 

tweetOtherInfo_df = tweetOtherInfo_df[['tweet_id','retweet_count','favorite_count']]


# ___

# ## Assessing Data ( <font color=green>Iteration - 1</font> )

# ### <font color=darkred>Visual Assessment</font>

# In[257]:


# Lets visualize 60 sample datapoints from each dataset
twitterArchive_df.head(60)


# In[258]:


imagePredict_df.tail(60)


# In[259]:


tweetOtherInfo_df.sample(60)


# ### <font color=dark red>Programmatic Assessment</font>

# #### `twitter-archive` table
# Table loaded into DataFrame: twitterArchive_df

# In[260]:


twitterArchive_df.count()


# In[261]:


# many variables of the dataframe,such as 'in_reply_to_status_id' , 'in_reply_to_user_id' , 'retweeted_status_id' , 'retweeted_status_user_id'
#,'retweeted_status_timestamp' & 'expanded_urls' filled with NaN


# In[262]:


len(twitterArchive_df)


# In[263]:


twitterArchive_df.shape


# In[264]:


# So there are 2356 datapoints/rows/observation present in the dataset


# In[265]:


twitterArchive_df.tweet_id.is_unique


# In[266]:


twitterArchive_df.tweet_id.value_counts().count()


# In[267]:


# Custom function to display only unique column names of a DataFrame

def print_unique_columns(df):
    for column in list(df.columns):
        if df[column].value_counts().count() == len(df):
            print (column)


# In[268]:


print_unique_columns(twitterArchive_df)


# In[269]:


# 'tweet_id','timestamp' & 'text' columns have 2356 unique values


# In[270]:


twitterArchive_df.index


# In[271]:


twitterArchive_df.columns


# In[272]:


twitterArchive_df.info()


# In[273]:


twitterArchive_df.describe()


# In[274]:


# Exploring data types of several pandas objects


# In[275]:


type(twitterArchive_df.timestamp.iloc[0])


# In[276]:


type(twitterArchive_df.retweeted_status_timestamp[twitterArchive_df.retweeted_status_timestamp.notnull()].iloc[0])


# In[277]:


twitterArchive_df.name.value_counts()


# In[278]:


# 745 rows of 'name' column have None values


# In[279]:


##twitterArchive_df.in_reply_to_user_id[~twitterArchive_df.in_reply_to_user_id.isnull()].astype("int64").describe()


# In[280]:


for text_ in twitterArchive_df.text[:10]:
    print(text_+"\n")


# In[281]:


# Above text is a preview of only first 10 value of 'text' column  


# In[282]:


twitterArchive_df.puppo.value_counts()


# In[283]:


# 'puppo' column has 2326 rows with None values & 30 rows with puppo values


# In[284]:


sum(~twitterArchive_df.puppo.duplicated())
##twitterArchive_df[twitterArchive_df.puppo.duplicated()]


# In[285]:


# Above output implies that the 'puppo' column is consist of only two type of values : None & puppo itself.


# In[286]:


twitterArchive_non_None = twitterArchive_df[~twitterArchive_df.name.isin(['None'])]
twitterArchive_non_None.shape


# In[287]:


# 'name' column has 1611 rows with non null values


# In[288]:


sum(twitterArchive_non_None.name.duplicated())


# In[289]:


# in 'name' column there are 655 duplicated values


# In[290]:


##twitterArchive_non_None[twitterArchive_non_None.name.duplicated()]
##twitterArchive_non_None[twitterArchive_non_None.name.duplicated()].name.value_counts()
twitterArchive_non_None[twitterArchive_non_None.name.duplicated()].name.value_counts().sort_values(ascending=False).head(60)


# In[291]:


# in 'name' column there are some inaccurate values such as a , an ,the,quite etc.


# In[292]:


sum(twitterArchive_df.expanded_urls.duplicated())


# In[293]:


# There are 137 duplicate value presents for 'expanded_urls' column/feature/variable


# In[294]:


##twitterArchive_non_NaN = twitterArchive_df[~twitterArchive_df.expanded_urls.isnull()]
sum(twitterArchive_df.expanded_urls.isnull()) # >>> output : 59

# OR , both works as same

twitterArchive_non_NaN = twitterArchive_df[~twitterArchive_df.expanded_urls.isin([np.nan])]
sum(twitterArchive_df.expanded_urls.isin([np.nan])) # >>> output : 59


# In[295]:


sum(twitterArchive_non_NaN.expanded_urls.str.startswith("http")) # >>> output : 2297


# In[296]:


twitterArchive_df.shape[0] == 2297+59


# In[297]:


#So it is proven all non NaN entries of the column 'expanded_urls' are actually urls


# __________

# #### `image-prediction` table
# Table loaded into DataFrame: imagePredict_df 

# In[298]:


imagePredict_df.info()


# In[299]:


imagePredict_df.index


# In[300]:


imagePredict_df.columns


# In[301]:


imagePredict_df.describe()


# In[302]:


imagePredict_df.shape


# In[303]:


# This DataFrame has 2075 rows/observations


# In[304]:


imagePredict_df.tweet_id.is_unique


# In[305]:


# Finding column with unique values only :-
print_unique_columns(imagePredict_df)


# In[306]:


# 'tweet_id' column has 2075 unique values


# In[307]:


imagePredict_df.img_num.value_counts()


# In[308]:


# 'img_num' column dont have any missing value.


# In[309]:


imagePredict_df.p1.value_counts().sort_values()


# In[310]:


# Image Prediction Algotithm predict highest number of dogs as 'golden_retriever' type. Second highest dog type is 'Labrador_retriever'.


# In[311]:


sum(imagePredict_df.jpg_url.str.startswith("http"))


# In[312]:


imagePredict_df.shape[0] == sum(imagePredict_df.jpg_url.str.startswith("http"))


# In[313]:


#So it is proven all entries of the column 'jpg_url' are actually urls


# In[314]:


#If I need to consider only dog pictures, then either imagePredict_df.p1_dog is True or imagePredict_df.p2_dog is True or imagePredict_df.p3_dog is True

tmp = imagePredict_df[(imagePredict_df.p1_dog == True) | (imagePredict_df.p2_dog == True) | (imagePredict_df.p3_dog == True)]

tmp.shape


# In[315]:


#So tmp DataFrame containing only Dog related tweets


# In[316]:


sum((tmp.p1_conf>tmp.p2_conf) & (tmp.p2_conf>tmp.p3_conf))


# In[317]:


## So p1_conf > p2_conf > p3_conf is the order of a picture confidence level(probabilty) for the Image Predction Algorithm.


# In[318]:


#in tmp2 DataFrame either p1_dog is set to True or p2_dog & p3_dog are set to True where p1_dog is set to False.

tmp2 = tmp[((tmp.p1_dog == True) | ((tmp.p1_dog == False) & (tmp.p2_dog == True) & (tmp.p3_dog == True)))]

sum((tmp2.p1_dog == True))


# In[319]:


sum((tmp2.p1_conf < tmp2.p2_conf + tmp2.p3_conf) & (tmp2.p1_dog == False))
##tmp2.shape # >>> 1633


# In[320]:


#Above 24 implies, if a picture p1_dog set to False ,then there are 24 occasion ,where pictures p2_conf & p3_conf summation is bigger than p1_conf
# In the above case ,despite of p1_conf is set to False,there are 24 times ,when there is a greater probabilty that that its a dog image.


# In[321]:


# Now I want to consider only those rows where either p1_conf is maximum along with other confidefence levels or summation of p2_conf & p3_conf is greater than p1_conf while p1_dog is not True. 
sum(((tmp2.p1_conf < tmp2.p2_conf + tmp2.p3_conf) & (tmp2.p1_dog == False)) | (tmp2.p1_dog == True))


# In[322]:


##sum(((tmp2.p1_conf > tmp2.p2_conf + tmp2.p3_conf) | (tmp2.p1_dog == True)) & (tmp2.p1_dog == False)) # >>> 77


# #### `tweet-other-info` table
# 
# Table loaded into DataFrame: tweetOtherInfo_df 

# In[323]:


tweetOtherInfo_df.info()


# In[324]:


tweetOtherInfo_df.shape


# In[325]:


# This DataFrame consists of 256 rows & 3 columns


# In[326]:


tweetOtherInfo_df.index


# In[327]:


tweetOtherInfo_df.columns


# In[328]:


type(tweetOtherInfo_df.tweet_id.iloc[0])


# In[329]:


type(tweetOtherInfo_df.retweet_count.iloc[0])


# In[330]:


type(tweetOtherInfo_df.favorite_count.iloc[0])


# In[331]:


# So, all columns are String type


# In[332]:


print_unique_columns(tweetOtherInfo_df)


# In[333]:


tweetOtherInfo_df.tweet_id.is_unique


# In[334]:


# 'tweet_id' column has unique values


# In[335]:


tweetOtherInfo_df.retweet_count.value_counts(ascending=False).head()


# In[336]:


tweetOtherInfo_df.favorite_count.value_counts(ascending=False).head()


# In[337]:


# 'retweet_count' & 'favorite_count' column have some values filled with "Not Exist"


# In[338]:


tweetOtherInfo_df.describe()


# ___

# #### Quality isuue
# ##### `twitter-archive` table
# - 'tweet_id' datatype is int64, require str(string)
# - type of 'timestamp' and 'retweeted_status_timestamp' are str not Timestamp
# - As per project description 'doggo' , 'floofer' , 'pupper' & 'puppo' columns have some missing values.
# - need to fix consistency issue for 'doggo','floofer','pupper'& 'puppo' columns and then assign NaN , if one of its corresponding column already filled with non null value.
# 
# ##### `image-prediction` table
# - 'tweet_id' datatype is int64, require str(string)
# - validate an image as a dog image if it meets certain requirements
# - A new column 'type' could be created which will represent dog types. Its value would be consist of 'p1' column value when 'p1_dog' column is True ,otherwise mix.  
# 
# ##### `tweet-other-info` table
# - 'retweet_count' & 'favorite_count' column have some missing values : "Not Exist" . Need to get rid of those rows/observations from the table.
# - 'retweet_count' & 'favorite_count' datatype are str(string) , required int64
# 

# #### Tidiness
# ##### `twitter-archive` table
# - 'doggo' , 'floofer' , 'pupper' & 'puppo' are stage of dogs,so they should be under 1 column
# 
# ##### `twitter-other-info` table
# - 'retweet_count' & 'favorite_count' features should be the part of the `twitter-archive` table

# ## Cleaning Data ( <font color=green>Iteration - 1</font> )

# ###  <font color=red>Quality</font>

# #### `twiter-archive` table

# In[339]:


# making copies of dataframes
twitterArchive_df_clean = twitterArchive_df.copy()


# ##### converting 'tweet_id' type from int64 to str
# DataFrame : `twitterArchive_df_clean` 

# **Define**
# 
# use astype() to conver datatype of a column

# **Code**

# In[340]:


twitterArchive_df_clean.tweet_id = twitterArchive_df_clean.tweet_id.astype("str")


# **Test**

# In[341]:


twitterArchive_df_clean.tweet_id.dtype


# In[342]:


type(twitterArchive_df_clean.tweet_id.iloc[0])


# ##### converting 'timestamp' , 'retweeted_status_timestamp'  data types from str to Timestamp
# DataFrame : `twitterArchive_df_clean` 

# **Define**
# 
# Use to_datetime method of pandas module to convert appropriate string to Timestamp object.
# 
# No matter whether do you use fillna filter or not , all default NaN(Not a Number) values would be converted to NaT(Not a Time).

# **Code**

# In[343]:


twitterArchive_df_clean.timestamp = pd.to_datetime(twitterArchive_df_clean.timestamp)


# In[344]:


##original_df_clean.retweeted_status_timestamp = pd.to_datetime(original_df_clean.retweeted_status_timestamp.fillna(""))
# OR
twitterArchive_df_clean.retweeted_status_timestamp = pd.to_datetime(twitterArchive_df_clean.retweeted_status_timestamp)


# **Test**

# In[345]:


twitterArchive_df_clean.timestamp.dtype


# In[346]:


type(twitterArchive_df_clean.timestamp.iloc[0])


# In[347]:


twitterArchive_df_clean.retweeted_status_timestamp.dtype


# ##### As per project description 'doggo' , 'floofer' , 'pupper' & 'puppo' columns have some missing values and accuracy issues
# DataFrame : DataFrame : `twitterArchive_df_clean` 

# **Define**
# 
# from 'text' column by .str through regular expression extract() , find out dog stages mentioned in the description text of each tweet.

# **Code**

# In[348]:


twitterArchive_df_clean.doggo = None
twitterArchive_df_clean.doggo = twitterArchive_df_clean.text.str.extract('\\b(doggo|Doggo)\\b', expand=True)[0]


# In[349]:


##twitterArchive_df_clean.text.str.extract('(doggo|Doggo)', expand=True)[0].value_counts()
##sum(~twitterArchive_df_clean.doggo.isin([np.nan]))


# In[350]:


twitterArchive_df_clean.puppo = None
twitterArchive_df_clean.puppo = twitterArchive_df_clean.text.str.extract('\\b(puppo|Puppo)\\b', expand=True)[0]


# In[351]:


twitterArchive_df_clean.pupper = None
twitterArchive_df_clean.pupper = twitterArchive_df_clean.text.str.extract('\\b(pupper|Pupper)\\b', expand=True)[0]


# In[352]:


twitterArchive_df_clean.floofer = None
twitterArchive_df_clean.floofer = twitterArchive_df_clean.text.str.extract('\\b(floofer|Floofer)\\b', expand=True)[0]


# **Test**

# In[353]:


twitterArchive_df_clean.doggo.value_counts()


# In[354]:


twitterArchive_df_clean.puppo.value_counts()


# In[355]:


twitterArchive_df_clean.pupper.value_counts()


# In[356]:


twitterArchive_df_clean.floofer.value_counts()


# ##### Need to fix consistency issue for 'doggo','floofer','pupper'& 'puppo' columns and then assign NaN , if one of its corresponding column already filled with non null value.
# DataFrame : DataFrame : `twitterArchive_df_clean`

# **Define**
# 
# use replace() to replace "Doggo" with "doggo",etc.
# and to assign NaN , use np.nan

# **Code**

# In[357]:


twitterArchive_df_clean.doggo.replace("Doggo","doggo",inplace=True)
##t = {'doggo': True, np.nan: False}
##twitterArchive_df_clean.doggo = twitterArchive_df_clean.doggo.map(t)


# In[358]:


twitterArchive_df_clean.floofer[twitterArchive_df_clean.doggo.notnull()] = np.nan
twitterArchive_df_clean.floofer.replace("Floofer","floofer",inplace=True)
##t = {'floofer': True, np.nan: False}
##twitterArchive_df_clean.floofer = twitterArchive_df_clean.floofer.map(t)


# In[359]:


twitterArchive_df_clean.pupper[twitterArchive_df_clean.doggo.notnull() | twitterArchive_df_clean.floofer.notnull()] = np.nan
twitterArchive_df_clean.pupper.replace("Pupper","pupper",inplace=True)
##t = {'pupper': True, np.nan: False}
##twitterArchive_df_clean.pupper = twitterArchive_df_clean.pupper.map(t)


# In[360]:


twitterArchive_df_clean.puppo[twitterArchive_df_clean.doggo.notnull() | twitterArchive_df_clean.floofer.notnull() | twitterArchive_df_clean.pupper.notnull()] = np.nan
twitterArchive_df_clean.puppo.replace("Puppo","puppo",inplace=True)
##t = {'puppo': True, np.nan: False}
##twitterArchive_df_clean.puppo = twitterArchive_df_clean.puppo.map(t)


# **Test**

# In[361]:


##sum(twitterArchive_df_clean.doggo[twitterArchive_df_clean.doggo.notnull()])
twitterArchive_df_clean.doggo.count()


# In[362]:


twitterArchive_df_clean.doggo.value_counts()


# In[363]:


##sum(twitterArchive_df_clean.floofer[twitterArchive_df_clean.floofer.notnull()])
twitterArchive_df_clean.floofer.count()


# In[364]:


twitterArchive_df_clean.floofer.value_counts()


# In[365]:


##sum(twitterArchive_df_clean.pupper[twitterArchive_df_clean.pupper.notnull()])
twitterArchive_df_clean.pupper.count()


# In[366]:


twitterArchive_df_clean.pupper.value_counts()


# In[367]:


##sum(twitterArchive_df_clean.puppo[twitterArchive_df_clean.puppo.notnull()])
twitterArchive_df_clean.puppo.count()


# In[368]:


twitterArchive_df_clean.puppo.value_counts()


# ___

# #### `image-prediction` table

# In[369]:


# making copies of dataframes
imagePredict_df_clean = imagePredict_df.copy()


# ##### converting 'tweet_id' type from int64 to str
# DataFrame : `imagePredict_df_clean`  

# **Define**
# 
# use astype() to conver datatype of a column

# **Code**

# In[370]:


imagePredict_df_clean.tweet_id = imagePredict_df_clean.tweet_id.astype("str")


# **Test**

# In[371]:


imagePredict_df_clean.tweet_id.dtype


# In[372]:


type(imagePredict_df_clean.tweet_id.iloc[0])


# ##### validate an image as a dog image if it meets certain requirements
# DataFrame : `imagePredict_df_clean`  

# **Define**
# 
# filter rows if either 'p1_dog' is True or 'p1_dog' is False ,but p2_conf+p3_conf > p1_conf

# **Code**

# In[373]:


tmp = imagePredict_df_clean[(imagePredict_df_clean.p1_dog == True) | (imagePredict_df_clean.p2_dog == True) | (imagePredict_df_clean.p3_dog == True)]

tmp2 = tmp[((tmp.p1_dog == True) | ((tmp.p1_dog == False) & (tmp.p2_dog == True) & (tmp.p3_dog == True)))]

imagePredict_df_clean = tmp2[((tmp2.p1_conf < tmp2.p2_conf + tmp2.p3_conf) & (tmp2.p1_dog == False)) | (tmp2.p1_dog == True)]


# **Test**

# In[374]:


imagePredict_df_clean


# In[375]:


#out of 2075 rows only 1556 meets the requirement


# In[376]:


imagePredict_df_clean.p1_dog[imagePredict_df_clean.p1_dog==True].count()


# In[377]:


imagePredict_df_clean.p1_dog[imagePredict_df_clean.p1_dog==False].count()


# ##### A new column 'type' could be created which will represent dog types. Its value would be consist of 'p1' column value when 'p1_dog' column is True ,otherwise mix.  

# **Define**

# **Code**

# In[378]:


imagePredict_df_clean["type"] = None


# In[379]:


# if p1_dog is True , we are copying corresponding 'p1' column values to 'type' column

imagePredict_df_clean.type[imagePredict_df_clean.p1_dog == True] = imagePredict_df_clean[imagePredict_df_clean.p1_dog == True].p1
##sum(imagePredict_df_clean.p1_dog == True)


# In[380]:


imagePredict_df_clean.type.fillna("mix",inplace=True)


# **Test**

# In[381]:


sum(imagePredict_df_clean.type.isnull())


# In[382]:


sum(imagePredict_df_clean.type == "mix")


# In[383]:


imagePredict_df_clean.type.value_counts()


# In[384]:


imagePredict_df_clean.head()


# ___

# #### `tweet-other-info` table

# In[385]:


# making copies of dataframes
tweetOtherInfo_df_clean = tweetOtherInfo_df.copy()


# ##### get rid of missing rows/observations from the table for column names : 'retweet_count' & 'favorite_count' 
# 
# DataFrame : `tweetOtherInfo_df_clean`  

# **Define**
# 
# Use isin() for filtering out missing values("Not Exist") from each cloumn 

# **Code**

# In[386]:


tweetOtherInfo_df_clean = tweetOtherInfo_df_clean[~tweetOtherInfo_df_clean.retweet_count.isin(["Not Exist"])]


# In[387]:


tweetOtherInfo_df_clean = tweetOtherInfo_df_clean[~tweetOtherInfo_df_clean.favorite_count.isin(["Not Exist"])]


# **Test**

# In[388]:


sum(tweetOtherInfo_df_clean.retweet_count.isin(["Not Exist"]))


# In[389]:


sum(tweetOtherInfo_df_clean.favorite_count.isin(["Not Exist"]))


# ##### converting 'retweet_count' & 'favorite_count' types from str to int64
# 
# DataFrame : `tweetOtherInfo_df_clean`  

# **Define**
# 
# Use .astype() to change the datatype of the column to int64 from str.

# **Code**

# In[390]:


tweetOtherInfo_df_clean.retweet_count = tweetOtherInfo_df_clean.retweet_count.astype("int64")


# In[391]:


tweetOtherInfo_df_clean.favorite_count = tweetOtherInfo_df_clean.favorite_count.astype("int64")


# **Test**

# In[392]:


tweetOtherInfo_df_clean.retweet_count.dtype


# In[393]:


tweetOtherInfo_df_clean.retweet_count.dtype


# ### <font color=red>Tidiness</font>

# #### 'doggo' , 'floofer' , 'pupper' & 'puppo' are stage of dogs,so they should be under 1 column on `twiter-archive` table

# **Define**
# 
# Create a new column called : stage
# And assign it with any non null value of 'doggo' or 'floofer' or 'pupper' or 'puppo' columns.
# And null values should be filled with NaN.
# After that drop 'doggo' , 'floofer' , 'pupper' & 'puppo' columns

# **Code**

# In[394]:


twitterArchive_df_clean["stage"] = (twitterArchive_df_clean.doggo.fillna("")+twitterArchive_df_clean.floofer.fillna("")+twitterArchive_df_clean.pupper.fillna("")+twitterArchive_df_clean.puppo.fillna(""))


# In[395]:


twitterArchive_df_clean["stage"] = twitterArchive_df_clean["stage"].replace("",np.nan)


# In[396]:


#Dropping columns
twitterArchive_df_clean.drop("doggo",axis=1,inplace=True)
twitterArchive_df_clean.drop("floofer",axis=1,inplace=True)
twitterArchive_df_clean.drop("pupper",axis=1,inplace=True)
twitterArchive_df_clean.drop("puppo",axis=1,inplace=True)


# **Test**

# In[397]:


twitterArchive_df_clean.head(60)


# In[398]:


twitterArchive_df_clean["stage"].value_counts()


# #### column name 'retweet_count' & 'favorite_count'  should be part of the `twiter-archive` table

# **Define**
# 
# Join/merge twitterArchive_df_clean DataFrame(twiter-archive table) with tweetOtherInfo_df_clean DataFrame(tweet-other-info table) on common column name 'tweet_id'.
# 
# Use .merge() function of the DataFrame to join other DataFrame. It would be a inner join.
# 
# After joining done, `tweet-other-info` table would not be of any usage.

# **Code**

# In[399]:


# how = 'inner' , by default
twitterArchive_df_clean = twitterArchive_df_clean.merge(tweetOtherInfo_df_clean,on="tweet_id")


# **Test**

# In[400]:


twitterArchive_df_clean


# In[401]:


# 11 tweet_id(s) are removed,as they are not present on both tables. 


# ## Assesment Data ( <font color=green>Iteration - 2</font> )

# ### <font color=darkred>Visual Assesment</font>

# In[402]:


twitterArchive_df_clean.head(60)


# In[403]:


imagePredict_df_clean.head(30)


# ### <font color=darkred>Programmatic Assasment</font>

# In[404]:


twitterArchive_df_clean.info()


# In[405]:


twitterArchive_df_clean.describe()


# In[406]:


print_unique_columns(twitterArchive_df_clean)


# In[407]:


##sum(tmp_df.expanded_urls.str.split("/").str[5] == tmp_df.tweet_id)


# #### Quality isuue
# ##### `twitter-archive` table
# 
# - As per project description, need to filter out retweets
# - As per project description 'name' column has some inaccurate values
# - As per project description 'rating_numerator','rating_denominator' columns have some inaccurate values
# - Need to change data types of 'rating_numerator' & 'rating_denominator' from str to float64
# - 'rating_numerator','rating_denominator' columns have some consistency issue, such as all denominator are not same constant.

# #### Tidiness
# ##### `image-prediction` table
# - 'jpg_url' & 'type' variable should be the part of `twitter-archive` table only

# ## Cleaning Data ( <font color=green>Iteration - 2</font> )

# ### <font color=red>Quality</font>

# #### `twiter-archive` table

# ##### As per project description ,need to filter out retweets
# DataFrame : `twitterArchive_df_clean`

# **Define**
# 
# Consider column 'retweeted_status_timestamp' of the dataFrame.
# And then use .isnull() to filter out non null values. 
# 

# **Code**

# In[408]:


twitterArchive_df_clean = twitterArchive_df_clean[twitterArchive_df_clean.retweeted_status_timestamp.isnull()]


# **Test**

# In[409]:


sum(twitterArchive_df_clean.retweeted_status_timestamp.notnull())


# ##### As per project description 'name' has some inaccurate values
# DataFrame : `twitterArchive_df_clean`

# **Define**
# 
# from 'text' column by .str through regular expression extract() , find out dog names mentioned in the description text of each tweet.

# **Code**

# In[410]:


# Any sentence start with <This is > or <Meet > followed by a word starts with a Capital letter and then stop with a <.(dot)>

twitterArchive_df_clean["name"] = twitterArchive_df_clean.text.str.extract('((This is|Meet) ([A-Z][a-z]*)\\.)', expand=True)[2] # [a-z] or \w 

##twitterArchive_df_clean.text.str.extract('((This is|Meet) ([A-Z][a-z]*)\\.)', expand=True)[2].count() # 1217

##twitterArchive_df_clean.text.str.extract('((?!This)(?!News)(?!Impressive)([A-Z][a-z]*)\\.)', expand=True).head(60) #Except <This> Start with a captal leter & ends with a <.dot>


# **Test**

# In[411]:


twitterArchive_df_clean["name"].value_counts(ascending=False)
##twitterArchive_df_clean[twitterArchive_df_clean["name"] == "Snoop"]


# ##### As per project description 'rating_numerator','rating_denominator' columns have some inaccurate values
# DataFrame : `twitterArchive_df_clean`

# **Define**
# 
# from 'text' column by .str through regular expression extract() , find out dog ratings mentioned in the description text of each tweet.
# 
# Dogs rating format : x.xx/xx, xx.xx/xx, xx/xx , xxx/xxx

# **Code**

# In[412]:


# in text of each tweet_id(s) there are som erating those are not valid. We ar going to skip those ratings,such as : 7/11 , 9/11 , 4/20 .
## In which 7/11 , 9/11 used as date by @dog_rates. And 4/20 


# In[413]:


''' 
'([0-9]{1,4}/(10|[1-9][0-9]{1,3}))'
'([0-9][0-9]?[.]?\d{0,2}/(10|[1-9][0-9]{1,3}))' better
'((?!7/11)(?!9/11)[0-9][0-9]?[.]?\d{0,2}/(10|[1-9][0-9]{1,3}))' exclude better
'''

twitterArchive_df_clean["rating_numerator"],twitterArchive_df_clean["rating_denominator"] = twitterArchive_df_clean.text.str.extract('((?!7/11)(?!9/11)[0-9][0-9]?[.]?\d{0,2}/(10|[1-9][0-9]{1,3}))', expand=True)[0].str.split("/").str

##twitterArchive_df_clean[twitterArchive_df_clean.text.str.extract('((?!7/11)(?!9/11)[0-9][0-9]?[.]?\d{0,2}/(10|[1-9][0-9]{1,3}))', expand=True)[0].values == "0/10"]

## Side Note:-
#50/50 -> 11/10
#182/10 
#11/15
#1776/10 - > 17.76/10
#666/10 -> 6.66/10
#420/10
#4/20
#0/10


# **Test**

# In[414]:


twitterArchive_df_clean.rating_numerator.count()


# In[415]:


twitterArchive_df_clean[twitterArchive_df_clean.rating_numerator.isnull()]


# In[416]:


##twitterArchive_df_clean[twitterArchive_df_clean.rating_denominator == "20"]


# In[417]:


twitterArchive_df_clean[twitterArchive_df_clean.rating_numerator == "11.27"]


# ##### Need to change data types of 'rating_numerator' & 'rating_denominator' from str to float64
# DataFrame : `twitterArchive_df_clean`

# **Define**
# 
# use astype() for type conversion.
# For this kind of type conversion ,at 1st neeed to flter out null values

# **Code**

# In[418]:


twitterArchive_df_clean.rating_denominator = twitterArchive_df_clean.rating_denominator[twitterArchive_df_clean.rating_denominator.notnull()].astype("float64")
twitterArchive_df_clean.rating_numerator = twitterArchive_df_clean.rating_numerator[twitterArchive_df_clean.rating_numerator.notnull()].astype("float64")


# **Test**

# In[419]:


twitterArchive_df_clean.rating_denominator.dtype


# In[420]:


twitterArchive_df_clean.rating_numerator.dtype


# ##### 'rating_numerator','rating_denominator' column has some consistency issue, such as all denominator are not same constant.
# DataFrame : `twitterArchive_df_clean`

# **Define**
# 
# scale down all 'rating_denominator' to 10

# **Code**

# In[421]:


# filtering denominator those have values other than 10.0

deno_non10 = twitterArchive_df_clean.rating_denominator[twitterArchive_df_clean.rating_denominator != 10.0]


# In[422]:


# change non 10.0 denominator values to 10.0 (except NaN)

twitterArchive_df_clean.rating_denominator[deno_non10.index] = deno_non10 / deno_non10 * 10


# In[423]:


# Scaling down numerator values corresponds of non 10.0 denominator values

twitterArchive_df_clean.rating_numerator[deno_non10.index] = twitterArchive_df_clean.rating_numerator[deno_non10.index] / deno_non10 * 10


# **Test**

# In[424]:


twitterArchive_df_clean.rating_denominator.value_counts()


# In[425]:


twitterArchive_df_clean.rating_denominator[twitterArchive_df_clean.rating_denominator.isnull()]


# ### <font color=red>Tidiness</font>

# #### column name 'jpg_url' & 'type'  should be part of the `twiter-archive` table

# **Define**
# 
# Join/merge twitterArchive_df_clean DataFrame(twiter-archive table) with imagePredict_df_clean DataFrame(image-prediction table) on common column name 'tweet_id'.
# Need to consider only 'tweet_id' ,'jpg_url' & 'type' columns of the imagePredict_df_clean DataFrame for this merging purpose.
# 
# Use pd.merge() to join them. And it would be a inner join.
# 
# After joining drop 'jpg_url' & 'type' columns from imagePredict_df_clean DataFrame.

# **Code**

# In[426]:


# inner join  

twitterArchive_df_clean = pd.merge(twitterArchive_df_clean,imagePredict_df_clean[['tweet_id','jpg_url','type']],on="tweet_id")


# In[427]:


imagePredict_df_clean.drop('jpg_url', axis=1,inplace=True)
imagePredict_df_clean.drop('type', axis=1,inplace=True)


# **Test**

# In[428]:


twitterArchive_df_clean


# In[429]:


# 1499 tweet_id(s) are present on both tables. 


# In[430]:


imagePredict_df_clean


# In[431]:


all_columns = pd.Series(list(twitterArchive_df_clean) + list(imagePredict_df_clean))
all_columns[all_columns.duplicated()]


# In[432]:


#So as like expected after merging done & droping columns , only 'tweet_id' column is left on both tables as a common column 


# ## Assessment Data ( <font color=green>Iteration - 3</font> )

# ### <font color=darkred>Visual Assessment</font>

# In[433]:


twitterArchive_df_clean.head(60)


# ### <font color=darkred>Programmatic Assessment</font>

# In[434]:


twitterArchive_df_clean.info()


# In[435]:


twitterArchive_df_clean.in_reply_to_status_id[twitterArchive_df_clean.in_reply_to_status_id.notnull()].dtype


# In[436]:


twitterArchive_df_clean.in_reply_to_user_id[twitterArchive_df_clean.in_reply_to_user_id.notnull()].dtype


# In[437]:


#So we can see data type of 'in_reply_to_status_id' , 'in_reply_to_user_id' are float64 not string! 


# In[438]:


twitterArchive_df_clean.shape


# In[439]:


twitterArchive_df_clean.describe()


# In[440]:


print_unique_columns(twitterArchive_df_clean)


# In[441]:


twitterArchive_df_clean.expanded_urls.is_unique


# In[442]:


twitterArchive_df_clean.jpg_url.is_unique


# In[443]:


twitterArchive_df_clean.rating_numerator.value_counts()


# In[444]:


twitterArchive_df_clean.rating_denominator.value_counts()


# In[445]:


twitterArchive_df_clean.source.value_counts()


# In[446]:


#Only 3 type of source value is there , so 'source' column data type could be a category.


# #### Quality isuue
# ##### `twitter-archive` table
# - type of 'in_reply_to_status_id' , 'in_reply_to_user_id' are float64 not str(String)
# - 'source' column could be a categorical
# - remove or delete empty/null columns 

# ## Cleaning Data ( <font color=green>Iteration - 3</font> )

# ### <font color=red>Quality</font>

# ##### converting 'in_reply_to_status_id' , 'in_reply_to_user_id'   types from float to str
# DataFrame : `twitterArchive_df_clean` 

# **Define**
# 
# Need to consider only non null values for this type of conversion.
# 
# At 1st convert all values from float to int64 ,this way we would get full numbers. Then Convert them to str

# **Code**

# In[447]:


twitterArchive_df_clean.in_reply_to_status_id = twitterArchive_df_clean.in_reply_to_status_id[twitterArchive_df_clean.in_reply_to_status_id.notnull()].astype("int64").astype("str")


# In[448]:


twitterArchive_df_clean.in_reply_to_user_id = twitterArchive_df_clean.in_reply_to_user_id[twitterArchive_df_clean.in_reply_to_user_id.notnull()].astype("int64").astype("str")


# **Test**

# In[449]:


type(twitterArchive_df_clean.in_reply_to_status_id[twitterArchive_df_clean.in_reply_to_status_id.notnull()].iloc[0])


# In[450]:


type(twitterArchive_df_clean.in_reply_to_user_id[twitterArchive_df_clean.in_reply_to_user_id.notnull()].iloc[0])


# ##### converting 'source'  column type from str object to category
# DataFrame : `twitterArchive_df_clean`  

# **Define**
# 
# 
# Convert 'source' column type to categorical

# **Code**

# In[451]:


twitterArchive_df_clean.source = twitterArchive_df_clean.source.astype('category')


# **Test**

# In[452]:


twitterArchive_df_clean.source.dtype


# In[453]:


twitterArchive_df_clean.dtypes


# ##### Remove retweet related columns from the DataFrame
# DataFrame : `twitterArchive_df_clean`

# **Define**
# 
# delete 'retweeted_status_id','retweeted_status_user_id' & 'retweeted_status_timestamp' using del 

# **Code**

# In[454]:


# Run only once

del twitterArchive_df_clean["retweeted_status_id"]
del twitterArchive_df_clean["retweeted_status_user_id"]
del twitterArchive_df_clean["retweeted_status_timestamp"]


# **Test**

# In[455]:


twitterArchive_df_clean.shape


# In[456]:


# Setting 'tweet_id' column as an index column

##twitterArchive_df_clean.set_index("tweet_id",inplace=True)


# In[457]:


twitterArchive_df_clean.info()


# ___

# **coping data from final cleaned copy of 'twitter-archive' table to twitter_archive_master DataFrame **

# In[458]:


twitter_archive_master = twitterArchive_df_clean


# In[479]:


twitter_archive_master


# ____

# # Data Analysis & Visualization

# ####### Loading additional Modules required for this Project

# In[459]:


import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
import matplotlib


# In[460]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 8,4
sb.set_style('whitegrid')


# **I observer athat above dataset could be considered as following parts:-**
# 
# unique features:
# 
# - tweet_id
# - timestamp
# - text
# - expanded_urls
# - jpg_url
# 
# features those could be used as a category:
# 
# - source
# - type
# - stage
# 
# features with numeric values:
# 
# - rating_numerator
# - rating_denominator
# - retweet_count
# - favorite_count
# 
# others remaining features:
# 
# - name
# - in_reply_to_status_id
# - in_reply_to_user_id
# 
# _And this part would help us to apply proper analizing tool on the given dataset._

# In[461]:


twitter_archive_master.describe()


# 'rating_denominator' is 10.0 , no surprise is there!
# 
# We can see median 'retweet_count' value is 1403 and median 'favorite_count' value is 4399.
# 
# Maximum 'rating_numerator' is 14.0 and minimum 'rating_numerator' is 2.0 ,where median 'rating_numerator' 11.0. Another interesting point to see , that Q1 for 'rating_numerator' is 10.0 , so that proves more than 75% of dogs ratings(by @dog_rates) are more than 10.0 out of 10.0

# In[462]:


twitter_archive_master.corr()


# In[463]:


twitter_archive_master.retweet_count.corr(twitter_archive_master.favorite_count,method="pearson")


# By Default .corr() function using a linear correlation method : "pearson".
# 
# From the above correlation table we can found that strongest valuable correlation exist between 'favorite_count' & 'retweet_count'. Their corr value is 0.913633
# 
# Where correlation between 'rating_numerator' & 'retweet_count' is 0.334656 .And correlation between 'rating_numerator' & 'favorite_count' is 0.440732

# In[464]:


twitter_archive_master.groupby(["source"])["tweet_id","in_reply_to_status_id","in_reply_to_user_id","name","stage"].count()


# From the above pivot table we can say,for maximum cases @dog_rates used iPhone to make tweets in twitter.
# 
# And for all 14 times when @dog_rates had used iPhone as a source device to reply to a user or a status.

# In[465]:


twitter_archive_master.groupby(["stage"])['rating_numerator','retweet_count','favorite_count'].min()


# In[466]:


twitter_archive_master.groupby(["stage"])['rating_numerator','retweet_count','favorite_count'].max()


# From the last 2 tables we can say dog stage : 'doggo' has the lowest 'rating_numerator' of 5.0
# 
# On the other hand, highest 'rating_numerator' for all dog stages are same(14.0) except stage : 'floofer'

# In[467]:


twitter_archive_master.groupby(["stage"])['retweet_count','favorite_count'].agg(['count','min','max','sum','mean','median','std'])


# From the above table we can see, dog stage name 'pupper' has the lowest amount of 'retweet_count' & 'favorite_count' 100 & 673 respectively.
# 
# On the other hand stage : 'doggo' has the highest 'retweet_count' of 77699 retweets. And stage : 'puppo' has the highest 'favorite_count' of 143896 favorites(likes).
# 
# Also we can see that stage : 'pupper' has the lowest mean value for 'retweet_count' & 'favorite_count' respectively 2509.90 and 8003.29 .Another thing to point out count value(147) for pupper is maximum along all other stages. That might be the cause of low mean values.

# In[468]:


twitter_archive_master.groupby(["stage","type"])['retweet_count','favorite_count'].mean()


# In[469]:


twitter_archive_master.groupby(["stage","type"])['retweet_count','favorite_count'].mean().max()


# In[470]:


twitter_archive_master.groupby(["stage","type"])['retweet_count','favorite_count'].mean().idxmax()


# From the last 3 tables , we found that
# 
# Dog stage : doggo & Dog type : Eskimo_dog has the highest mean value for 'retweet_count' : 51130
# 
# Dog stage : puppo & Dog type : Lakeland_terrier has the highest mean value for 'favorite_count' : 143896

# ___

# ### <font color=darkred>Data Visualization</font>

# In[471]:


twitter_archive_master.rating_numerator.value_counts().plot(kind="bar",title = "Bar Chart for 'rating_numerator'")


# From the above bar-chart, we can see most frequent 'rating_numerator' is 12.0 and its occurred more than 375 times.

# In[472]:


color_theme = ['#9902FD', '#FFA07A', '#B0E0E6','#0981FF']
twitter_archive_master.stage.fillna("N.A.").value_counts().plot(kind="pie",colors=color_theme,title="Pie Chart for Different dog stages")


# From the above pie-chart , we can see dog stage : 'pupper' has the 2nd highest share after stage : N.A.(Not Available)

# In[473]:


twitter_archive_master.source.value_counts().plot(kind="barh",title="Bar horizontal chart for 'source'")


# In[474]:


twitter_archive_master.retweet_count.plot(kind="hist",title="Histogram for 'retweet_count'",xlim=(0,40000))


# So from the above Histogram we can say, that maximum number of 'retweet_count' exist between 0 to 750 for this dataset

# In[475]:


twitter_archive_master.favorite_count.plot(kind="hist",title= "Histogram for 'favorite_count'",xticks=[0,15000,30000,60000,100000,120000,140000])


# So from the above Histogram we can say, that maximum number of 'favorite_count' exist between 0 to 15000. Also we can see an outlier.

# In[476]:


twitter_archive_master.plot(kind="scatter",x="retweet_count",y="favorite_count",c=["darkgrey"],s=50,alpha=0.2,title = "'retweet_count' V/S 'favorite_count'")


# Above scatterplot showing relation between 'retweet_count' & 'favorite_count' .And It is a strong relationship as like we have already computed correlation of 0.913633

# In[477]:


twitter_archive_master.boxplot(column="favorite_count",by="stage")


# In[478]:


twitter_archive_master.groupby(["stage"])["favorite_count"].median()


# From last 2 analysis we can say that dog stage : 'doggo' has the highest 'favorite_count' mean value of 11502

# ---

# ####    [[ Reference ]]
# 
# https://stackoverflow.com/questions/4897353/regex-to-disallow-more-than-1-dash-consecutively
# 
# https://stackoverflow.com/questions/2616974/limit-length-of-characters-in-a-regular-expression
# 
# https://stackoverflow.com/questions/26985228/python-regular-expression-match-multiple-words-anywhere
# 
# http://docs.tweepy.org/en/v3.2.0/api.html
# 
# https://developer.twitter.com/en/docs/basics/response-codes
# 
# https://www.statisticssolutions.com/correlation-pearson-kendall-spearman/
# 
# https://en.wikipedia.org/wiki/Covariance
# 
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html
# 
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html
# 
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
