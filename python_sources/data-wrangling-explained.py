#!/usr/bin/env python
# coding: utf-8

# # Data Wrangling
# I worked on this project where i had to wrangle tweet data from different sources to create a clean dataset on which i could perform analysis and provide useful insights. First, let me explain what is Data Wrangling?
# > *Data Wrangling* is a process where first data is gathered from different sources, then the quality of the data is assessed and finally the data is cleaned to create a dataset on which exploratory data analysis could be performed.
# 
# Basically the following three processes are performed in Data Wrangling:-
# 
# - **Gather** : The data is gathered from different sources. The data could be downloaded from a link, scraped from a website, uploaded from txt, csv and more kind of files.
# - **Assess** : After gathering data, it is assesed visually as well as programatically. The data is assesed on the basis of its quality and tidiness. Dirty(poor quality) data and untidy(messy) data are the two unwanted traits of gathered data that should be assessed properly and then cleaned.
# - **Clean** : After assessing the gathered data and noting down all the unwanted traits, the data is cleaned programatically. Cleaning consists of three steps: Define, Code & Test.

# ## Gather
# For this I had to gather three pieces of data, all three from different sources. First, I gathered the WeRateDogs Twitter archive from a csv file which was manually downloaded from a link provided. Second, I gathered the tweet image predictions data from a link programatically using the python library requests. Finally, I used Tweepy python access library for Twitter to fetch the tweet data for each tweet_id in the WeRateDogs. Using the tweepy library, I got JSON data which I wrote to a text file, gathering all the tweets JSON data in text file. Later on fetched tweet ID, retweet count, and favorite count from the text file line by line and then created a data base.

# In[ ]:


get_ipython().system('pip install tweepy')


# In[ ]:


#Libraries Used
import pandas as pd
import requests
import tweepy
import os
import json
import numpy as np
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
response = requests.get(url)
with open(os.path.join(os.getcwd(), url.split('/')[-1]), mode='wb') as file:
    file.write(response.content)


# Since, the input directory is a read-only directory. So,we can write the file into the working directory.

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Get your credentials from [here](https://developer.twitter.com/en.html)    

# In[ ]:


consumer_key = 'U4hUt6MkwsBunrLeP7gBrfs9q'
consumer_secret = '9RBX4KeUOxVhBRab0TBHjDcJp9hcSyvHieyA50as2auN5PxzWJ'
access_token = '1929558530-UuS1sgoWlZtz5xHhJVbWpq0pWoCdR9X7H8Cq89P'
access_secret = '36sqFNfP4b8QIY374adQUgUBrk0Ui5UEB4i3Z5e2qS5Qm'


# In[ ]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, parser=tweepy.parsers.JSONParser(), wait_on_rate_limit=True)


# In[ ]:


df_twitter_archive = pd.read_csv('../input/twitter-archive-enhanced.csv')


# In[ ]:


for tweet_id in df_twitter_archive.tweet_id:
    try:
        tweet_json = api.get_status(tweet_id, tweet_mode = 'extented')
        with open('/kaggle/working/tweet_json.txt', mode='a') as file:
            json.dump(tweet_json, file)
            file.write('\n')
    except Exception as e:
        print(str(tweet_id) + ': ' + str(e))


# In[ ]:


lists = [] #an empyty list to store a dictionaries
with open('/kaggle/working/tweet_json.txt') as file:
    lines = file.read().splitlines()
    for line in lines:
        data = json.loads(line)
        row = {
            'tweet_id'      : data['id'],
            'retweet_count' : data['retweet_count'],
            'favorite_count': data['favorite_count']
        }
        lists.append(row)
df_tweet_data = pd.DataFrame(lists,columns=['tweet_id','retweet_count','favorite_count'])


# In[ ]:


df_img_predictions = pd.read_csv('/kaggle/working/image-predictions.tsv', sep='\t')


# ## Assess
# After gathering all the 3 files, I stored their data into a dataframe for easier assessment and cleaning. In order to assess the data, I examined it visually and programatically using python's pandaslibrary. First, i printed out all the dataframes entirely, used the info() function to assess the datatypes, used describe() function to summarise the quantitative variables in the datasets, etc. Then i examined the dataframes more specifically by examining each variable separately and found out the following issues:-

# In[ ]:


df_twitter_archive


# In[ ]:


df_img_predictions


# In[ ]:


df_tweet_data


# In[ ]:


df_twitter_archive.info()


# In[ ]:


df_img_predictions.info()


# In[ ]:


df_tweet_data.info()


# In[ ]:


all_columns = pd.Series(list(df_twitter_archive) + list(df_img_predictions) + list(df_tweet_data))
all_columns[all_columns.duplicated()]


# In[ ]:


df_twitter_archive.tweet_id.nunique()


# In[ ]:


df_img_predictions.tweet_id.nunique()


# In[ ]:


df_tweet_data.tweet_id.nunique()


# In[ ]:


df_twitter_archive[df_twitter_archive.text.duplicated()]


# In[ ]:


df_twitter_archive.source.value_counts()


# In[ ]:


df_twitter_archive.sample(25)


# In[ ]:


df_twitter_archive.name.isnull().sum()


# In[ ]:


df_twitter_archive.loc[np.random.randint(0,df_twitter_archive.shape[0],40), ['text','name']]


# In[ ]:


df_twitter_archive.name.value_counts()


# In[ ]:


df_twitter_archive.describe()


# In[ ]:


df_twitter_archive.rating_numerator.value_counts()


# In[ ]:


df_twitter_archive.rating_denominator.value_counts()


# In[ ]:


df_img_predictions.describe()


# In[ ]:


df_tweet_data.describe()


# ### Quality
# - Missing and incorrect dog names extracted from text. 'a' is the most popular name which itself is not a name.
# - Timestamp is in string format.
# - Source not extracted properly from hyperlink tag.
# - A lot of null values are not null.
# - Columns: in_reply_to_status_id, in_reply_to_user_id, retweeted_status_user_id, retweeted_status_id and retweeted_status_timestamp, have a lot of null values.
# - Gender of dog could be extracted from text.
# - Hashtags could also be extracted.
# - Absurd rating values.
# - Records without dog breed prediction
# 
# ### Tidiness
# - The dog stage columns in twitter_archive can be arranged into a single column.
# - The image predictions could be condensed to show just the most confident dog breed prediction.
# - All three dataframes can be combined into one single dataframe.

# ## Clean
# Cleaning process consists of three steps: Define, code & Test. First we define how to tackle the issue. Then, we code to resolve the issue and finally we test our code to see if the issues with the data have been resolved. So, in order to clean these 3 dataframes, I carried out the 3 steps for each of the issues and was finally able to achive a clean dataframe. For cleaning purposes, I used pandas's functions: cut, merge, apply, etc. The cleaned dataset was then stored into a csv file.

# In[ ]:


#Create a copy of all the gathered dataframes
df_twitter_archive_copy = df_twitter_archive.copy()
df_img_predictions_copy = df_img_predictions.copy()
df_tweet_data_copy = df_tweet_data.copy()


# ### Missing Data

# #### Missing and incorrect dog names extracted from text

# ##### Define
# Extract the correct names from the text column using regular expression and also get rid of the incorrect names like 'a', 'an', 'the',etc. <b>search()</b> function of the <b>re</b> library can be used to extract the names from the text. Notice that the dog names always start with an uppercase character and then is followed by all lowercase characters.

# ##### Code

# In[ ]:


#In the text the name always starts with a capital letter.
def extract_name_from_text(row):
    try:
        if 'This is' in row['text']:
            name = re.search('This is ([A-Z]\w+)',row['text']).group(1)
        elif 'Meet' in row.text:
            name = re.search('Meet ([A-Z]\w+)', row['text']).group(1)
        elif 'Say hello to' in row.text:
            name = re.search('Say hello to ([A-Z]\w+)', row['text']).group(1)
        elif 'named' in row.text:
            name = re.search('named ([A-Z]\w+)', row['text']).group(1)
        else:
            name = ''
    except AttributeError:
        name = ''
    return name


# In[ ]:


df_twitter_archive_copy['name'] = df_twitter_archive_copy.apply(extract_name_from_text, axis=1)


# ##### Test

# In[ ]:


df_twitter_archive_copy.name.value_counts()


# #### Source not extracted properly from hyperlink tag

# ##### Define
# Extract the proper source of the dog tweet using regular expression and since there are only 4 unique sources convert source to a categorical variable. Create a generic function to extract the sources and then use <b>apply()</b> function of the <b>pandas</b> library to apply the function to the entire column.

# ##### Code

# In[ ]:


def extract_source(row):
    try:
        source = re.search('>(.+)</a>', row['source']).group(1)
    except AttributeError:
        source = ''
    return source


# In[ ]:


df_twitter_archive_copy['source'] = df_twitter_archive_copy.apply(extract_source, axis=1)
df_twitter_archive_copy['source'] = df_twitter_archive_copy.source.astype('category')


# ##### Test

# In[ ]:


df_twitter_archive_copy.source.dtype


# In[ ]:


df_twitter_archive_copy.source.value_counts()


# #### Gender of the dog could be extracted from text

# ##### Define
# On assessing it can be seen that almost all of the texts are indicative of the gender of the dog as 'He'/'She' is used. Extract the gender of the dog tweet using string operations by searching for He/She in the text.

# ##### Code

# In[ ]:


def extract_gender(row):
    if 'He' in row['text']:
        gender = 'M'
    elif 'She' in row['text']:
        gender = 'F'
    else:
        gender = ''
    return gender


# In[ ]:


df_twitter_archive_copy['gender'] = df_twitter_archive_copy.apply(extract_gender, axis=1)
df_twitter_archive_copy['gender'] = df_twitter_archive_copy.gender.astype('category')


# ##### Test

# In[ ]:


df_twitter_archive_copy.gender.dtype


# In[ ]:


df_twitter_archive_copy.gender.value_counts()


# #### Hashtags could also be extracted from text

# ##### Define
# Extract the hashtag from the tweet using regular expressions. Since, # is always exceeded by alphanumeric characters and after the hashtag there is a whitespace or fullstop, a regular expression can be created.

# ##### Code

# In[ ]:


def extract_hashtag(row):
    try:
        if '#' in row['text']:
            hashtag = re.search('#(\w+)[\s\.]', row['text']).group(1)
        else:
            hashtag = float('NaN')
    except AttributeError:
        hashtag = ''
    return hashtag

df_twitter_archive_copy['hashtag'] = df_twitter_archive_copy.apply(extract_hashtag, axis=1)


# ##### Test

# In[ ]:


df_twitter_archive_copy.hashtag.value_counts()


# ### Tidiness

# #### The dog stage columns in `twitter_archive` can be arranged into a single column

# #### Define
# Condense the 4 dog stages(doggo, floofer, puppo, gender) into a single stage column and convert it into a categorical variable. Also, remove the unwanted columns.

# In[ ]:


def get_dog_stage(row):
    if 'doggo' in row['text'].lower():
        stage = 'doggo'
    elif 'floof' in row['text'].lower():
        stage = 'floofer'
    elif 'pupper' in row['text'].lower():
        stage = 'pupper'
    elif 'puppo' in row['text'].lower():
        stage = 'puppo'
    else:
        stage = ''
    return stage


# In[ ]:


df_twitter_archive_copy['stage'] = df_twitter_archive_copy.apply(get_dog_stage, axis=1)
df_twitter_archive_copy['stage'] = df_twitter_archive_copy.stage.astype('category')


# In[ ]:


df_twitter_archive_copy.drop(['doggo','pupper','floofer','puppo'], axis=1, inplace=True)


# #### Test

# In[ ]:


df_twitter_archive_copy.stage.value_counts()


# In[ ]:


df_twitter_archive_copy.stage.dtype


# In[ ]:


list(df_twitter_archive_copy)


# #### The `image predictions` could be condensed to show just the most confident dog breed prediction

# ##### Define
# Instead of showing 3 predictions, show the top dog breed prediction. Only consider that prediction for which the dog prediction is true. Also, remove the unwanted columns.

# ##### Code

# In[ ]:


breed = []
confidence = []

def get_breed_and_confidence(row):
    if row['p1_dog'] == True:
        breed.append(row['p1'])
        confidence.append(row['p1_conf'])
    elif row['p2_dog'] == True:
        breed.append(row['p2'])
        confidence.append(row['p2_conf'])
    elif row['p3_dog'] == True:
        breed.append(row['p3'])
        confidence.append(row['p3_conf'])
    else:
        breed.append('Not identified')
        confidence.append(np.nan)
        
df_img_predictions_copy.apply(get_breed_and_confidence, axis=1)
df_img_predictions_copy['breed'] = pd.Series(breed)
df_img_predictions_copy['confidence'] = pd.Series(confidence)
df_img_predictions_copy.drop(['p1','p1_conf','p1_dog','p2','p2_conf','p2_dog','p3','p3_conf','p3_dog'], axis=1, inplace=True)


# ##### Test

# In[ ]:


df_img_predictions_copy.head()


# In[ ]:


df_img_predictions_copy.info()


# #### All three dataframes can be combined into one single dataframe

# #### Define
# All the columns in the 3 dataframes describe the dog data and can be fit into a single table for further analysis and visualisations. Use the pandas merge function to merge all the three dataframes on tweet_id.

# #### Code

# In[ ]:


df = pd.merge(df_twitter_archive_copy, df_img_predictions_copy, on='tweet_id')
df = df.merge(df_tweet_data_copy, on='tweet_id')


# #### Test

# In[ ]:


list(df)


# In[ ]:


df.head()


# In[ ]:


df.info()


# ### Quality

# #### Records without dog breed prediction

# ##### Define
# Remove the records from dataframe where breed is Not identified. Use pandas query() function to select the records accordingly.

# ##### Code

# In[ ]:


df = df.query('breed != "Not identified"')


# ##### Test

# In[ ]:


df.query('breed == "Not identified"').shape[0]


# In[ ]:


df.info()


# #### Columns: in_reply_to_status_id, in_reply_to_user_id, retweeted_status_user_id, retweeted_status_id and retweeted_status_timestamp, have a lot of null values

# ##### Define
# Remove these unwanted using drop function in pandas.

# ##### Code

# In[ ]:


df.drop(['in_reply_to_status_id', 'in_reply_to_user_id', 'retweeted_status_user_id', 'retweeted_status_id', 'retweeted_status_timestamp'], axis=1, inplace=True)


# ##### Test

# In[ ]:


df.columns


# #### Incorrect Data types

# ##### Define
# tweet_id which is in int64 format should be in string format as we don't need to perform any mathematic operations on tweet_id. The timestamp should be an datetime object instead of string. Use pandas to_string() function to convert tweet_id to string and use to_datetime() function to convert timestamp to datetime object.

# ##### Code

# In[ ]:


df.tweet_id = df.tweet_id.to_string()
df.timestamp = pd.to_datetime(df.timestamp, yearfirst=True)


# ##### Test

# In[ ]:


df.info()


# #### Absurd rating values

# ##### Define

# The ratings seem to be absurd as the numerator is greater than the denominator and also the numerator varies over a long range.Create a new variable called rating which stores the ratio of the numerator and denominator and accordingly divide the dogs into different categories using the ratio value. Also, remove the rating_numerator and rating_denominator colums.

# ##### Code

# In[ ]:


df_twitter_archive_copy.rating_numerator.describe()


# In[ ]:


df['rating'] = df.rating_numerator/df.rating_denominator

#Use ratings to divide into categories
df['rating_category'] = pd.cut(df.rating, bins = [0.0, np.percentile(df.rating,25), np.percentile(df.rating,50), np.percentile(df.rating,75), np.max(df.rating)],labels=['Low','Below_average','Above_average','High'])

#Drop the unwanted columns
df.drop(['rating_numerator','rating_denominator'], axis=1, inplace=True)


# ##### Test

# In[ ]:


df.rating_category.value_counts()


# In[ ]:


df.columns


# #### A lot of null values are not null.

# ##### Define
# 

# A lot of columns still have value as '' or 0.0. These should be coverted to NaN in case of a quantitative variable(rating) and None in case of a qualitative variable(name,  gender, stage, breed, rating_category). Also rating as 0 should be NaN.

# ##### Code

# In[ ]:


df.loc[df['name'] == '', 'name'] = None
df.loc[df['gender'] == '', 'gender'] = None
df.loc[df['stage'] == '', 'stage'] = None
df.loc[df['breed'] == '', 'breed'] = None
df.loc[df['rating'] == 0.0, 'rating'] = np.nan
df.loc[df['rating'] == 0.0, 'rating_category'] = None


# ##### Test

# In[ ]:


df.info()


# ### Store

# In[ ]:


#Store the final cleaned dataframe
df.to_csv('twitter_archive_master.csv', index=False)


# ### Analyse

# #### Gender Analysis
# `Male` dogs are more famous as compared to female dogs. 

# In[ ]:


df.gender.value_counts().plot(kind='bar');


# #### Top Sources
# Out of the 4 sources, `Twitter for iPhone` is clearly the most widely used source to share tweets peratining to dogs.  

# In[ ]:


df.source.value_counts().plot(kind='bar');


# #### Top Names
# `Cooper`, `Lucy`, `Tucker` and `Charlie` are the most common dog names.

# In[ ]:


df.name.value_counts()[0:19].plot(kind='bar');


# #### Top Breeds
# `Golden Retriever` is top breed. 

# In[ ]:


df.breed.value_counts()[0:19].plot(kind='bar');


# #### Average Retweet and Favorite counts for dog breeds
# `Standard_poodle` had the highest average retweet count while `Saluki` had the highest favorite count.

# In[ ]:


#group by breed and store the means of retweet_count and favorite_count.
df_group = df.groupby(['breed'])['retweet_count', 'favorite_count'].mean()
#order by retweet_count and favorite_count.
df_group = df_group.sort_values(['retweet_count', 'favorite_count'], ascending=False)
#plot the top 15 average counts.
df_group.iloc[0:14,].plot(kind='bar');

