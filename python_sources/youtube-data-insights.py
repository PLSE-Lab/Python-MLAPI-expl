#!/usr/bin/env python
# coding: utf-8

# # EDA on Youtube data
# **The place where cats are celebrieties** Let's check out human psyche through trends in 5 countries. I will focus on US trends
# ![](https://cdn.wccftech.com/wp-content/uploads/2017/08/Screen-Shot-2017-08-30-at-12.37.20-AM.png)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import scale
import statsmodels.api as sm
import statsmodels.formula.api as smf

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-white')


# In[ ]:


cav = pd.read_csv("../input/CAvideos.csv")
dev = pd.read_csv("../input/DEvideos.csv")
frv = pd.read_csv("../input/FRvideos.csv")
gbv = pd.read_csv("../input/GBvideos.csv")
usv = pd.read_csv("../input/USvideos.csv")


# In[ ]:


cav.head()


# In[ ]:


frv.info()


# In[ ]:


usv.head()


# In[ ]:


usv.info()


# In[ ]:


usv.describe()


# 
# sns.boxplot('views','likes', data=usv)
# plt.show()

# ### We will find out the relationship between views with respect to comments and likes.
# Assuming comments and likes have equal effect, I will create a combined summary with respct to views

# In[ ]:


esVLD = smf.ols('views ~ likes + dislikes', usv).fit()
esVLD.summary()


# Summary statistics are useful in every model design in machine learning. Here t-statistics shows departure of standard value from hypothetical value. P-value from given table is equal to zero so we can reject the null hypothesis.
# Positive skewness shows that the distribution is right skewed.
# ![![image.png]](attachment:image.png)

# In[ ]:



sns.jointplot(x='views', y='likes', 
              data=usv, color ='red', kind ='reg', 
              size = 8.0)
plt.show()


# In[ ]:


import json


usv['category_id'] = usv['category_id'].astype(str)
# usv_cat_name['category_id'] = usv['category_id'].astype(str)

category_id = {}

with open('../input/US_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        category_id[category['id']] = category['snippet']['title']

usv.insert(4, 'category', usv['category_id'].map(category_id))
# usv_cat_name.insert(4, 'category', usv_cat_name['category_id'].map(category_id))
category_list = usv['category'].unique()
category_list


# In[ ]:


# labels = usv.groupby(['category_id']).count().index
labels = category_list
trends  = usv.groupby(['category_id']).count()['title']
explode = (0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0 ,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig, ax = plt.subplots(figsize=(10,10))
ax.pie(trends, labels=labels, autopct='%1.1f%%',explode = explode,
        shadow=False, startangle=180)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
trends


# ### What can we deduce ?
# As far as US population is concerned, entertainment and education are most watched categories. Auto and vehicles consist of 10 % of trends. Music and Pets have almost equal distribution (8% approx). Frankly I expected more traffic towards comedy views. But we also have to keep in mind that these cataegories are classified using tags. So it is entirely possible that some comdey videos are classified incorrectly. The sports videos in general have less traffic but this can change when there's NBA or superbowl season.

# ### Checking which catergory has more video comments

# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize=(20,10))

phour=usv.groupby("category").count()["comment_count"].plot.bar()
phour.set_xticklabels(phour.get_xticklabels(),rotation=45)
plt.title("Comment count vs category of Videos")
sns.set_context()


# We can see that entertainment and music receives more comments compared to any other category. This means these categories are important enough to people that they invest their time. It also means people have high sense of opinion towards these aspects.

# 
# plt.figure()
# sns.distplot(usv["comment_count"], hist=False, rug=True);

# mostv_ent = usv[usv['category_id']=='Entertainment'][['title','category_id','views']].sort_values(by='views', ascending=False)
# 
# mostv_ent = mostv_ent.groupby('title')['views'].mean().sort_values(ascending=False).head(3)
# 

# ## Combined Analysis of 5 Countries

# In[ ]:


import glob
files = [file for file in glob.glob('../input/*.{}'.format('csv'))]
sorted(files)
ycd_initial = list()
for csv in files:
    ycd_partial = pd.read_csv(csv)
    ycd_partial['country'] = csv[9:11] #Adding the new column as "country"
    ycd_initial.append(ycd_partial)

ycd = pd.concat(ycd_initial)
ycd.info()


# In[ ]:


ycd.head()


# **Feature Engineering for keeping clean values and removing null data**

# In[ ]:


ycd.apply(lambda x: sum(x.isnull()))


# In[ ]:


column_list=[] 
# Exclude Description column of description because many YouTubers don't include anything in description. This is important to not accidentally delete those values. This for loop will display existing columns in given dataset.
for column in ycd.columns:
    if column not in ["description"]:
        column_list.append(column)
print(column_list)


# In[ ]:


ycd.dropna(subset=column_list, inplace=True) 
# Drop NA values


# In[ ]:


ycd.head()


# In[ ]:


# Feature engineering

#Adjusting Date and Time format in right way
ycd["trending_date"]=pd.to_datetime(ycd["trending_date"],errors='coerce',format="%y.%d.%m")
ycd["publish_time"]=pd.to_datetime(ycd["publish_time"],errors='coerce')
#Create some New columns which will help us to dig more into this data.
ycd["T_Year"]=ycd["trending_date"].apply(lambda time:time.year).astype(int)
ycd["T_Month"]=ycd["trending_date"].apply(lambda time:time.month).astype(int)
ycd["T_Day"]=ycd["trending_date"].apply(lambda time:time.day).astype(int)
ycd["T_Day_in_week"]=ycd["trending_date"].apply(lambda time:time.dayofweek).astype(int)
ycd["P_Year"]=ycd["publish_time"].apply(lambda time:time.year).astype(int)
ycd["P_Month"]=ycd["publish_time"].apply(lambda time:time.month).astype(int)
ycd["P_Day"]=ycd["publish_time"].apply(lambda time:time.day).astype(int)
ycd["P_Day_in_Week"]=ycd["publish_time"].apply(lambda time:time.dayofweek).astype(int)
ycd["P_Hour"]=ycd["publish_time"].apply(lambda time:time.hour).astype(int)


# ### Heatmap showing correlation between numerical variables.
# Data scientists should always keep in mind that correlation alone is not perfect metric. We should also check for causality and independence between given attributes. Generally manipulating dependent variables without knowing it's repercussions can cause massive shift while building up predictive models. Hence as a thumb rule we check summary statistics.

# In[ ]:


plt.figure(figsize = (15,10))
ycd.describe()
sns.heatmap(ycd[["views", "likes","dislikes","comment_count"]].corr(), annot=True)
plt.show()


# ### Category analysis with respect to countries. 
# Here we will figure out which countries watch which categories.
# 

# In[ ]:


category_from_json={}
with open("../input/US_category_id.json","r") as file:
    data=json.load(file)
    for category in data["items"]:
        category_from_json[category["id"]]=category["snippet"]["title"]
        
        
list1=["views likes dislikes comment_count".split()] 
for column in list1:
    ycd[column]=ycd[column].astype(int)
#Similarly Convert The Category_id into String,because later we're going to map it with data extracted from json file    
list2=["category_id"] 
for column in list2:
    ycd[column]=ycd[column].astype(str)


# In[ ]:



from collections import OrderedDict

ycd["Category"]=ycd["category_id"].map(category_from_json)

ycd.groupby(["Category","country"]).count()["video_id"].unstack().plot.barh(figsize=(20,10), stacked=True, cmap = "inferno")
plt.yticks(rotation=0, fontsize=20) 
plt.xticks(rotation=0, fontsize=20) 
plt.title("Category analysis with respect to countries", fontsize=20)
plt.legend(handlelength=5, fontsize  = 10)
plt.show()


# In[ ]:


def trend_plot(country):
    ycd[ycd["country"] == country][["video_id", "trending_date"]].groupby('video_id').count().sort_values    (by="trending_date",ascending=False).plot.kde(figsize=(15,10), cmap = "rainbow")
    plt.yticks(fontsize=18) 
    plt.xticks(fontsize=15) 
    plt.title("\nYouTube trend in "+ country +"\n", fontsize=25)
    plt.legend(handlelength=2, fontsize  = 20)
    plt.show()
#country_list = df.groupby(['country']).count().index
country_list = ["FR", "CA", "GB","US","DE"]
for country in country_list:
    trend_plot(country)


# ### Generating wordcloud

# In[ ]:


from wordcloud import WordCloud
import nltk
#nltk.download("all")
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re


# In[ ]:


def get_cleaned_data(tag_words):
    #Removes punctuation,numbers and returns list of words
    cleaned_data_set=[]
    cleaned_tag_words = re.sub('[^A-Za-z]+', ' ', tag_words)
    word_tokens = word_tokenize(cleaned_tag_words)
    filtered_sentence = [w for w in word_tokens if not w in en_stopwords]
    without_single_chr = [word for word in filtered_sentence if len(word) > 2]
    cleaned_data_set = [word for word in without_single_chr if not word.isdigit()]  
    return cleaned_data_set
MAX_N = 1000
#Collect all the related stopwords.
en_stopwords = nltk.corpus.stopwords.words('english')
de_stopwords = nltk.corpus.stopwords.words('german')
fr_stopwords = nltk.corpus.stopwords.words('french')   
en_stopwords.extend(de_stopwords)
en_stopwords.extend(fr_stopwords)


# In[ ]:


def word_cloud(category):
    tag_words = ycd[ycd['Category']== category]['tags'].str.lower().str.cat(sep=' ')
    temp_cleaned_data_set = get_cleaned_data(tag_words) #get_cleaned_data() defined above.
    
    #Lets plot the word cloud.
    plt.figure(figsize = (20,15))
    cloud = WordCloud(background_color = "white", max_words = 200,  max_font_size = 30)
    cloud.generate(' '.join(temp_cleaned_data_set))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title("\nWord cloud for " + category + "\n", fontsize=40)


# In[ ]:


category_list = ["Music", "Entertainment","News & Politics"]
for category in category_list:
    word_cloud(category)


# **These wordclouds show the important words for these categories. Late Show hosts dominates entertainment searches while words like "official", "Hip hop" even "punjabi songs" are very popular in music industry**
# The big words shows how frequent these searches are. According tom my prediction this data set was published weeks after black panther movie because even though it's not top trending in word cloud but it's still visible there.

# ### Best time to Publish videos in America

# In[ ]:


def best_publish_time(list, title):
    plt.style.use('ggplot')
    plt.figure(figsize=(16,8))
    #list3=df1.groupby("Publish_Hour").count()["Category"].plot.bar()
    list_temp = list.plot.bar()
    #list3.set_xticklabels(list3.get_xticklabels(),rotation=30, fontsize=15)
    list_temp.set_xticklabels(list_temp.get_xticklabels(),rotation=0, fontsize=15)
    plt.title(title, fontsize=25)
    plt.xlabel(s="Best Publishing hour", fontsize=20)
    sns.set_context(font_scale=1)


# In[ ]:


list = ycd[ycd['country'] == 'US'].groupby("P_Hour").count()["Category"]
title = "\nBest Publish Time for USA\n"
best_publish_time(list, title)


# It Looks like from 2 PM to 6 PM seems to be popular time for uploads.

# 

# In[ ]:





# **Please upvote if this Kernel was helpful. Also check out some of my other projects. Have a nice Kaggleing ! : ) ** 
# 
# 

# In[ ]:




