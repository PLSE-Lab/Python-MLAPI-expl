#!/usr/bin/env python
# coding: utf-8

# ### **Contents:**
# 1.  <a href='#introduction'>Introduction</a>
# 2.  <a href='#eda'>Exploratory Data analysis</a>
#     1. <a href='#atd'>Data Set Information</a>
#     1. <a href='#dc'>Data Cleaning</a>
#         1. <a href='#dd'>Deduplication</a>
#         1. <a href='#missing'>Missing values</a>                  
#     1. <a href='#hotelname'>Basic stats for the feature: Hotel_Name</a>
#     1. <a href='#avg'>Basic stats for the feature: Average_Score </a>
#     1. <a href='#rn'>Basic stats for the feature: Review_Nationality</a>
#     1. <a href='#rd'>Basic stats for the feature: Review_Date'</a>
#     1. <a href='#tnr'>Basic stats for the feature: Total_Number_of_Reviews_Reviewer_Has_Given</a>
#     1. <a href='#latlng'>plotting Hotel's location on the map using: lat, lng </a>
#     1. <a href='#rtp'>Basic stats for the feature: Review_Total_Positive_Word_Counts</a>    
#     1. <a href='#rtn'>Basic stats for the feature: Review_Total_Negative_Word_Counts</a>
# 1. <a href='#posneg'>Counting no of positive and negative reviews</a>
# 1.  <a href='#popular'>Finding the top 20 famous Hotels</a>
# 1. <a href='#phr'>Finding the top 20 positive rated Hotels</a>
# 1. <a href='#pre'>Preprocessing Reviews for Sentiment Analysis</a>    

# <a id='introduction'></a>
# # 1. Introduction
# This is the exploratory data analysis for 515k Hotel reviews in Europe. By looking at the reviews of the Hotels we can get the information about the top rated hotels in Europe and can predict which hotel is best to stay while visiting Europe.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from collections import Counter
import re, nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import folium
from matplotlib.colors import LinearSegmentedColormap
import missingno as msno
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# <a id='eda'></a>
# # 2  Exploratory Data Analysis
# <a id='atd'></a>
# ## 2.1 **About the Dataset**
# 
# This dataset contains **515,738  customer reviews** and scoring of **1493  Luxury Hotels** across Europe.
# The csv file contains 17 fields. The description of each field is as below:
# 
# * **Hotel_Address**: Address of hotel.
# * **Review_Date**: Date when reviewer posted the corresponding review.
# * **Average_Score**: Average Score of the hotel, calculated based on the latest comment in the last year.
# * **Hotel_Name**: Name of Hotel
# * **Reviewer_Nationality**: Nationality of Reviewer
# * **Negative_Review**: Negative Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Negative'
# * **Review_Total_Negative_Word_Counts**: Total number of words in the negative review.
# * **Positive_Review**: Positive Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Positive'
# * **Review_Total_Positive_Word_Counts**: Total number of words in the positive review.
# * **Reviewer_Score**: Score the reviewer has given to the hotel, based on his/her experience
# * **Total_Number_of_Reviews_Reviewer_Has_Given**: Number of Reviews the reviewers has given in the past.
# * **Total_Number_of_Reviews**: Total number of valid reviews the hotel has.
# * **Tags**: Tags reviewer gave the hotel.
# * **days_since_review**: Duration between the review date and scrape date.
# * **Additional_Number_of_Scoring**: There are also some guests who just made a scoring on the service rather than a review. This number indicates how many valid scores without review in there.
# * **lat**: Latitude of the hotel
# * **lng**: longtitude of the hotel
# 
# 
# Meanwhile, the geographical location of hotels are also provided for further analysis.

# In[ ]:


#loading the dataset from the Hotel_reviews dataset
df = pd.read_csv("../input/Hotel_Reviews.csv")


# In[ ]:


#printing the columns names of the datset
df.columns


# In[ ]:


#printing the shape of the dataset
df.shape


# In[ ]:


print ('Number of data points : ', df.shape[0],        '\nNumber of features:', df.shape[1])
df.head()


# <a id='dc'></a>
# ## 2.2** Data Cleaning** 
# <a id='dd'></a>
# ###  2.1.1 Deduplication

# In[ ]:


#Removing duplicates from the dataset
print(sum(df.duplicated()))
df = df.drop_duplicates()
print('After removing Duplicates: {}'.format(df.shape))


# * We observe that **526** reviews are **duplicate** and we removed them.

# <a id='missing'></a>
# ### **Checking for the missing values in the dataset.**

# In[ ]:


msno.matrix(df)


# *  ### From the above plot we can see that there are some missing values in  **lat**(Latitude) and **lng**(Longitude) features  in the dataset. 
# * ### Lets look at the missing values in the dataset.

# In[ ]:


nans = lambda df: df[df.isnull().any(axis=1)]
nans_df = nans(df)
nans_df = nans_df[['Hotel_Name','lat','lng']]
print('No of missing values in the dataset: {}'.format(len(nans_df)))


# In[ ]:


nans_df.Hotel_Name.describe()


# * We see that there are about **3268  Nan** (missing values) from  **17**  Hotels  **lat,lng** information is not available in the dataset.
# * i.e., **1.13%** of **Hotels** lat,lng information is missing.

# In[ ]:


# let's look at the reviews frequency of the missing Hotels.
nans_df.Hotel_Name.value_counts()


# * Instead of removing the **Nan** values from the dataset,
# Try to fill the **Nan** values with the similar **Hotel_Addresses *lat, lng* values**  in the dataset. 
# If the **Hotel_Address** is matched with the other rows(i.e Nan valued rows) in the dataset, Fill the **Nan** values in the dataset with the matched values(i.e., **lat**,**lng**).
# * Let's look into the datset to find the similar Hotel information is availble.

# In[ ]:


print('No of reviews in the dataset to that Hotel:')
print('Fleming s Selection Hotel Wien City: {}'.format(len(df.loc[df.Hotel_Name == 'Fleming s Selection Hotel Wien City'])))
print('Hotel City Central: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel City Central'])))
print('Hotel Atlanta: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel Atlanta'])))
print('Maison Albar Hotel Paris Op ra Diamond: {}'.format(len(df.loc[df.Hotel_Name == 'Maison Albar Hotel Paris Op ra Diamond'])))
print('Hotel Daniel Vienna: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel Daniel Vienna'])))
print('Hotel Pension Baron am Schottentor: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel Pension Baron am Schottentor'])))
print('Austria Trend Hotel Schloss Wilhelminenberg Wien: {}'.format(len(df.loc[df.Hotel_Name == 'Austria Trend Hotel Schloss Wilhelminenberg Wien'])))
print('Derag Livinghotel Kaiser Franz Joseph Vienna: {}'.format(len(df.loc[df.Hotel_Name == 'Derag Livinghotel Kaiser Franz Joseph Vienna'])))
print('NH Collection Barcelona Podium: {}'.format(len(df.loc[df.Hotel_Name == 'NH Collection Barcelona Podium'])))
print('City Hotel Deutschmeister: {}'.format(len(df.loc[df.Hotel_Name == 'City Hotel Deutschmeister'])))
print('Hotel Park Villa: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel Park Villa'])))
print('Cordial Theaterhotel Wien: {}'.format(len(df.loc[df.Hotel_Name == 'Cordial Theaterhotel Wien'])))
print('Holiday Inn Paris Montmartre: {}'.format(len(df.loc[df.Hotel_Name == 'Holiday Inn Paris Montmartre'])))
print('Roomz Vienna: {}'.format(len(df.loc[df.Hotel_Name == 'Roomz Vienna'])))
print('Mercure Paris Gare Montparnasse: {}'.format(len(df.loc[df.Hotel_Name == 'Mercure Paris Gare Montparnasse'])))
print('Renaissance Barcelona Hotel: {}'.format(len(df.loc[df.Hotel_Name == 'Renaissance Barcelona Hotel'])))
print('Hotel Advance: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel Advance'])))


# * From the above figures we see that the missing values and available values in the dataset are same.(i.e the inflat,lng values are not available in the entire dataset).
# * So, Now we can fill the **NaN** values in the dataset manually. (Simply we can ignore those rows in the dataset by removing them. But i decided not to delete the information and fill the **lat,lng** values manually just because when it comes to Business problem if i try to remove the data i am losing information of 17 Hotel's. It seems like losing our 17 clients.)
# 
# * For filling the **lat,lng** information of Hotel's by using this site [http://latlong.org/].

# In[ ]:


#latitude information of Hotels
loc_lat = {'Fleming s Selection Hotel Wien City':48.209270,
       'Hotel City Central':48.2136,
       'Hotel Atlanta':48.210033,
       'Maison Albar Hotel Paris Op ra Diamond':48.875343,
       'Hotel Daniel Vienna':48.1888,
       'Hotel Pension Baron am Schottentor':48.216701,
      'Austria Trend Hotel Schloss Wilhelminenberg Wien':48.2195,
      'Derag Livinghotel Kaiser Franz Joseph Vienna':48.245998,
      'NH Collection Barcelona Podium':41.3916,
      'City Hotel Deutschmeister':48.22088,
      'Hotel Park Villa':48.233577,
      'Cordial Theaterhotel Wien':48.209488,
      'Holiday Inn Paris Montmartre':48.888920,
      'Roomz Vienna':48.186605,
      'Mercure Paris Gare Montparnasse':48.840012,
      'Renaissance Barcelona Hotel':41.392673,
      'Hotel Advance':41.383308}


# In[ ]:


#longitude information of Hotels
loc_lng ={'Fleming s Selection Hotel Wien City':16.353479,
       'Hotel City Central':16.3799,
       'Hotel Atlanta':16.363449,
       'Maison Albar Hotel Paris Op ra Diamond':2.323358,
       'Hotel Daniel Vienna':16.3840,
       'Hotel Pension Baron am Schottentor':16.359819,
      'Austria Trend Hotel Schloss Wilhelminenberg Wien':16.2856,
      'Derag Livinghotel Kaiser Franz Joseph Vienna':16.341080,
      'NH Collection Barcelona Podium':2.1779,
      'City Hotel Deutschmeister':16.36663,
      'Hotel Park Villa':16.345682,
      'Cordial Theaterhotel Wien':16.351585,
      'Holiday Inn Paris Montmartre':2.333087,
      'Roomz Vienna':16.420643,
      'Mercure Paris Gare Montparnasse':2.323595,
      'Renaissance Barcelona Hotel':2.167494,
      'Hotel Advance':2.162828}


# In[ ]:


#filling the latitude information
df['lat'] = df['lat'].fillna(df['Hotel_Name'].apply(lambda x: loc_lat.get(x)))
#filling longitude information
df['lng'] = df['lng'].fillna(df['Hotel_Name'].apply(lambda x: loc_lng.get(x)))


# In[ ]:


#looking whether information is correctly filled or not.
msno.matrix(df)


# In[ ]:


#saving the data to pickle files
df.to_pickle('After_filling_Nans')


# In[ ]:


#loading the data from the pickle file
df = pd.read_pickle('After_filling_Nans')


# <a id='hotelname'></a>
# ### ** Basic stats for the feature: Hotel_Name**

# In[ ]:


df.Hotel_Name.describe()


# * There are **1492** Hotel Names and the most reviewed Hotel is **Britannia International Hotel Canary Wharf** with **4789** reviews.

# In[ ]:


# Let's look at the top 10 reviewed Hotels
Hotel_Name_count = df.Hotel_Name.value_counts()
Hotel_Name_count[:10].plot(kind='bar',figsize=(10,8))


# <a id='avg'></a>
# ### **Basic stats for the feature: Average_Score **

# In[ ]:


import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 50, 18
rcParams["axes.labelsize"] = 16
from matplotlib import pyplot
import seaborn as sns


# In[ ]:


data_plot = df[["Hotel_Name","Average_Score"]].drop_duplicates()
sns.set(font_scale = 2.5)
a4_dims = (30, 12)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.countplot(ax = ax,x = "Average_Score",data=data_plot)


# * **we see that most of the Hotels average_score lie in the range of 8.0 and 9.1 range **

# <a id='rn'></a>
# ### **Basic stats for the feature: Review_Nationality**

# In[ ]:


text = ""
for i in range(df.shape[0]):
    text = " ".join([text,df["Reviewer_Nationality"].values[i]])


# In[ ]:


from wordcloud import WordCloud
wordcloud = WordCloud(background_color='black', width = 600,                      height=200, max_font_size=50, max_words=40).generate(text)
wordcloud.recolor(random_state=312)
plt.imshow(wordcloud)
plt.title("Wordcloud for countries ")
plt.axis("off")
plt.show()


# In[ ]:


df.Reviewer_Nationality.describe()


# In[ ]:


# Let's look at the Top 10 Reviewer's Nationalities
Reviewer_Nat_Count = df.Reviewer_Nationality.value_counts()
print(Reviewer_Nat_Count[:10])


# ### The Reviewers belongs to **227** different countries and almost  **47.57%(245110/515212)** of Reviewers are from **United Kingdom**

# <a id='rd'></a>
# ### **Basic stats for the feature: Review_Date**

# In[ ]:


df.Review_Date.describe()


# *  There Reviews are given on **731 dates** and the most Reviews are given on **8/2/2017**

# In[ ]:


# Let's look at the top 10 Reviews given dates
Review_Date_count = df.Review_Date.value_counts()
Review_Date_count[:10].plot(kind='bar')


# <a id='tnr'></a>
# ###  **Basic stats for the feature: Total_Number_of_Reviews_Reviewer_Has_Given	**

# In[ ]:


Reviewers_freq = df.Total_Number_of_Reviews_Reviewer_Has_Given.value_counts()
Reviewers_freq[:10].plot(kind='bar')


# In[ ]:


Reviewers_freq[:10]


#  * We see that almost **29.99%**  (154506 / 515212)  of user's reviewed for the **first_time**.

# <a id='latlng'></a>
# ##  Let's plot a interactive map visualitation inorder to see where the Hotels are located using **lat,lng** information.
# * we are using the beautiful map visualization library called folium. (if you want to know more about folium library check out this link    https://media.readthedocs.org/pdf/folium/latest/folium.pdf )

# In[ ]:


#Loading the unique Hotel's information to plot them on the map
temp_df = df.drop_duplicates(['Hotel_Name'])
len(temp_df)


# In[ ]:


map_osm = folium.Map(location=[47, 6], zoom_start=5, tiles = 'Stamen Toner' )

temp_df.apply(lambda row:folium.Marker(location=[row["lat"], row["lng"]])
                                             .add_to(map_osm), axis=1)

map_osm


# **Observations:**
# * From the map we can see all the 1492 Hotels are located in 6 cities (Bercelona, Paris, Milan, Vienna, London, Amsterdam).         

# <a id='rtp'></a>
# ### **Basic stats for the feature: Review_Total_Positive_Word_Counts	**

# In[ ]:


pos_words = df.Review_Total_Positive_Word_Counts.value_counts()
pos_words[:10]


# *  we see that **0** words are more in number it means they are completely **Negative reviews**. Lets have a look at them.

# In[ ]:


a = df.loc[df.Review_Total_Positive_Word_Counts == 0]
print('No of completely Negative reviews in the dataset:',len(a))
b = a[['Positive_Review','Negative_Review']]
b[:10]


# By looking those reviews we can conclude that they are completely **Negative reviews**.

# <a id='rtn'></a>
# ### **Basic stats for the feature: Review_Total_Negative_Word_Counts	**

# In[ ]:


neg_words = df.Review_Total_Negative_Word_Counts.value_counts()
neg_words[:10]


# *  we see that **0** words are more in number it means they are completely **Positive reviews**. Lets have a look at them.

# In[ ]:


a = df.loc[df.Review_Total_Negative_Word_Counts == 0 ]
print('No of completely positive reviews in the dataset:',len(a))
b = a[['Positive_Review','Negative_Review']]
b[:10]


# ## **Observations**:
# * By using **Review_Total_Negative_Word_Counts and Review_Total_Positive_Word_Counts **  attributes we can classify **1,63,661** reviews only. (i.e.,**31.76%** of the total reviews). so these attributes can't be used for classifying the reviews.
# * i also found that with the **word_count- 2 **  ** ' everything', ' Everything', ' nothing', ' Nothing' ** words are present in both *Positive_Review* and *Negative_Review*. ( Those words are preceded by white spaces in the begning' that's why their words count is **2** 
# * By including ** ' everything', ' Everything', ' nothing', ' Nothing' ** words we can classify more reviews

# <a id='posneg'></a>
# ### **Calculating no of positve and negative reviews**

# In[ ]:


# For classifying positive and negative reviews
df['pos_count']=0
df['neg_count']=0


# In[ ]:


# since we found the words are in mixed case letters and with trailing whitespace 
#we remove those white spaces and converting the reviews to lowercases
df['Negative_Review']=[x.lower().strip() for x in df['Negative_Review']]
df['Positive_Review']=[x.lower().strip() for x in df['Positive_Review']]


# In[ ]:


#if the Positive_Review contains the words 'no positive' and 'nothing' are considered as a Negative_Review.
# if the Negative_Review contains the word 'everything' it is also considered as Negative_Review.
# we are maiking those reveiews as 1 in neg_count(attribute).
df["neg_count"] = df.apply(lambda x: 1 if x["Positive_Review"] == 'no positive' or                            x['Positive_Review']=='nothing' or                            x['Negative_Review']=='everything'                            else x['pos_count'],axis = 1)


# In[ ]:


#if the Negative_Review contains the words 'no negative' and 'nothing' are considered as a Positive_Review.
#if the Positive_Review contains the word 'Everything' it is also considered as positive_Review. 
#we are making those reviews as 1 in the pos_count(attribute). 
df["pos_count"] = df.apply(lambda x: 1 if x["Negative_Review"] == 'no negative' or                            x['Negative_Review']=='nothing' or                            x['Positive_Review']=='everything'                            else x['pos_count'],axis = 1)


# In[ ]:


#seeing how many reviews are classified as positive one's
df.pos_count.value_counts()


# In[ ]:


#seeing how many reviews are classified as negative one's
df.neg_count.value_counts()


# ### By adding those words we classified (1,49,981 + 37,854) i.e., 1,87,835 reviews.

# In[ ]:


# Calculating no of positive and negative reviews for each Hotel and storing them into reviews dataset. 
reviews = pd.DataFrame(df.groupby(["Hotel_Name"])["pos_count","neg_count"].sum())


# In[ ]:


reviews.head()


# In[ ]:


# Adding index to the reviews dataframe
reviews["HoteL_Name"] = reviews.index
reviews.index = range(reviews.shape[0])
reviews.head()


# In[ ]:


#calculating total number of reviews for each hotel
reviews["total"] = reviews["pos_count"] + reviews["neg_count"]
#calculating the positive ratio for each Hotel.
reviews["pos_ratio"] = reviews["pos_count"].astype("float")/reviews["total"].astype("float")


# <a id='popular'></a>
# ### **Finding the top 20 famous Hotels**

# In[ ]:


#looking at the famous 20 hotels location in the map. Famous Hotels are calculated based on the total
#no of reviews the Hotel has.
famous_hotels = reviews.sort_values(by = "total",ascending=False).head(100)
pd.set_option('display.max_colwidth', 2000)
popular = famous_hotels["HoteL_Name"].values[:20]
popular_hotels =df.loc[df['Hotel_Name'].isin(popular)][["Hotel_Name",                                "Hotel_Address",'Average_Score','lat','lng']].drop_duplicates()
maps_osm = folium.Map(location=[47, 6], zoom_start=5, tiles = 'Stamen Toner' )
popular_hotels.apply(lambda row:folium.Marker(location=[row["lat"], row["lng"]])
                                             .add_to(maps_osm), axis=1)

maps_osm


# In[ ]:


#look at the Hotel_Name and Hotel_Address of those Hotels
popular_hotels


# ### **Observations:**
# * Among the famous **20** Hotel's **19** Hotels are located in **London** and one more is located in **Amsterdam**

# <a id=phr></a>
# ### **Finding the top 20 positive rated Hotels**

# In[ ]:


#Looking at top 20 famous hotels with positive reviews.
pos = famous_hotels.sort_values(by = "pos_ratio",ascending=False)["HoteL_Name"].head(20).values
famous_pos = df.loc[df['Hotel_Name'].isin(pos)][["Hotel_Name","Hotel_Address",'lat','lng','Average_Score']].drop_duplicates()
positive_map = folium.Map(location=[47, 6], zoom_start=5, tiles = 'Stamen Toner' )
famous_pos.apply(lambda row:folium.Marker(location=[row["lat"], row["lng"]])
                                             .add_to(positive_map), axis=1)

positive_map


# In[ ]:


#look at the Hotel_Name and Hotel_Address of those Hotels
famous_pos


# ### **Observation:**
# * Among the top **20** Hotels with positive reviews  **11** Hotels are located in London, **4** in Netherlands, **2** in Milan, **2** in Spain and **1** in Vienna
# * Most of the famous positive reviewed hotels ratings are between **8.6** to **9.3**.

# In[ ]:


#saving the dataframe to pickle file
reviews.to_pickle('reviews')


# <a id='pre'></a>
# # ** Preprocessing:**

# since the dataset has already removed the unicode and punctuation in the text data and transformed text into lower case....
# Half of the work of preprocessing is done. Let's do the remaining preprocessing tasks like removing stopwords, stemming.

# In[ ]:


#loading the positive reviews and negative reviews to a single column as text
pos_reviews = df['Positive_Review'].values
pos_reviews = pos_reviews.tolist()
neg_reviews = df['Negative_Review'].values
neg_reviews = neg_reviews.tolist()
text = pos_reviews+neg_reviews


# In[ ]:


#providing score attribute to the review
score = ['positive' for i in range(len(pos_reviews))]
score += ['negative' for i in range(len(neg_reviews))]
#performing one-hot encoding to the score attrubute.(1- positive and 0- negative)
for i in range(0,len(score)):
    if score[i] == 'positive':
        score[i] = 1
    else:
        score[i] = 0


# In[ ]:


#loading required data to dataframe.
text_df = pd.DataFrame()
text_df['reviews'] = text
text_df['score'] = score
text_df.head()


# In[ ]:


# Perfoming preprocessing
start_time = time.time()
text = text_df['reviews'].values
print("Removing stop words...........................")
stop = set(stopwords.words('english'))
words = []
summary = []
all_pos_words = []
all_neg_words = []
for i in range(0,len(text)):
    if type(text[i]) == type('') :
        sentence = text[i]
        sentence = re.sub("[^a-zA-Z]"," ", sentence)
        buffer_sentence = [i for i in sentence.split() if i not in stop]
        word = ''
        for j in buffer_sentence:
            if len(j) >= 2:
                if i<=(len(text)/2): 
                    all_pos_words.append(j)
                else:
                    all_neg_words.append(j)
                word +=' '+j
        summary.append(word)    
print("performing stemming............................")
porter = PorterStemmer()
for i in range(0,len(summary)):
    summary[i] = porter.stem(summary[i])
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


# no of words in positive and negative reviews
len(all_pos_words),len(all_neg_words)


# In[ ]:


# displaying the frequency of words in positive and negative reviews 
freq_dist_pos = Counter(all_pos_words)
freq_dist_neg = Counter(all_neg_words)
print('Most common positive words : ',freq_dist_pos.most_common(20))
print('Most common negative words : ',freq_dist_neg.most_common(20))


# In[ ]:


# no of positive and negative words
len(freq_dist_neg),len(freq_dist_pos)


# In[ ]:


#converting the summary numpy array
score = text_df['score'].values


# In[ ]:


# loading the data to dataframe and saving it into pickle file
text_df = pd.DataFrame()
text_df['Summary'] = summary
text_df['score'] = score
text_df.to_pickle('text_df')


# ### Sentiment Analysis for these reviews can be seen in this kernal:
# [https://www.kaggle.com/anumulamuralidhar/sentiment-analysis-for-515k-hotel-reviews]
