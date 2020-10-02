#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import os
import seaborn as sns
print(os.listdir("../input"))
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=False)
from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
import folium
from tqdm import tqdm
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from gensim.models import word2vec
import nltk
# Any results you write to the current directory are saved as output.1


# In[ ]:


df=pd.read_csv("../input/zomato.csv")


# ## [Getting Basic Ideas]()<a id="2"></a> <br>
# 

# In[ ]:


print("dataset contains {} rows and {} columns".format(df.shape[0],df.shape[1]))


# In[ ]:


df.info()


# In[ ]:


df


# **Columns description**
# 
# - **url**
# contains the url of the restaurant in the zomato website
# 
# - **address**
# contains the address of the restaurant in Bengaluru
# 
# - **name**
# contains the name of the restaurant
# 
# - **online_order**
# whether online ordering is available in the restaurant or not
# 
# - **book_table**
# table book option available or not
# 
# - **rate**
# contains the overall rating of the restaurant out of 5
# 
# - **votes**
# contains total number of rating for the restaurant as of the above mentioned date
# 
# - **phone**
# contains the phone number of the restaurant
# 
# - **location**
# contains the neighborhood in which the restaurant is located
# 
# - **rest_type**
# restaurant type
# 
# - **dish_liked**
# dishes people liked in the restaurant
# 
# - **cuisines**
# food styles, separated by comma
# 
# - **approx_cost(for two people)**
# contains the approximate cost for meal for two people
# 
# - **reviews_list**
# list of tuples containing reviews for the restaurant, each tuple 
# 
# - **menu_item**
# contains list of menus available in the restaurant
# 
# - **listed_in(type)**
# type of meal
# 
# - **listed_in(city)**
# contains the neighborhood in which the restaurant is listed
# 

# - As you can see **Cafe coffee day,Onesta,Just Bake** has the most number of outlets in and around bangalore.
# - This is rather interesting,we will inspect each of them later.

# ## [Rating distribution]()<a id="6"></a> <br>
# 

# In[ ]:


plt.figure(figsize=(6,5))
rating=df['rate'].dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3.5)  else np.nan ).dropna()
sns.distplot(rating,bins=20)


# 1. Almost more than 50 percent of restaurants has rating between 3 and 4.
# 2. Restaurants having rating more than 4.5 are very rare.

# ## [Which are the most common restaurant type in Banglore?]()<a id="8"></a> <br>
# 

# In[ ]:


plt.figure(figsize=(7,7))
rest=df['rest_type'].value_counts()[:20]
sns.barplot(rest,rest.index)
plt.title("Restaurant types")
plt.xlabel("count")


# 1. No doubt about this as Banglore is known as the tech capital of India,people having busy and modern life will prefer Quick Bites.
# 2. We can observe tha Quick Bites type restaurants dominates.

# ## [Which are  the most popular cuisines of Bangalore?]()<a id="14"></a> <br>
# 

# In[ ]:


plt.figure(figsize=(7,7))
cuisines=df['cuisines'].value_counts()[:10]
sns.barplot(cuisines,cuisines.index)
plt.xlabel('Count')
plt.title("Most popular cuisines of Bangalore")


# 1. We can observe that **North Indian,chinese,South Indian and Biriyani** are most common.
# 2. Is this imply the fact that Banglore is more influenced by North Indian culture more than South?
# - We will inspect further......

# ## [Analysing Reviews]()<a id="26"></a><br>

# - In this section we will go on to prepare reviews dataframe.
# - We will extract reviews and ratings of each restaurant and create a dataframe with it.
# 

# In[ ]:


all_ratings = []

for name,ratings in tqdm(zip(df['name'],df['reviews_list'])):
    ratings = eval(ratings)
    for score, doc in ratings:
        if score:
            score = score.strip("Rated").strip()
            doc = doc.strip('RATED').strip()
            score = float(score)
            all_ratings.append([name,score, doc])


# In[ ]:


rating_df = pd.DataFrame(all_ratings,columns=['name','rating','review'])
rating_df['review']=rating_df['review'].apply(lambda x : re.sub('[^a-zA-Z0-9\s]',"",x))


# In[ ]:


rating_df.to_csv("Ratings.csv")


# In[ ]:


columns = ['rest_type','dish_liked','cuisines','votes']
df_ratings_add = df[columns]
df_ratings_add


# In[ ]:


result = pd.concat([rating_df, df_ratings_add], axis=1, join='inner')


# In[ ]:


result


# In[ ]:


result['sent']=result['rating'].apply(lambda x: 1 if int(x)>=3.0 else 0)
result


# In[ ]:


punctuation_signs = list("?:!.,;")

for punct_sign in punctuation_signs:
    result['cuisines'] = result['cuisines'].str.replace(punct_sign, '')
    result['dish_liked'] = result['dish_liked'].str.replace(punct_sign, '')
    result['review'] = result['review'].str.replace(punct_sign, '')
    result['rest_type'] = result['rest_type'].str.replace(punct_sign, '')


# In[ ]:


result


# In[ ]:


result


# In[ ]:


# \r and \n
result['cuisines'] = result["cuisines"].str.replace("\r", " ")
result['cuisines'] = result["cuisines"].str.replace("\n", " ")
result['cuisines'] = result["cuisines"].str.replace("    ", " ")
result['cuisines'] = result["cuisines"].str.replace('"', '')
result['cuisines'] = result["cuisines"].str.lower()
result['cuisines'] = result["cuisines"].dropna()

punctuation_signs = list("?:!.,;")
for punct_sign in punctuation_signs:
    result['cuisines'] = result["cuisines"].str.replace(punct_sign, '')
    
result['cuisines'] = result["cuisines"].str.replace("'s", "")


# In[ ]:


# Downloading the stop words list
nltk.download('stopwords')
# Loading the stop words in english
stop_words = list(stopwords.words('english'))

result['cuisines'] = result["cuisines"]

for stop_word in stop_words:

    regex_stopword = r"\b" + stop_word + r"\b"
    result['cuisines'] = result["cuisines"].str.replace(regex_stopword, '')


# In[ ]:


result['cuisines']


# In[ ]:


# find the corelation between inputs

num_cols = ['rating','votes','rest_type']

corr = result[num_cols].corr()

# plot heatmap
sns.heatmap(corr, 
            xticklabels=corr.columns.values, yticklabels=corr.columns.values,
            cmap=sns.light_palette("navy"),
           )
plt.show()


# In[ ]:


X = result['cuisines']
y = result['sent']


# In[ ]:


processed_tweets = []
 
for tweet in range(0, len(X)):  
    # Remove all the special characters
    processed_tweet = re.sub(r'\W', ' ', str(X[tweet]))
 
    # remove all single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
 
    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
 
    # Substituting multiple spaces with single space
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
 
    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
 
    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()
 
    processed_tweets.append(processed_tweet)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer  
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(processed_tweets).toarray()


# In[ ]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


X_train


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))


# In[ ]:


y_pred = knn.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

