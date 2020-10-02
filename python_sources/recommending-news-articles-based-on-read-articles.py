#!/usr/bin/env python
# coding: utf-8

# ## Preface

# Due to easy availibilty of enormous items(services) on your favourite online platforms like *e-commerce*, *job portal*, *food delivery*, *music or video streaming*, it is very hard and time consuming to find out the desired item of your choice quickly. These platforms could help you by **recommending** items as per your interest and preference by just analyzing your past interaction or behaviour with the system.  
# 
# From **Amazon** to **Linkedin**, **Uber eats** to **Spotify**, **Netflix** to **Facebook**, **Recommender systems** are most extensively used to suggest "Similar items", "Relevant jobs", "preferred foods", "Movies of interest" etc to their users. 
# 
# **Recommender system** with appropiate item suggestions helps in boosting sales, increasing revenue, retaining customers and also adds competitive advantage. 
# There are basically two kind of **recommendation** methods.
# 1. **Content based recommendation**
# 2. **Collaborative filtering**

# **Content based recommendation ** is based on similarity among users/items obtained through their **attributes**. It uses the additional information(meta data) about the **users** or **items** i.e. it relies on what kind of **content** is already available. This meta data could be **user's demograpic information** like *age*, *gender*, *job*, *location*, *skillsets* etc. Similarly for **items** it can be *item name*, *specifications*, *category*, *registration date* etc.
# 
# So the core idea is to recommend items by finding similar items/users to the concerned **item/user** based on their **attributes**. 
# 
# In this kernel, I am going to discuss about **Content based recommendation** using **News category** dataset. The goal is to recommend **news articles** which are similar to the already read article by using attributes like article *headline*, *category*, *author* and *publishing date*.
# 
# So let's get started without any further delay.

# ## Notebook - Table of Content

# 1. [**Importing necessary Libraries**](# 1.-Importing-necessary-Libraries)   
# 2. [**Loading Data**](#2.-Loading-Data)  
# 3. [**Data Preprocessing**](#3.-Data-Preprocessing)  
#     3.a [**Fetching only the articles from 2018**](#3.a-Fetching-only-the-articles-from-2018)  
#     3.b [**Removing all the short headline articles**](#3.b-Removing-all-the-short-headline-articles)  
#     3.c [**Checking and removing all the duplicates**](#3.c-Checking-and-removing-all-the-duplicates)  
#     3.d [**Checking for missing values**](#3.d-Checking-for-missing-values)  
# 4. [**Basic Data Exploration**](#4.-Basic-Data-Exploration)  
#     4.a [**Basic statistics - Number of articles,authors,categories**](#4.a-Basic-statistics---Number-of-articles,authors,categories)  
#     4.b [**Distribution of articles category-wise**](#4.b-Distribution-of-articles-category-wise)  
#     4.c [**Number of articles per month**](#4.c-Number-of-articles-per-month)   
#     4.d [**PDF for length of headlines**](#4.d-PDF-for-length-of-headlines)
# 5. [**Text Preprocessing**](#5.-Text-Preprocessing)  
#     5.a [**Stopwords removal**](#5.a-Stopwords-removal)  
#     5.b [**Lemmatization**](#5.b-Lemmatization)  
# 6. [**Headline based similarity on new articles**](#6.-Headline-based-similarity-on-new-articles)  
#     6.a [**Using Bag of Words method**](#6.a-Using-Bag-of-Words method)  
#     6.b [**Using TF-IDF method**](#6.b-Using-TF-IDF-method)  
#     6.c [**Using Word2Vec embedding**](#6.c-Using-Word2Vec-embedding)   
#     6.d [**Weighted similarity based on headline and category**](#6.d-Weighted-similarity-based-on-headline-and-category)  
#     6.e [**Weighted similarity based on headline, category and author**](#6.e-Weighted-similarity-based-on-headline,-category-and-author)  
#     6.f [**Weighted similarity based on headline, category, author and publishing day**](#6.f-Weighted-similarity-based-on-headline,-category,-author-and-publishing-day)  
#  

# ## 1. Importing necessary Libraries

# In[ ]:


import numpy as np
import pandas as pd

import os
import math
import time

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

# Below libraries are for text processing using NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Below libraries are for feature representation using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Below libraries are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances


# ## 2. Loading Data

# In[ ]:


news_articles = pd.read_json("/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json", lines = True)


# In[ ]:


news_articles.info()


# The dataset contains about two million records of six different features. 

# In[ ]:


news_articles.head()


# ## 3. Data Preprocessing

# ### 3.a Fetching only the articles from 2018  

# Since the dataset size is quite large so processing through entire dataset may consume too much time. To refrain from this, we are only considering the latest articles from the year 2018. 

# In[ ]:


news_articles = news_articles[news_articles['date'] >= pd.Timestamp(2018,1,1)]


# In[ ]:


news_articles.shape


# Now, the number of news articles comes down to 8583.

# ### 3.b Removing all the short headline articles 

# After stop words removal from headline, the articles with very short headline may become blank headline articles. So let's remove all the articles with less words(<5) in the headline.   

# In[ ]:


news_articles = news_articles[news_articles['headline'].apply(lambda x: len(x.split())>5)]
print("Total number of articles after removal of headlines with short title:", news_articles.shape[0])


# ### 3.c Checking and removing all the duplicates

# Since some articles are exactly same in headlines, so let's remove all such articles having duplicate headline appearance.

# In[ ]:


news_articles.sort_values('headline',inplace=True, ascending=False)
duplicated_articles_series = news_articles.duplicated('headline', keep = False)
news_articles = news_articles[~duplicated_articles_series]
print("Total number of articles after removing duplicates:", news_articles.shape[0])


# ### 3.d Checking for missing values

# In[ ]:


news_articles.isna().sum()


# ## 4. Basic Data Exploration 

# ### 4.a Basic statistics - Number of articles,authors,categories

# In[ ]:


print("Total number of articles : ", news_articles.shape[0])
print("Total number of authors : ", news_articles["authors"].nunique())
print("Total number of unqiue categories : ", news_articles["category"].nunique())


# ### 4.b Distribution of articles category-wise

# In[ ]:


fig = go.Figure([go.Bar(x=news_articles["category"].value_counts().index, y=news_articles["category"].value_counts().values)])
fig['layout'].update(title={"text" : 'Distribution of articles category-wise','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Category name",yaxis_title="Number of articles")
fig.update_layout(width=800,height=700)
fig


# From the bar chart, we can observe that **politics** category has **highest** number of articles then **entertainment** and so on.  

# ### 4.c Number of articles per month

# Let's first group the data on monthly basis using **resample()** function. 

# In[ ]:


news_articles_per_month = news_articles.resample('m',on = 'date')['headline'].count()
news_articles_per_month


# In[ ]:


fig = go.Figure([go.Bar(x=news_articles_per_month.index.strftime("%b"), y=news_articles_per_month)])
fig['layout'].update(title={"text" : 'Distribution of articles month-wise','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Month",yaxis_title="Number of articles")
fig.update_layout(width=500,height=500)
fig


# From the bar chart, we can observe that **January** month has **highest** number of articles then **March** and so on.  

# ### 4.d PDF for the length of headlines 

# In[ ]:


fig = ff.create_distplot([news_articles['headline'].str.len()], ["ht"],show_hist=False,show_rug=False)
fig['layout'].update(title={'text':'PDF','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Length of a headline",yaxis_title="probability")
fig.update_layout(showlegend = False,width=500,height=500)
fig


# The probability distribution function of headline length is almost similar to a **Guassian distribution**, where most of the headlines are 58 to 80 words long in length. 

# By Data processing in Step 2, we get a subset of original dataset which has different index labels so let's make the indices uniform ranging from 0 to total number of articles. 

# In[ ]:


news_articles.index = range(news_articles.shape[0])


# In[ ]:


# Adding a new column containing both day of the week and month, it will be required later while recommending based on day of the week and month
news_articles["day and month"] = news_articles["date"].dt.strftime("%a") + "_" + news_articles["date"].dt.strftime("%b")


# Since after text preprocessing the original headlines will be modified and it doesn't make sense to recommend articles by displaying modified headlines so let's copy the dataset into some other dataset and perform text preprocessing on the later.

# In[ ]:


news_articles_temp = news_articles.copy()


# ## 5. Text Preprocessing

# ### 5.a Stopwords removal

# Stop words are not much helpful in analyis and also their inclusion consumes much time during processing so let's remove these. 

# In[ ]:


stop_words = set(stopwords.words('english'))


# In[ ]:


for i in range(len(news_articles_temp["headline"])):
    string = ""
    for word in news_articles_temp["headline"][i].split():
        word = ("".join(e for e in word if e.isalnum()))
        word = word.lower()
        if not word in stop_words:
          string += word + " "  
    if(i%1000==0):
      print(i)           # To track number of records processed
    news_articles_temp.at[i,"headline"] = string.strip()


# ### 5.b Lemmatization

# Let's find the base form(lemma) of words to consider different inflections of a word same as lemma.

# In[ ]:


lemmatizer = WordNetLemmatizer()


# In[ ]:


for i in range(len(news_articles_temp["headline"])):
    string = ""
    for w in word_tokenize(news_articles_temp["headline"][i]):
        string += lemmatizer.lemmatize(w,pos = "v") + " "
    news_articles_temp.at[i, "headline"] = string.strip()
    if(i%1000==0):
        print(i)           # To track number of records processed


# ## 6. Headline based similarity on new articles

# Generally, we assess **similarity** based on **distance**. If the **distance** is minimum then high **similarity** and if it is maximum then low **similarity**.
# To calculate the **distance**, we need to represent the headline as a **d-dimensional** vector. Then we can find out the **similarity** based on the **distance** between vectors.
# 
# There are multiple methods to represent a **text** as **d-dimensional** vector like **Bag of words**, **TF-IDF method**, **Word2Vec embedding** etc. Each method has its own advantages and disadvantages. 
# 
# Let's see the feature representation of headline through all the methods one by one.

# ### 6.a Using Bag of Words method

# A **Bag of Words(BoW)** method represents the occurence of words within a **document**. Here, each headline can be considered as a **document** and set of all headlines form a **corpus**.
# 
# Using **BoW** approach, each **document** is represented by a **d-dimensional** vector, where **d** is total number of **unique words** in the corpus. The set of such unique words forms the **Vocabulary**.

# In[ ]:


headline_vectorizer = CountVectorizer()
headline_features   = headline_vectorizer.fit_transform(news_articles_temp['headline'])


# In[ ]:


headline_features.get_shape()


# The output **BoW matrix**(headline_features) is a sparse matrix.

# In[ ]:


pd.set_option('display.max_colwidth', -1)  # To display a very long headline completely


# In[ ]:


def bag_of_words_based_model(row_index, num_similar_items):
    couple_dist = pairwise_distances(headline_features,headline_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
               'headline':news_articles['headline'][indices].values,
                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})
    print("="*30,"Queried article details","="*30)
    print('headline : ',news_articles['headline'][indices[0]])
    print("\n","="*25,"Recommended articles : ","="*23)
    #return df.iloc[1:,1]
    return df.iloc[1:,]

bag_of_words_based_model(133, 11) # Change the row index for any other queried article


# Above function recommends **10 similar** articles to the **queried**(read) article based on the headline. It accepts two arguments - index of already read artile and the total number of articles to be recommended.
# 
# Based on the **Euclidean distance** it finds out 10 nearest neighbors and recommends. 
# 
# **Disadvantages**
# 1. It gives very low **importance** to less frequently observed words in the corpus. Few words from the queried article like "employer", "flip", "fire" appear less frequently in the entire corpus so **BoW** method does not recommend any article whose headline contains these words. Since **trump** is commonly observed word in the corpus so it is recommending the articles with headline containing "trump".   
# 2. **BoW** method doesn't preserve the order of words.
# 
# To overcome the first disadvantage we use **TF-IDF** method for feature representation. 
# 

# ### 6.b Using TF-IDF method

# **TF-IDF** method is a weighted measure which gives more importance to less frequent words in a corpus. It assigns a weight to each term(word) in a document based on **Term frequency(TF)** and **inverse document frequency(IDF)**.
# 
# **TF(i,j)** = (# times word i appears in document j) / (# words in document j)
# 
# **IDF(i,D)** = log_e(#documents in the corpus D) / (#documents containing word i)
# 
# weight(i,j) = **TF(i,j)** x **IDF(i,D)**
# 
# So if a word occurs more number of times in a document but less number of times in all other documents then its **TF-IDF** value will be high.
# 

# In[ ]:


tfidf_headline_vectorizer = TfidfVectorizer(min_df = 0)
tfidf_headline_features = tfidf_headline_vectorizer.fit_transform(news_articles_temp['headline'])


# In[ ]:


def tfidf_based_model(row_index, num_similar_items):
    couple_dist = pairwise_distances(tfidf_headline_features,tfidf_headline_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
               'headline':news_articles['headline'][indices].values,
                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})
    print("="*30,"Queried article details","="*30)
    print('headline : ',news_articles['headline'][indices[0]])
    print("\n","="*25,"Recommended articles : ","="*23)
    
    #return df.iloc[1:,1]
    return df.iloc[1:,]
tfidf_based_model(133, 11)


# Compared to **BoW** method, here **TF-IDF** method recommends the articles with headline containing words like "employer", "fire", "flip" in top 5 recommendations and these words occur less frequently in the corpus.   

# **Disadvantages :- **
# 
# **Bow** and **TF-IDF** method do not capture **semantic** and **syntactic** similarity of a given word with other words but this can be captured using **Word embeddings**.
# 
# For example: there is a good association between words like "trump" and "white house", "office and employee", "tiger" and "leopard", "USA" and "Washington D.C" etc. Such kind of **semantic** similarity can be captured using **word embedding** techniques.
# **Word embedding** techniques like **Word2Vec**, **GloVe** and **fastText** leverage semantic similarity between words. 

# ### 6.c Using Word2Vec embedding

# **Word2Vec** is one of the techniques for **semantic** similarity which was invented by **Google** in 2013. For a given corpus, during training it observes the patterns and respresents each word by a **d-dimensional** vector. To get better results we need fairly large corpus.
# 
# Since our corpus size is small so let's use Google's pretrained model on **google news** articles. This standard model contains vector representation for billions of words obtained by training on millions of new articles. Here, each word is represented by a **300** dimensional dense vector. 
# 
# 
# 

# In[ ]:


from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle


# Since this **pre-trained Word2Vec** model is **1.5 GB** in compressed form. So it needs a high end RAM to load it in the memory after unzipping.
# 
# Here, we are loading this pre-build model from a **pickle** file which contains this model in advance.

# In[ ]:


os.chdir(r'/kaggle/input/')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


with open('googlew2v/word2vec_model', 'rb') as file:
    loaded_model = pickle.load(file)


# Since the model gives vector representation for each **word** but we calculate the distance between **headlines** so we need to obtain vector representation for each **headline**. One way to obtain this is by first adding vector representation of all the words available in the **headline** and then calculating the average. It is also known as **average Word2Vec** model.   
# 
# Below code cell performs the same. 

# In[ ]:


vocabulary = loaded_model.keys()
w2v_headline = []
for i in news_articles_temp['headline']:
    w2Vec_word = np.zeros(300, dtype="float32")
    for word in i.split():
        if word in vocabulary:
            w2Vec_word = np.add(w2Vec_word, loaded_model[word])
    w2Vec_word = np.divide(w2Vec_word, len(i.split()))
    w2v_headline.append(w2Vec_word)
w2v_headline = np.array(w2v_headline)


# In[ ]:


def avg_w2v_based_model(row_index, num_similar_items):
    couple_dist = pairwise_distances(w2v_headline, w2v_headline[row_index].reshape(1,-1))
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
               'headline':news_articles['headline'][indices].values,
                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})
    print("="*30,"Queried article details","="*30)
    print('headline : ',news_articles['headline'][indices[0]])
    print("\n","="*25,"Recommended articles : ","="*23)
    #return df.iloc[1:,1]
    return df.iloc[1:,]

avg_w2v_based_model(133, 11)


# Here, **Word2Vec** based representation recommends the headlines containing the word **white house** which is associated with the word **trump** in the queried article. Similarly, it recommends the headlines with words like "offical", "insist" which have semantic similarity to the words "employer", "sue" in the queried headline.

# So far we were recommending using only one feature i.e. **headline** but in order to make a **robust** recommender system we need to consider **multiple** features at a time. Based on the business interest and rules, we can decide weight for each feature.
# 
# Let's see different models with combinations of different features for article similarity.

# ### 6.d Weighted similarity based on headline and category

# Let's first see articles similarity based on **headline** and **category**. We are using **onehot encoding** to get feature representation for **category**.
# 
# Sometimes as per the business requirements, we may need to give more preference to the articles from the same **category**. In such cases we can assign more weight to **category** while recommending. Higher the weight, more the significance. Similarly less weight leads to less signficance to a particular feature.
# 

# In[ ]:


from sklearn.preprocessing import OneHotEncoder 


# In[ ]:


category_onehot_encoded = OneHotEncoder().fit_transform(np.array(news_articles_temp["category"]).reshape(-1,1))


# In[ ]:


def avg_w2v_with_category(row_index, num_similar_items, w1,w2): #headline_preference = True, category_preference = False):
    w2v_dist  = pairwise_distances(w2v_headline, w2v_headline[row_index].reshape(1,-1))
    category_dist = pairwise_distances(category_onehot_encoded, category_onehot_encoded[row_index]) + 1
    weighted_couple_dist   = (w1 * w2v_dist +  w2 * category_dist)/float(w1 + w2)
    indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
               'headline':news_articles['headline'][indices].values,
                'Weighted Euclidean similarity with the queried article': weighted_couple_dist[indices].ravel(),
                'Word2Vec based Euclidean similarity': w2v_dist[indices].ravel(),
                 'Category based Euclidean similarity': category_dist[indices].ravel(),
                'Categoty': news_articles['category'][indices].values})
    
    print("="*30,"Queried article details","="*30)
    print('headline : ',news_articles['headline'][indices[0]])
    print('Categoty : ', news_articles['category'][indices[0]])
    print("\n","="*25,"Recommended articles : ","="*23)
    #return df.iloc[1:,[1,5]]
    return df.iloc[1:, ]

avg_w2v_with_category(528,10,0.1,0.8)


# Above function takes two extra arguments **w1** and **w2** for weights corresponding to **headline** and **category**. It is always a good practice to pass the **weights** in a range scaled from **0 to 1**, where a value close to 1 indicates high weight whereas close to 0 indicates less weight.  
# 
# Here, we can observe that the recommended articles are from the same **category** as the queried article **category**. This is due to passing of high value to **w2**.

# ### 6.e Weighted similarity based on headline, category and author

# Now let's see calcualte articles similarity based on **author** along with **headline** and **category**. Again, we are encoding **author** through **onehot encoding**.

# In[ ]:


authors_onehot_encoded = OneHotEncoder().fit_transform(np.array(news_articles_temp["authors"]).reshape(-1,1))


# In[ ]:


def avg_w2v_with_category_and_authors(row_index, num_similar_items, w1,w2,w3): #headline_preference = True, category_preference = False):
    w2v_dist  = pairwise_distances(w2v_headline, w2v_headline[row_index].reshape(1,-1))
    category_dist = pairwise_distances(category_onehot_encoded, category_onehot_encoded[row_index]) + 1
    authors_dist = pairwise_distances(authors_onehot_encoded, authors_onehot_encoded[row_index]) + 1
    weighted_couple_dist   = (w1 * w2v_dist +  w2 * category_dist + w3 * authors_dist)/float(w1 + w2 + w3)
    indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
                'headline':news_articles['headline'][indices].values,
                'Weighted Euclidean similarity with the queried article': weighted_couple_dist[indices].ravel(),
                'Word2Vec based Euclidean similarity': w2v_dist[indices].ravel(),
                'Category based Euclidean similarity': category_dist[indices].ravel(),
                'Authors based Euclidean similarity': authors_dist[indices].ravel(),       
                'Categoty': news_articles['category'][indices].values,
                'Authors': news_articles['authors'][indices].values})
    print("="*30,"Queried article details","="*30)
    print('headline : ',news_articles['headline'][indices[0]])
    print('Categoty : ', news_articles['category'][indices[0]])
    print('Authors : ', news_articles['authors'][indices[0]])
    print("\n","="*25,"Recommended articles : ","="*23)
    #return df.iloc[1:,[1,6,7]]
    return df.iloc[1:, ]


avg_w2v_with_category_and_authors(528,10,0.1,0.1,1)


# Above function takes one extra weight argument **w3** for **author**.
# 
# In the ouput, we can observe that the recommended articles are from the same **author** as the queried article **author** due to high weightage to **w3**.

# ### 6.f Weighted similarity based on headline, category, author and publishing day 

# Now let's see calcualte articles similarity based on the publishing **week day** **author** along with **headline**, **category** and **author**. Again, we are encoding this new feature through **onehot encoding**.

# In[ ]:


publishingday_onehot_encoded = OneHotEncoder().fit_transform(np.array(news_articles_temp["day and month"]).reshape(-1,1))


# In[ ]:


def avg_w2v_with_category_authors_and_publshing_day(row_index, num_similar_items, w1,w2,w3,w4): #headline_preference = True, category_preference = False):
    w2v_dist  = pairwise_distances(w2v_headline, w2v_headline[row_index].reshape(1,-1))
    category_dist = pairwise_distances(category_onehot_encoded, category_onehot_encoded[row_index]) + 1
    authors_dist = pairwise_distances(authors_onehot_encoded, authors_onehot_encoded[row_index]) + 1
    publishingday_dist = pairwise_distances(publishingday_onehot_encoded, publishingday_onehot_encoded[row_index]) + 1
    weighted_couple_dist   = (w1 * w2v_dist +  w2 * category_dist + w3 * authors_dist + w4 * publishingday_dist)/float(w1 + w2 + w3 + w4)
    indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
                'headline_text':news_articles['headline'][indices].values,
                'Weighted Euclidean similarity with the queried article': weighted_couple_dist[indices].ravel(),
                'Word2Vec based Euclidean similarity': w2v_dist[indices].ravel(),
                'Category based Euclidean similarity': category_dist[indices].ravel(),
                'Authors based Euclidean similarity': authors_dist[indices].ravel(),   
                'Publishing day based Euclidean similarity': publishingday_dist[indices].ravel(), 
                'Categoty': news_articles['category'][indices].values,
                'Authors': news_articles['authors'][indices].values,
                'Day and month': news_articles['day and month'][indices].values})
    print("="*30,"Queried article details","="*30)
    print('headline : ',news_articles['headline'][indices[0]])
    print('Categoty : ', news_articles['category'][indices[0]])
    print('Authors : ', news_articles['authors'][indices[0]])
    print('Day and month : ', news_articles['day and month'][indices[0]])
    print("\n","="*25,"Recommended articles : ","="*23)
    #return df.iloc[1:,[1,7,8,9]]
    return df.iloc[1:, ]


avg_w2v_with_category_authors_and_publshing_day(528,10,0.1,0.1,0.1,1)


# Above function takes one extra weight argument **w4** for **day of the week and month**.
# 
# In the ouput, we can observe that the recommended articles are from the same **day of the week and month** as the queried article due to high weightage to **w4**.
