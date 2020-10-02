#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('cp -r ../input/tweetcovid19/* ./')


# # Covid-19 Tweets - Sentiment Analysis

# <b>Covid-19</b> is an infectious disease caused by the newly discovered coronavirus.It was first identified in December 2019 in <b>Wuhan, China</b>, and has resulted in an ongoing <b>pandemic</b>.
# It spreads primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes. Most infected with the virus have faced mild respiratory illness with around <b>80%</b> cases being <b>asymptomatic</b>. People with underlying medical problems are found to be more likely to develop serious illness.

#  ![](img/covid19.jpg)

# Tweets are a good way for people to express their thoughts in front of a large audience. Therefore analyzing tweets and predicting their sentiments helps us to better understand what is going through everyones mind, how people are coping up being quarantined for months, how it has affected the life of so many people and also some positive effects (yes there are a few) that this pandemic has caused.

# I have carried out some analysis on the same below. The sentiments for the tweets have not been provided in the dataset. So I have presented my way on how one could go about this problem so hope this helps. Also I would appreciate any kind of feedback.

# In[ ]:


#Import the necessary libraries
import pandas as pd
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
import warnings 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


df = pd.read_csv('data/covid_19_tweets.CSV')


# Let's look at the top five rows of our dataset to get a basic overview

# In[ ]:


df.head()


# The first two columns and also the 'screen_name' column can be dropped as they wont be necessary during Visualization

# In[ ]:


df.drop(['status_id', 'user_id', 'screen_name'], axis = 1, inplace = True)


# Check the amount of missing values present

# In[ ]:


num_tweets = len(df)


# In[ ]:


df.isnull().sum()/num_tweets * 100


# There are 7 columns with more than 85% of it's values missing so I will be dropping these as they won't provide much info

# In[ ]:


missing_cols = list(df.columns[(df.isnull().sum()/num_tweets * 100) > 85.0])

df.drop(missing_cols, axis = 1, inplace = True)


# Adding a 'Language' column using the given language codes

# In[ ]:



from iso639 import languages

def get_language(x):
    try:
        return languages.get(alpha2=x).name 
    except KeyError:
        return x


# In[ ]:


df['language'] = df['lang'].apply(lambda x: get_language(x))


# In[ ]:


df['language'].value_counts()[:10]


# In[ ]:


df['language'] = df['language'].str.replace('und','Undefined')


# In[ ]:


plt.figure(figsize = (15,7))
sns.barplot(x = df['language'].value_counts()[:10].index , y = df['language'].value_counts()[:10]/num_tweets*100)
plt.xlabel('Language', fontsize = 20)
plt.ylabel('Percentage of Tweets', fontsize = 20)
plt.xticks(fontsize = 15)
plt.title('Top Ten Languages with most Tweets', fontsize=20)
plt.show()


# We see that 56% of all tweets are in English which is no surprise followed by Spanish and French at 15% and 5% respectively.

# # Sentiment Classification

# Before I can continue with any visualization I have to first Classify the sentiment of each tweet

# For now I'll be only focusing on the English tweets

# In[ ]:


df_eng = df[df['language'] == 'English']


# In[ ]:


df_eng['language'].value_counts()


# Let's take a look at one of the tweets

# In[ ]:


df_eng['text'][1128]


# As you can see the tweets are not clean. They contain urls, hashtags, stop words, special chracters and some also contains emoticons. So i've made a function to clean and tokenize them which will be in my repo.

# In[ ]:


from clean_text import CleanText 
clean = CleanText()


# In[ ]:


df_eng['text_clean'] = clean.clean(df_eng['text']) #clean() removes urls, emoticons and hashtags


# In[ ]:


df_eng['text_clean'][1128]


# In[ ]:


df_eng['text_clean'] = df_eng['text_clean'].apply(lambda x: clean.tokenize(x)) #remove punctuations, stopwords, lemmatize and splits the sentences into tokens


# In[ ]:


df_eng['text_clean'][1128]


# All the tweets have been cleaned and tokenized. This will make it easier to vectorize the words and help the model to predict the sentiment of each tweet.

# In[ ]:


#Saving the dataframe as a pickle file to resume where I left off incase the kernel crashes or if I have to continue some other day
df_eng.to_pickle('pickle_files/tweets_eng.pkl') #Also reading and writing pickle files are much faster than csv


# In[ ]:


df_eng = pd.read_pickle('pickle_files/tweets_eng.pkl')


# ### Word Embeddings

# Before we can start with the classification we have to find a way to represent the words.
# Now there two popular ways to do this: <b> Word Vectors</b> and <b>Word Embeddings</b>
# 
# <b>Word Vectors</b> are high dimensional sparse (mostly 0s) vectors where each vector represents a word which is simply one hot encoded.
# <b>Word Embeddings</b> unlike word vectors represent the words in dense vectors. The words are mapped into a meaningful space where the distance between words is related to their semantic similarity.
# 

# I'll be using pretrained <b>GloVe</b> embeddings to represent the words

# In[ ]:


docs = df_eng['text_clean']

#tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1

#encode the documents
encoded_docs = t.texts_to_sequences(docs)

#pad docs to max length
padded_docs = pad_sequences(encoded_docs, maxlen = 22, padding = 'post') 


# As the sentiments for the tweets have not been provided the classifier can be trained on any public data which contains labels. So I trained my model on an <b>Airline tweets sentiment</b> dataset(availble in kaggle) in which the tweets were separated into three sentiments <b>'Positive', 'Negative'</b> and <b>'Neutral'</b>
# 
# This dataset was trained on a 6 layer <b>LSTM</b> network with pretained <b>GloVe embeddings</b>. The model is available in my repository as 'sentiment_classifier.py'. The final model which I've used below had an accuracy of:<b> Train set 87%, Dev set 81%, Test set 80%</b>

# In[ ]:


# Loading the classifier 
classifier = keras.models.load_model('Models/sentiment_classifier4.h5') #Negative: 0, Neutral: 1, Postive: 2


# In[ ]:


labels_categorical = classifier.predict(padded_docs) # Predicting the Sentiments of the Covid-19 tweets


# In[ ]:


labels_categorical[:10] #Output of each class by the softmax function


# In[ ]:


np.argmax(labels_categorical[:10], axis = 1) #np.argmax to get labels of the classes Negative: 0, Neutral: 1, Postive: 2


# In[ ]:


df_eng['labels'] = np.argmax(labels_categorical, axis = 1)


# In[ ]:


df_eng.to_pickle('pickle_files/final_df.pkl') 


# In[ ]:


df_eng = pd.read_pickle('pickle_files/final_df.pkl')


# In[ ]:


def label_to_sentiment(label):
    if label == 0:
        return 'Negative'
    elif label == 1:
        return 'Neutral'
    else:
        return 'Positive'


# In[ ]:


df_eng['sentiment'] = df_eng['labels'].apply(lambda x: label_to_sentiment(x))


# In[ ]:


pd.set_option('max_colwidth', 200)
df_eng[['text','sentiment']].iloc[368:373] #Let's check some random tweets to see if the predicted sentiments make sense


# We can see that the classifier has managed to classify the sentiments quite well.

# In[ ]:


plt.figure(figsize = (15,7))
sns.barplot(x = df_eng['sentiment'].value_counts().index, y = df_eng['sentiment'].value_counts()/len(df_eng)*100)
plt.xlabel('Sentiment', fontsize = 20)
plt.ylabel('Percentage of Tweets(%)', fontsize = 20)
plt.xticks(fontsize = 15)
plt.title('Distribution of Tweets based on Sentiment', fontsize = 20)
plt.show()


# 75% of the tweets are Negative whereas only 10% of the tweets are Positive

# In[ ]:


from wordcloud import WordCloud
def plot_wordcloud(data):
    words = []
    for sent in data:
        for word in sent:
            words.append(word) 
    words = pd.Series(words).str.cat(sep=' ')
    wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(words)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# In[ ]:


plot_wordcloud(df_eng['text_clean'][df_eng['sentiment'] == 'Positive'])


# There some words which are common in both the positive and negative tweets like Coronavirus, Pandemic, Lockdown because they are the main subject of these tweets and are therefore in most of them.

# <b>Positive tweets:</b> These mostly contain words like 'help', 'social distancing',  'self reporting' which are related to spreading awareness about how the spread of virus can be slowed down by staying at home and containing the pandemic. Also people have shared different ways through which one can take care of their mental health like self reporting.

# In[ ]:


plot_wordcloud(df_eng['text_clean'][df_eng['sentiment'] == 'Negative'])


# <b>Negative tweets:</b> These tweets mostly contain words like 'business', 'work', 'government' which are related to how businesses are facing heavy losses during this lockdown and how this has caused many employees to be laid off as their companies are not able to make enough profits.

# # Hashtags Visualization

# Hashtags help to categorize any content. Therefore in a way they give a good idea about the theme and contents of a particular tweet.

# In[ ]:


import nltk


# In[ ]:


import re
def extract_hashtag(text):
    hashtags=[]
    for i in text:
        ht=re.findall(r'#(\w+)',i)
        hashtags.append(ht)
    return hashtags


# In[ ]:


all_hashtags=extract_hashtag(df_eng.text)
def df_hashtag(sentiment_label):
    hashtags=extract_hashtag(df_eng.text[df_eng['sentiment']==sentiment_label])
    ht_fredist=nltk.FreqDist(sum(hashtags,[]))
    df_ht=pd.DataFrame({'Hashtag':list(ht_fredist.keys()),'Count':list(ht_fredist.values())})
    return df_ht


# In[ ]:


#Hashtags dataframes
ht_neg_df=df_hashtag('Negative')
ht_neu_df=df_hashtag('Neutral')
ht_pos_df=df_hashtag('Positive')


# In[ ]:


ht_neg_df.to_pickle('ht_neg_df.pkl')
ht_neu_df.to_pickle('ht_neu_df.pkl')
ht_pos_df.to_pickle('ht_pos_df.pkl')


# In[ ]:


ht_neg_df = pd.read_pickle('pickle_files/ht_neg_df.pkl')
ht_neu_df = pd.read_pickle('pickle_files/ht_neu_df.pkl')
ht_pos_df = pd.read_pickle('pickle_files/ht_pos_df.pkl')


# In[ ]:


def plot_hashtag(df,title):
    data=df.nlargest(columns="Count",n=20)
    plt.figure(figsize=(16,5))
    ax=sns.barplot(data=data,x='Hashtag',y='Count')
    plt.suptitle(title, fontsize=20)
    plt.xlabel('Hashtag', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.xticks(rotation=90)
    plt.tick_params(labelsize=15)
    plt.show()


# In[ ]:


plot_hashtag(ht_pos_df,'Positive sentiments')


# In[ ]:


plot_hashtag(ht_neg_df,'Negative sentiments')


# In[ ]:


plot_hashtag(ht_neu_df,'Neutral sentiments')


# ## Sentiment-wise Likes ratio

# In[ ]:


plt.figure(figsize = (15,8))
df_eng.groupby(['sentiment'])['favourites_count'].mean().plot(color='red', linestyle='dashed', marker='o',
                                                            markerfacecolor='red', markersize=10)
plt.title('Sentiment-wise Likes ratio', fontsize = 20)
plt.xlabel('Sentiment', fontsize = 20)
plt.ylabel('Average Likes', fontsize = 20)
plt.xticks(fontsize = 15)
plt.show()


# On average Positive and Neutral tweets gets more likes than Negative tweets.

# # Sentiment-wise Retweets ratio

# In[ ]:


plt.figure(figsize = (15,8))
df_eng.groupby(['sentiment'])['retweet_count'].mean().plot(color='green', linestyle='dashed', marker='o',
                                                            markerfacecolor='g', markersize=10)
plt.title('Sentiment-wise Retweets ratio', fontsize = 20)
plt.xlabel('Sentiment', fontsize = 20)
plt.ylabel('Average Retweets', fontsize = 20)
plt.xticks(fontsize = 15)
plt.show()


# Negative tweets are retweeted the most on average followed by Positive tweets.

# ## Top 3 Most Liked Tweets

# In[ ]:


df_eng.sort_values(by = 'favourites_count', ascending = False).iloc[:3][['text','favourites_count']]


# ## Top 3 Most Retweeted Tweets

# In[ ]:


df_eng.sort_values(by = 'retweet_count', ascending = False).iloc[:3][['text','retweet_count']]


# # Time Series Analysis

# In[ ]:


df_eng['time'] = pd.to_datetime(df_eng['created_at'])


# In[ ]:


df_eng.groupby(['time'])['text'].count().plot(marker='.', alpha=0.5, figsize=(15, 5))
plt.xlabel('Time', fontsize = 20)
plt.ylabel('Tweet Count', fontsize = 20)
plt.xticks(fontsize = 12)
plt.title('Rate of Overall Tweets', fontsize = 20)
plt.show()


# On average people are active between 8:00am to 9:00pm

# In[ ]:


df_eng[df_eng['sentiment'] == 'Positive'].groupby(['time'])['text'].count().plot(marker='.', alpha=0.5, figsize=(15, 5),
                                                                                 color = 'g',markerfacecolor='g')
plt.xlabel('Time', fontsize = 20)
plt.ylabel('Tweet Count', fontsize = 20)
plt.xticks(fontsize = 12)
plt.title('Rate of Positive Tweets', fontsize = 20)
plt.show()


# Positive tweets seem to occur more frequently between 3:00pm to 8:00pm

# In[ ]:


df_eng[df_eng['sentiment'] == 'Negative'].groupby(['time'])['text'].count().plot(marker='.', alpha=0.5, figsize=(15, 5),
                                                                                 color = 'r',markerfacecolor='r')
plt.xlabel('Time', fontsize = 20)
plt.ylabel('Tweet Count', fontsize = 20)
plt.xticks(fontsize = 12)
plt.title('Rate of Negative Tweets', fontsize = 20)
plt.show()


# Negative tweets are more frequent between 7:00am to 11:00pm and they seem to occur at amuch higher rate than Positive tweets.

# # Topic Modelling using LDA

# <b>Topic Modelling</b> helps us to understand and summarise large collections of text. So far I have used <b>WordClouds</b> to manually look through the words and get a basic understanding of the different topics the tweets were related to. But there are much better statistical methods like <b>LDA</b> which can be used to discover <b>abstract topics</b> in a collection of documents.
# 
# <b>LDA</b> or <b>Latent Dirichlet Allocation</b> is used to classify text in a document to a particular topic. It backtracks and tries to figure out which topics would create the given documents. Then it assigns words to a given topic with some probability.

# In[ ]:


# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our corpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(docs)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]


# In[ ]:


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel


# In[ ]:


# Running and Trainign LDA model on the document term matrix.
ldamodel2 = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=10, iterations=10) 


# In[ ]:


for idx, topic in ldamodel2.show_topics(formatted=False, num_words= 30):
    print('Topic: {} \nWords: {}'.format(idx+1, '|'.join([w[0] for w in topic])))


# The LDA model has managed to classify the words into 5 abstract topics. On reading the individual words of a topic we can see that the words are not randomly assigned and that these words together have some meaning. 
# 
# The <b>first topic</b> contains words like <b>'worker', 'business', 'health', 'open'</b> which are related the losses faced by <b>businesses</b> and also the reaction to people regarding the <b>government's decisions</b>. The <b>second topic</b> is related to the <b>daily updates</b> on the number of cases that have increased along with the death rate and recovery rate. Some topics are not as easily interpretable like the <b>third topic</b>, it may seem a little vague at first compared to the other four topics but it seems like it is linked with how people are questioning Trump about the <b>current affairs</b>. The <b>fourth topic</b> is associated with <b>news</b> and the <b>fifth topic</b> with something more <b>positive</b> than the other topics for instance it seems to be linked with <b>spreading awareness</b> about social distancing, helping and supporting others.

# # Conclusion:

# This pandemic has been a psychological crisis. It has the potential to drastically affect the mental well-being of many. Therefore even in this time of social distancing it is important to understand what one is going through and provide them with all the necessary help and support they need. 

# Therefore the above analysis of the tweets is one way through which people and government can understand the mental and emotional state of a given population and aid them through these times.

# Thank You for taking some time and reading this notebook. Stay Home Stay Safe.
