#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# pd.set_option('display.max_colwidth', -1)
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
from wordcloud import WordCloud
from collections import Counter
import csv
from matplotlib import rcParams
from nltk.corpus import stopwords
import nltk
from nltk.util import ngrams
stop = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,plot_confusion_matrix
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Reading the CSV Files

# In[ ]:


true = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
false = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")
true.head()


# In[ ]:


false.head()


# In[ ]:


true.subject.value_counts()


# # Exploratory data analysis

# In[ ]:


rcParams['figure.figsize'] = 15,10
true.subject.value_counts().plot(kind="bar")


# ## The above Viz shows that target column is equally distributed in true category

# In[ ]:


rcParams['figure.figsize'] = 15,10
false.subject.value_counts().plot(kind="bar")


# ## The above Viz shows that target column is not  equally distributed in False category and News label is more than other labels

# ## Sepearting the dataset into the different dataframe based on the label column

# In[ ]:


politics = true[true['subject']=="politicsNews"]
worldnews = true[true['subject']=="worldnews"]
print(politics.shape)
print(worldnews.shape)


# In[ ]:


politics_text_len = politics['text'].str.len()
worldnews_text_len = worldnews['text'].str.len()


# In[ ]:


print("The maximum lenght of string in Politcs news is {} words".format(max(politics_text_len)))
print("The maximum lenght of string in World news is {} words".format(max(worldnews_text_len)))


# ### since i cannot able to plot this i have just printed the maximum lenght of strin[](http://)g value in the each labels

# # Simple Pre-Processing on politics and world news dataset - True Tweets

# >Tokenization 
# >Stop words removal

# ## Toekenization
# >In Python tokenization basically refers to splitting up a larger body of text into smaller lines, words or even creating words for a non-English language.Here we are performing the word tokenization from NLTK Library

# ## Stopwords
# >stop words are words which are filtered out before or after processing of natural language data (text).[1] Though "stop words" usually refers to the most common words in a language, there is no single universal list of stop words used by all natural language processing tools, and indeed not all tools even use such a list

# In[ ]:


def tokenizeandstopwords(text):
    tokens = nltk.word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    meaningful_words = [w for w in token_words if not w in stop]
    joined_words = ( " ".join(meaningful_words))
    return joined_words


# In[ ]:


politics['text'] = politics['text'].apply(tokenizeandstopwords)
worldnews['text'] = worldnews['text'].apply(tokenizeandstopwords)


# ## WordCloud
# > A tag cloud (word cloud or wordle or weighted list in visual design) is a novelty visual representation of text data, typically used to depict keyword metadata (tags) on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.[2] This format is useful for quickly perceiving the most prominent terms to determine its relative prominence. When used as website navigation aids, the terms are hyperlinked to items associated with the tag

# # Defining the word Cloud function to generate the word cloud

# In[ ]:


def generate_word_cloud(text):
    wordcloud = WordCloud(
        width = 3000,
        height = 2000,
        background_color = 'black').generate(str(text))
    fig = plt.figure(
        figsize = (40, 30),
        facecolor = 'k',
        edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


# # World Cloud form true News Dataset

# ## Word Cloud for Politics Label

# In[ ]:


politics_text = politics.text.values
generate_word_cloud(politics_text)


# ## Word Cloud for Worldnews Label

# In[ ]:


worldnews_text = worldnews.text.values
generate_word_cloud(worldnews_text)


# # False News Dataset Analysis

# In[ ]:


false.head()


# In[ ]:


set(false.subject)


# ## Seperating the dataset into the different dataframe based on the labels

# In[ ]:


Government_News = false[false['subject']=="Government News"]
Middle_east = false[false['subject']=="Middle-east"]
News = false[false['subject']=="News"]
US_News = false[false['subject']=="US_News"]
politics = false[false['subject']=="politics"]


# In[ ]:


Government_News['text'] = Government_News['text'].apply(tokenizeandstopwords)
Middle_east['text'] = Middle_east['text'].apply(tokenizeandstopwords)
News['text'] = News['text'].apply(tokenizeandstopwords)
US_News['text'] = US_News['text'].apply(tokenizeandstopwords)
politics['text'] = politics['text'].apply(tokenizeandstopwords)


# # Word Cloud for Fake news

# ## Word Cloud for Goverment news Label

# In[ ]:


govertment_news_text = Government_News['text'].values
generate_word_cloud(govertment_news_text)


# ## Word Cloud for Middle east news Label

# In[ ]:


middleast_news_text = Middle_east['text'].values
generate_word_cloud(middleast_news_text)


# ## Word Cloud for General News Label

# In[ ]:


news_text = News['text'].values
generate_word_cloud(news_text)


# ## Word Cloud for Us News Label

# In[ ]:


usnews_text = US_News['text'].values
generate_word_cloud(usnews_text)


# ## Word Cloud for Politics Label in Fake dataset

# In[ ]:


politicsFake_text = politics['text'].values
generate_word_cloud(politicsFake_text)


# # Merging true and fake news dataset

# In[ ]:


false['target'] = 'fake'
true['target'] = 'true'
news = pd.concat([false, true]).reset_index(drop = True)
news.head()


# In[ ]:


news.shape


# In[ ]:


news['text'] = news['text'].apply((lambda y:re.sub("http://\S+"," ", y)))
news['text'] = news['text'].apply((lambda x:re.sub("\@", " ",x.lower())))


# In[ ]:


news.head()


# In[ ]:


def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]


# In[ ]:


true_word = basic_clean(''.join(str(true['text'].tolist())))


# # N-Gram

# >In the fields of computational linguistics and probability, an n-gram is a contiguous sequence of n items from a given sample of text or speech. The items can be phonemes, syllables, letters, words or base pairs according to the application. The n-grams typically are collected from a text or speech corpus. When the items are words, n-grams may also be called shingles

# # N-gram Analysis - Bigram and Trigram 

# # N-gram for true news

# In[ ]:


true_bigrams_series = (pd.Series(nltk.ngrams(true_word, 2)).value_counts())[:20]


# ## True News - Bigram

# In[ ]:


true_bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Bigrams')
plt.ylabel('Bigram')
plt.xlabel('# of Occurances')


# ## True News - Trigram

# In[ ]:


true_trigrams_series = (pd.Series(nltk.ngrams(true_word, 3)).value_counts())[:20]
true_trigrams_series.sort_values().plot.barh(color='red', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Trigrams')
plt.ylabel('Trigram')
plt.xlabel('# of Occurances')


# # N-Gram -False word Analysis

# In[ ]:


false_word = basic_clean(''.join(str(false['text'].tolist())))


# In[ ]:


flase_bigrams_series = (pd.Series(nltk.ngrams(false_word, 2)).value_counts())[:20]


# ## False News - Bigram

# In[ ]:


flase_bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Bigrams')
plt.ylabel('Bigram')
plt.xlabel('# of Occurances')


# ## False News - Trigram

# In[ ]:


false_trigrams_series = (pd.Series(nltk.ngrams(false_word, 3)).value_counts())[:20]
false_trigrams_series.sort_values().plot.barh(color='red', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Trigrams')
plt.ylabel('Trigram')
plt.xlabel('# of Occurances')


# # Full Dataset Analysis

# In[ ]:


words = basic_clean(''.join(str(news['text'].tolist())))


# ## Full Data - Bigram

# In[ ]:


bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:20]


# In[ ]:


bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Bigrams')
plt.ylabel('Bigram')
plt.xlabel('# of Occurances')


# ## Full Data - Trigram

# In[ ]:


trigrams_series = (pd.Series(nltk.ngrams(words, 3)).value_counts())[:20]


# In[ ]:


trigrams_series.sort_values().plot.barh(color='red', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Trigrams')
plt.ylabel('Trigram')
plt.xlabel('# of Occurances')


# # Building a Basic Model

# In[ ]:



x_train,x_test,y_train,y_test = train_test_split(news['text'], news.target, test_size=0.2, random_state=2020)

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[ ]:


print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))


# In[ ]:


plot_confusion_matrix(model,x_test,y_test)


# In[ ]:




