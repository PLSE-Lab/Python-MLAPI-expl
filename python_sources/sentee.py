#!/usr/bin/env python
# coding: utf-8

# # Amazon Alexa Reviews Sentiment Analysis using NLP
# <b>About the Data</b>
# 
# This dataset consists of a nearly 3000 Amazon customer reviews (input text), star ratings, date of review, variant and feedback of various amazon Alexa products like Alexa Echo, Echo dots, Alexa Firesticks etc. for learning how to train Machine for sentiment analysis.
# 
# <b>What has been done with the data ?</b></br>
# 
# This data has been used to analyse Amazon's Alexa product line-up, and perform sentiment analysis to understand why users like a certain product.
# </br>
# 
# <b>Source:</b>
# Extracted from Amazon's website
# 

# In[ ]:


#Importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


#reading the data
data=pd.read_csv('amazon_alexa.tsv', sep='\t')
data.info()


# In[ ]:


#Checking for missing data
data.isnull().any().any()


# In[ ]:


#Determine overall sentiment using histogram to plot feedback
overall_sentiment = data['feedback']
plt.hist(overall_sentiment, bins = 2)
plt.xlabel('Negative             Positive ')
plt.ylabel('Number of Reviews')
plt.show() 


# In[ ]:


data.feedback.value_counts()


# ##### 1: Positive Sentiment, 0: Negative Sentiment
# ##### The product users have an overwhelmingly positive sentiment associated to the Alexa.

# In[ ]:


data.groupby('variation').mean()[['rating']].plot.barh(figsize=(12, 7),colormap = 'ocean')
plt.title("Variation wise Mean Ratings");


# #### All of the Alexa Product variation's mean ratings are above 4 ( in a rating scale 1-5) with the Oak & Walnut finish rated slightly higher. 

# In[ ]:


Sentiment_count=data.groupby('rating').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['verified_reviews'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


# In[ ]:


data.rating.value_counts()


# In[ ]:


# adding a length column for analyzing the length of the reviews

data['length'] = data['verified_reviews'].apply(len)

data.groupby('length').describe().sample(5)


# In[ ]:


color = plt.cm.rainbow(np.linspace(0, 1, 15))
data['variation'].value_counts().plot.bar(color = color, figsize = (15, 9))
plt.title('Distribution of Variations in Alexa', fontsize = 20)
plt.xlabel('variations')
plt.ylabel('count')
plt.show()


# #### The variations plot above represents the number of different variants of the Amazon Alexa that have been reviewed. 

# In[ ]:


#finding which words occur the most
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english',tokenizer = token.tokenize)


# In[ ]:


words = cv.fit_transform(data.verified_reviews)
sum_words = words.sum(axis=0)


# In[ ]:


words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])


# In[ ]:


color = plt.cm.jet(np.linspace(0, 1, 20))
frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)
plt.title("Top 20 Most Frequently Occuring Words")
plt.show()


# <p> A word cloud is a novelty visual representation of text data, typically used to depict keyword metadata on websites, or to visualize free form text. The importance of each WORD is shown with font size or color.</p>

# In[ ]:


#todo:wordcloud
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white',width=800, height=500).generate_from_frequencies(dict(words_freq))
plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize=22);


# #### The words echo and love have been used the most; Followed by adjectives such as : Great,music,works,sound,easy.

# In[ ]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[ ]:


corpus = []
for i in range(0, 3150):
    review = re.sub('[^a-zA-Z]', ' ', data['verified_reviews'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(data['verified_reviews'])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, data['rating'], test_size=0.3, random_state=123)


# In[ ]:


#model building, model used: Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted)*100)


# In[ ]:


text_counts= cv.fit_transform(data['verified_reviews'])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['rating'], test_size=0.3, random_state=1)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted)*100)


# In[ ]:


predicted


# #### Our multinomial classifier model yields a 75% accuracy
