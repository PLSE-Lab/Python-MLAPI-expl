#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np




# In[ ]:


#for text analysis
# fro plotting
import matplotlib.pyplot as plt
import seaborn as sns
# import natural language processing packages
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize

#spacy-> advances nltk package
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()


# In[ ]:



df =pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis =1,inplace=True)
df.columns=['category','text']


# In[ ]:


# to check for any null or empty files
df.info()


# In[ ]:


# describing the dataset into spam and ham
df['category'].value_counts().plot(kind = 'pie',explode=[0,0.1],autopct='%.2f')
plt.xlabel('Spam vs Ham')
plt.legend(["ham","Spam"])
plt.show()


# In[ ]:


display(df.head(n=20))


# while spliiting the data using train_test_split we have to be careful as our training may contain more normal messages . if it contains more normal messages then it may predict most as ham and we may not know too
# 
# 

# 

#  so we have to use accuracy metric that remembers this . we can live by missing some of spam message but we don't want to mark ham as spam.
#  so we will be using fbeta accuracy as it uses weighted harmonic mean of precision and recall
#  

# 

# In[ ]:


# some of the top messages 
topM= df.groupby('text')['category'].agg([len,np.max]).sort_values(by = 'len',ascending =False).head(n=5)
print(topM)


# In[ ]:


# studying individual spam words and hand words by grouping them
spam_messages = df[df['category']=='spam']['text']
print(spam_messages)
ham_messages =df[df['category']=='ham']['text']


# In[ ]:


# separating major spam and ham words
spam_words = []
ham_words = []


# In[ ]:


def extractSpam(spam_messages):
    global spam_words
    words =[word.lower() for word in word_tokenize(spam_messages) if word.lower() not in stopwords.words('english')and word.lower().isalpha()]
    spam_words.append(words)
def extractHam(ham_messages):
    global ham_words
    words =[word.lower() for word in word_tokenize(ham_messages) if word.lower() not in stopwords.words('english')and word.lower().isalpha()]
    ham_words.append(words)

spam_messages.apply(extractSpam)
ham_messages.apply(extractHam)
    


# In[ ]:


#visually representing data
#converting list to string  using list comprehension for the safe side
# as WordCloud expects str instance
spam_list = ' '.join([str(elem) for elem in spam_words])
ham_list = ' '.join([str(elem) for elem in ham_words])

from wordcloud import WordCloud as WC
spam_wc = WC(width =600,height =400).generate(spam_list)
plt.figure(figsize = (10,8),facecolor = 'k')  # k means black
plt.imshow(spam_wc)# to display as image
plt.show()


# In[ ]:


ham_wc = WC(width = 600,height = 300).generate(ham_list)
plt.figure(figsize =(10,8),facecolor = 'k')
plt.imshow(ham_wc)
plt.show()


# In[ ]:


#cleaning data to be used in algorithm 
#removing stopwords, punctuations and stemmed words
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')


# In[ ]:


#cleaning text messages
def clean_text(df):
    df = df.translate(str.maketrans('','',string.punctuation))
    words = [stemmer.stem(word) for word in df.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(words)
df['text'] = df['text'].apply(clean_text) # passsing values to functions one by one
df.head(n=10)


# In[ ]:


#converting data into a form that machine learning algorithm can make sense of 
#using COunt vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
features = cv.fit_transform(df['text'])
print(features.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, df['category'], test_size=0.33)


# for splitting the test set and training set
# 

# In[ ]:


#naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
Multinb = MultinomialNB()
Multinb.fit(X_train,y_train)
Multinb.score(X_test,y_test)
y_predict=Multinb.predict(X_test)
print(classification_report(y_test,y_predict))


# In[ ]:





# In[ ]:




