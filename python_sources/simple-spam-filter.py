#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# In this notebook I will build a simple spam filter using naive bayes & labelled dataset of email obtained from [kaggle](https://www.kaggle.com/omkarpathak27/identify-spam-using-emails/data). I will also explain various preprocessing steps involved for text data followed by feature extraction & calssification model.

# In[1]:


import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Exploratory Analysis

# In[2]:


emails = pd.read_csv('../input/emails.csv')


# In[3]:


emails.head()


# In[4]:


#Lets read a single email 

emails.get_value(58,'text')


# In[5]:


emails.shape
#Total 5728 emails


# In[6]:


#Checking class distribution
emails.groupby('spam').count()
#23.88% emails are spam which seems good enough for our task


# In[7]:


#Lets see the distribution of spam using beautiful seaborn package

label_counts = emails.spam.value_counts()
plt.figure(figsize = (12,6))
sns.barplot(label_counts.index, label_counts.values, alpha = 0.9)

plt.xticks(rotation = 'vertical')
plt.xlabel('Spam', fontsize =12)
plt.ylabel('Counts', fontsize = 12)
plt.show()


# In[8]:


#Lets check if email length is coorelated to spam/ham
emails['length'] = emails['text'].map(lambda text: len(text))

emails.groupby('spam').length.describe()


# In[9]:


#emails length have some extreme outliers, lets set a length threshold & check length distribution
emails_subset = emails[emails.length < 1800]
emails_subset.hist(column='length', by='spam', bins=50)

#Nothing much here, lets process the contents of mail now for building spam filter


# ### Text Data Preprocessing
# Mails provided in data are full of unstuctured mess, so its important to preprocess this text before feature extraction & modelling. Thanks to [nltk](https://www.nltk.org/) library, its very easy to do this preprocessing now  with few lines of python code.

# #### Tokenization
# Tokenization converts continuous stream of words into seprate token for each word.

# In[10]:


emails['tokens'] = emails['text'].map(lambda text:  nltk.tokenize.word_tokenize(text)) 


# In[11]:


#Lets check tokenized text from first email

print(emails['tokens'][1])


# #### Stop Words Removal
# Stop words usually refers to the most common words in a language like 'the', 'a', 'as' etc. These words usually do not convey any useful information needed for spam filter so lets remove them.

# In[12]:


#Removing stop words

stop_words = set(nltk.corpus.stopwords.words('english'))
emails['filtered_text'] = emails['tokens'].map(lambda tokens: [w for w in tokens if not w in stop_words]) 


# In[13]:


#Every mail starts with 'Subject :' lets remove this from each mail 

emails['filtered_text'] = emails['filtered_text'].map(lambda text: text[2:])


# In[14]:


#Lets compare an email with stop words removed

print(emails['tokens'][3],end='\n\n')
print(emails['filtered_text'][3])

#many stop words like 'the', 'of' etc. were removed


# In[15]:


#Mails still have many special charater tokens which may not be relevant for spam filter, lets remove these
#Joining all tokens together in a string
emails['filtered_text'] = emails['filtered_text'].map(lambda text: ' '.join(text))

#removing apecial characters from each mail 
emails['filtered_text'] = emails['filtered_text'].map(lambda text: re.sub('[^A-Za-z0-9]+', ' ', text))


# #### Lemmatization
# Its the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form. so word like 'moved' & 'moving' will be reduced to 'move'. 

# In[16]:


wnl = nltk.WordNetLemmatizer()
emails['filtered_text'] = emails['filtered_text'].map(lambda text: wnl.lemmatize(text))


# In[29]:


#Lets check one of the mail again after all these preprocessing steps
emails['filtered_text'][4]


# In[18]:


#Wordcloud of spam mails
spam_words = ''.join(list(emails[emails['spam']==1]['filtered_text']))
spam_wordclod = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wordclod)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[19]:


#Wordcloud of non-spam mails
spam_words = ''.join(list(emails[emails['spam']==0]['filtered_text']))
spam_wordclod = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wordclod)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# ### Spam Filtering Models
# After preprocessing we have clean enough text, lets convert these mails into vectors of numbers using 2 popular methods: [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) & [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). After getting vectors for each mail we will build our classifier using [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).

# ### 1. Bag of Words
# It basically creates a vector with frequency of each word from vocabulary in given mail. Like name suggests bag of words does not treat text as a sequence but a collection of unrelated bag of words. Its easy to create these vectors using CountVectorizer() from scikit learn.
# 

# In[20]:


count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(emails['filtered_text'].values)


# In[21]:


print(counts.shape)


# ### Naive Bayes Classifier

# In[22]:


classifier = MultinomialNB()
targets = emails['spam'].values
classifier.fit(counts, targets)


# In[23]:


#Predictions on sample text
examples = ['cheap Viagra', "Forwarding you minutes of meeting"]
example_counts = count_vectorizer.transform(examples)
predictions = classifier.predict(example_counts)


# In[30]:


print(predictions)


# ### 2. TF-IDF
# tf-idf is a numerical statistic that is intended to reflect how important a word is to a mail in collection of all mails or corpus. This is also a vector with tf-idf values of each word for each mail. To understsnd how tf-fdf values are computed please check my [blog post](https://mohitatgithub.github.io/2018-04-28-Learning-tf-idf-with-tidytext/) on understanding tf-idf. Here we will use TfidfTransformer() from scikit learn to generate this vector.

# In[24]:


tfidf_vectorizer = TfidfTransformer().fit(counts)
tfidf = tfidf_vectorizer.transform(counts)


# In[25]:


print(tfidf.shape)


# In[26]:


classifier = MultinomialNB()
targets = emails['spam'].values
classifier.fit(counts, targets)


# In[27]:


#Predictions on sample text
examples = ['Free Offer Buy now',"Lottery from Nigeria","Please send the files"]
example_counts = count_vectorizer.transform(examples)
example_tfidf = tfidf_vectorizer.transform(example_counts)
predictions_tfidf = classifier.predict(example_tfidf)


# In[28]:


print(predictions_tfidf)


# Future Scope: 
# 1. Steps discussed above for feature vectorization & model building can also be stacked together using [scikit learn pipelines](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). 
# 2. I have not used a train test split here, we can throughly evaluate our model with a seprate text set & further using cross-validation.
# 3. Text data can be further processed & new features can be used to build more robust filters using other techniques like N-grams. We can also try other machine learning algortihms like SVM, KNN etc.

# Thanks for reading, I am a novice in text analysis so please share your feedback on improvements & errors.
