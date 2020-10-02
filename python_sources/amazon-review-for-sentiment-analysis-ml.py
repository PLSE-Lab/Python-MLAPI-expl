#!/usr/bin/env python
# coding: utf-8

# ## Getting data from kaggle

# In[ ]:


# Run this cell and select the kaggle.json file downloaded
# from the Kaggle account settings page.

#from google.colab import files
#files.upload()  #Kaggle>account>API json file


# In[ ]:


# Let's make sure the kaggle.json file is present.
#!ls -lha kaggle.json


# In[ ]:


# Next, install the Kaggle API client.
#!pip install -q kaggle


# In[ ]:


# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.

#!mkdir -p ~/.kaggle
#!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.

#!chmod 600 ~/.kaggle/kaggle.json


# In[ ]:


# Copy the kaggle data set locally.
#!kaggle datasets download -d datafiniti/consumer-reviews-of-amazon-products


# In[ ]:


#! unzip -q -n 'consumer-reviews-of-amazon-products.zip'


# In[ ]:


#!ls


# ## If you are using notebook other then Kaggle (i.e. Jupyter or Colab) kindly execute the above code to import data to your notebook.

# In[ ]:


#Importing necessary libraries
import pandas as pd
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize


# In[ ]:


#Creating dataframe of amazon reviews from csv
df=pd.read_csv('../input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')
df.columns


# In[ ]:


#Shape of dataframe
df.shape


# In[ ]:


#Filtering Columns
df=df[['reviews.rating' , 'reviews.text' , 'reviews.title']]


# ## Inspection of Data

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


#Checking for null values
print(df.isnull().sum())


# In[ ]:


#ploting graph on the basis of review ratings
df["reviews.rating"].value_counts().sort_values().plot.bar()


# ### Above graph shows maximum reviews are positive.

# In[ ]:


#Rating value counts
df['reviews.rating'].value_counts()


# In[ ]:


sentiment = {1: 0,
            2: 0,
            3: 1,
            4: 2,
            5: 2}

df["sentiment"] = df["reviews.rating"].map(sentiment)


# In[ ]:


df.head()


# In[ ]:


#sentiment value counts
df.sentiment.value_counts()


# In[ ]:


#implementing bag of words using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(df['reviews.text'])


# In[ ]:


print(text_counts[0])


# In[ ]:


#Split train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_counts, df['reviews.rating'], test_size=0.3, random_state=1)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)


# In[ ]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
print("BoW MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# In[ ]:


#Using TF-IDF approach
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(df['reviews.text'])


# In[ ]:


#Split train and test set
X_train, X_test, y_train, y_test = train_test_split(text_tf, df['reviews.rating'], test_size=0.3, random_state=1)


# In[ ]:


#Again Generating model Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("TF-IDF MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# In[ ]:


# Random Forests
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier().fit(X_train, y_train)
predicted= classifier.predict(X_test)
print("Random Forests Accuracy:",metrics.accuracy_score(y_test, predicted))


# **Hence Random Forests gives Accuracy: 84.6, which is a good score**
