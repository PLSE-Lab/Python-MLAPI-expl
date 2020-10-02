#!/usr/bin/env python
# coding: utf-8

# # Amazon Alexa Reviews

# In[ ]:


# Importing the packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import seaborn as sns


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Loading the dataset (Here .tsv means tabs separated value)

df_reviews=pd.read_csv("/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv",sep='\t')
df_reviews.head()


# In[ ]:


df_reviews.shape             # Checking the number of rows and columns.


# In[ ]:


df_reviews.info()


# 
# # Observation:
#  There are no missing values in dataframe.
# 'Date' column should be in datetime type.

# In[ ]:


df_reviews.describe()                  # There are 2 numeric columns 'rating' and 'feedback'


# 
# # Observation :
# There are rating values from 1 to 5 and feedback from 0 to 1

# # Converting the date column

# In[ ]:


df_reviews['date']=pd.to_datetime(df_reviews['date'])


# In[ ]:


df_reviews['date'].dt.year.value_counts()


# # Observation:
# All the data is from the year 2018

# In[ ]:


df_reviews['date'].min()


# In[ ]:


df_reviews['date'].max()


# # Observation:
# The data is provided for days between 16th May and 31st July.

# # Visualizations

# In[ ]:


# Lets have a look at the time series plot.
plt.figure(figsize=(15,6))
sns.countplot(x='date',data=df_reviews)
plt.xticks(rotation=90)
plt.show();


# In[ ]:


# Lets look at the purchases on each of the three months

sns.countplot(df_reviews['date'].dt.month)


# In[ ]:


# This is the count of products sold or the reviews recieved in each month.

df_reviews['date'].dt.month.value_counts()


# 
# There are a lot of reviews obtained for the month of July. This can also mean that there was a lot of devices purchases on the month of July.

# In[ ]:


sns.countplot(x='rating',data=df_reviews)


# # Observation:
# There are a lot of good reviews .Lets have a look at their numbers.

# In[ ]:


df_reviews.rating.value_counts()


# In[ ]:


sns.countplot(x='feedback',data=df_reviews)


# # Observation:
# Alot of people have given positive feedback.

# In[ ]:


# Now lets have a look at the word count of the reviews.

df_reviews['length']=df_reviews['verified_reviews'].apply(lambda x:len(x.split(" ")))
df_reviews.head()


# In[ ]:


# Lets have a look at the histogram of the length of reviews.
plt.hist(x='length',data=df_reviews,bins=30);


# 
# # Observation:
# There are a lot of people who have given short reviews.Very few reviews are long.

# In[ ]:


df_reviews.length.describe()


# 
# # Observation:
# The median number of words are 14 and the mean of the words are 25.

# In[ ]:


# Is there a relation between review word count and the rating
sns.lmplot(x='length',y='rating',data=df_reviews)


# 
# People who have left poor reviews have written less number of words,but the people who have written good reviews have written a lot of words.

# # Lets have a look at the negative feedbacks.

# In[ ]:


neg=df_reviews[df_reviews['feedback']== 0]


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
text = neg['verified_reviews'].values
wordcloud = WordCloud(
width=3000,
height=2000,
background_color = 'black',
stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
figsize=(40,30),
facecolor = 'k',
edgecolor = 'k')
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# **Apart from "Amazon ,device,Alexa,work",we find negative words such as problem,disappointed,speaker,sound.These are not words which make the feedback negative.**

# You can find other words which would be a reason ,why the feedback was bad?

# In[ ]:


sns.countplot(x='rating',data=neg)


# 
# All the negative feedback are 1 or 2 ratings.

# # Sentiment Analysis
# 

# In[ ]:


N = df_reviews.shape[0]
N


# In[ ]:


corpus = []

import re
import nltk                                                 # For text preprocessing we use nltk
from nltk.corpus import stopwords                           # Removing all the stopwords
from nltk.stem.porter import PorterStemmer                  # Reducing words to base form


# 
# **Stowords: Words which are filtered before applying NLP as they have no meaning.
# E.g . a,again,am,are,but,etc....
# Porter Stemmer is used to remove any morphological affixes from words,leaving ony the word stem.
# E.g. running--> run
# having --> have

# In[ ]:


nltk.set_proxy('SYSTEM PROXY')

##nltk.download()
nltk.download('stopwords')


# In[ ]:


ps=PorterStemmer()

for i in range (0,N):
    review = re.sub('[^a-zA-Z ]',' ',df_reviews['verified_reviews'][i]) # Removing special symbols like...,! and kkeeping only 
    review = review.lower()                                             # Lower case
    review = review.split()                                             # String split into words
    review = [ps.stem(word) for word in review                         # Reducing words to base form
    if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)


# In[ ]:


corpus


# 
# # Applying TF-IDF
# 
# Tf-idf is an NLP technique to weight words how important they are. How do we calculate the importance of words?
# Words that are used frequently in many documents will have a lower weighting while infrequent ones will have a higher weighting.
# The Tf-idf value increases proportionally to the number of times a word appears in the document,but is offset by the frequency of helps to adjust for the fact that some words appear more frequently in general.
# We calculate the item frequency and inverse document frequency by the following formula.
# Term Frequency: TF(t) = (Number of times term t appears in a document)/(Total number of terms in the document)
# Inverse Document Frequency: IDF(t) =log_e(Total number of documents/Number of documents with term t in it).
# tf-idf score=TF(t)*IDF(t)

# In[ ]:


#TF-IDF Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(use_idf=True, strip_accents='ascii')


# # separating Labels and Features

# In[ ]:


y = df_reviews['feedback']


# In[ ]:


X = vectorizer.fit_transform(corpus)


# In[ ]:


X.shape


# In[ ]:


y.shape


# # Train Test Split

# In[ ]:


# Split the test and train

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42, test_size=0.2)


# # Building the classifier

# # Decission Tree

# In[ ]:


# Importing the decission tree classifier

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(class_weight='balanced')


# In[ ]:


dt_clf.fit(X_train, y_train)               # Model fitting on X_train,y_train
y_pred = dt_clf.predict(X_test)            # Model prediction on X_test


# In[ ]:


# Split the test and train
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


# Classification Report
cr = classification_report(y_test ,y_pred)
print(cr)


# In[ ]:


# Accuracy Score
acc=accuracy_score(y_test,y_pred)
print(acc)


# # Techniques  will explore here as:
# Loading Data, Data cleaning and Data visualization using Python
# End to End EDA using Statistics
# Predictive Modelling of Customer Sentiment Reviews using Decision trees
