#!/usr/bin/env python
# coding: utf-8

# # TASK #1: UNDERSTAND THE PROBLEM STATEMENT AND BUSINESS CASE

# 
# <table>
#   <tr><td>
#     <img src="https://drive.google.com/uc?id=11BquVVgQTebvVO5NZ2TGA526rulbWBv5"
#          alt="Fashion MNIST sprite"  width="1000">
#   </td></tr>
#   <tr><td align="center">
#     <b>Figure 1. Analyzing Customer Sentiment
#   </td></tr>
# </table>
# 

# ![alt text](https://drive.google.com/uc?id=1HfZvPCWAwKoYl1qogYlxD_CIZYxYw0aI)

# ![alt text](https://drive.google.com/uc?id=1XGc89Cxi0ooFQIc6o041cz8-qwXg7l3g)

# data source: https://www.kaggle.com/sid321axn/amazon-alexa-reviews/kernels

# # TASK #2: IMPORT LIBRARIES AND DATASETS

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# You have to include the full link to the csv file containing your dataset

reviews_df = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', sep='\t')


# In[ ]:


reviews_df.sample(3)


# In[ ]:


reviews_df.info()
reviews_df.describe()


# In[ ]:


reviews_df['verified_reviews']


# # TASK #3: EXPLORE DATASET

# In[ ]:


sns.heatmap(reviews_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[ ]:


reviews_df.hist(bins = 30, figsize = (13,5), color = 'r')


# In[ ]:


# Let's get the length of the messages

reviews_df['length'] = reviews_df['verified_reviews'].apply(len)
reviews_df.head()


# In[ ]:


reviews_df['length'].plot(bins=100, kind='hist') 


# In[ ]:


reviews_df.length.describe()


# In[ ]:


# Let's see the longest message 43952
reviews_df[reviews_df['length'] == 2851]['verified_reviews'].iloc[0]


# In[ ]:


# Let's see the shortest message 
reviews_df[reviews_df['length'] == 1]['verified_reviews'].iloc[0]


# In[ ]:


# Let's see the message with mean length 
reviews_df[reviews_df['length'] == 133]['verified_reviews'].iloc[0]


# In[ ]:


positive = reviews_df[reviews_df['feedback']==1]


# In[ ]:


negative = reviews_df[reviews_df['feedback']==0]


# In[ ]:


negative.sample(2)


# In[ ]:


positive.sample(2)


# In[ ]:


sns.countplot(reviews_df['feedback'], label = "Count") 


# In[ ]:


sns.countplot(x = 'rating', data = reviews_df)


# In[ ]:


plt.figure(figsize = (40,15))
sns.barplot(x = 'variation', y='rating', data = reviews_df, palette = 'deep')


# In[ ]:


sentences = reviews_df['verified_reviews'].tolist()
len(sentences)


# In[ ]:


print(sentences)


# In[ ]:


sentences_as_one_string =" ".join(sentences)


# In[ ]:


sentences_as_one_string


# In[ ]:


from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))


# In[ ]:


negative_list = negative['verified_reviews'].tolist()

negative_list


# In[ ]:


negative_sentences_as_one_string = " ".join(negative_list)


# In[ ]:



plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(negative_sentences_as_one_string))


# # TASK #4: PERFORM DATA CLEANING

# In[ ]:


# Let's drop the date
reviews_df = reviews_df.drop(['date', 'rating', 'length'],axis=1)


# In[ ]:


reviews_df.sample(3)


# In[ ]:


variation_dummies = pd.get_dummies(reviews_df['variation'], drop_first = True)
# Avoid Dummy Variable trap which occurs when one variable can be predicted from the other.


# In[ ]:


variation_dummies.head(3)


# In[ ]:


# first let's drop the column
reviews_df.drop(['variation'], axis=1, inplace=True)


# In[ ]:


# Now let's add the encoded column again
reviews_df = pd.concat([reviews_df, variation_dummies], axis=1)


# In[ ]:


reviews_df.sample(3)


# # TASK #5: LEARN HOW TO REMOVE PUNCTUATION FROM TEXT

# In[ ]:


import string
string.punctuation


# In[ ]:


Test = 'Hello Mr. Future, I am so happy to be learning AI now!!'


# In[ ]:


Test_punc_removed = [char for char in Test if char not in string.punctuation]


# In[ ]:


# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)


# # TASK 6: UNDERSTAND HOW TO REMOVE STOPWORDS

# In[ ]:


import nltk # Natural Language tool kit 

nltk.download('stopwords')


# In[ ]:


# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')


# In[ ]:


Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]


# In[ ]:


Test_punc_removed_join_clean # Only important (no so common) words are left


# # TASK 7: UNDERSTAND HOW TO PERFORM COUNT VECTORIZATION (TOKENIZATION)

# ![alt text](https://drive.google.com/uc?id=1eQi-Gq66e-sNw1ZvGs-zkJg95mCYdFoJ)

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)


# In[ ]:


print(vectorizer.get_feature_names())


# In[ ]:


print(X.toarray())  


# # TASK #8: PERFORM DATA CLEANING BY APPLYING EVERYTHING WE LEARNED SO FAR!

# In[ ]:


# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# In[ ]:


# Let's test the newly added function
reviews_df_clean = reviews_df['verified_reviews'].apply(message_cleaning)


# In[ ]:


print(reviews_df_clean[3]) # show the cleaned up version


# In[ ]:


print(reviews_df['verified_reviews'][3]) # show the original version


# In[ ]:


reviews_df_clean


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])


# In[ ]:


print(vectorizer.get_feature_names())


# In[ ]:


print(reviews_countvectorizer.toarray())  


# In[ ]:


reviews_countvectorizer.shape


# In[ ]:


# first let's drop the column
reviews_df.drop(['verified_reviews'], axis=1, inplace=True)
reviews = pd.DataFrame(reviews_countvectorizer.toarray())


# In[ ]:


# Now let's concatenate them together
reviews_df = pd.concat([reviews_df, reviews], axis=1)


# In[ ]:


reviews_df.head()


# In[ ]:


# Let's drop the target label coloumns
X = reviews_df.drop(['feedback'],axis=1)


# In[ ]:


y = reviews_df['feedback']


# # TASK #9: UNDERSTAND THE THEORY AND INTUITION BEHIND NAIVE BAYES

# ![alt text](https://drive.google.com/uc?id=1Xox54bvjhGOhrG-fSxEUIEgw1R3g-RIt)

# ![alt text](https://drive.google.com/uc?id=18Z4ug4UuyQG79lyPKs1zQwtrP_S4_yoU)

# ![alt text](https://drive.google.com/uc?id=1sVLtg8GaE3ZhNEZX1WJbxs7KAQyQ5dpX)

# ![alt text](https://drive.google.com/uc?id=1NT6Fm-lWUWNsu9i8uzVS4Q5pcm5gp8RK)

# ![alt text](https://drive.google.com/uc?id=1C32q5Uguymr9012x1lzRD5btnvJ-kW9r)

# ![alt text](https://drive.google.com/uc?id=1g5aXo5E-RIjRBy6-LLLA8gjG2j9dIL5X)

# ![alt text](https://drive.google.com/uc?id=106OXP_z89Hqh1JYVaROIbst0N0CgFRuT)

# ![alt text](https://drive.google.com/uc?id=1AXTHZ9KVUsJjMm9Whc4Adi5T4OznsSYn)

# # TASK #10: TRAIN A NAIVE BAYES CLASSIFIER MODEL

# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# # TASK #11: ASSESS TRAINED MODEL PERFORMANCE  

# ![alt text](https://drive.google.com/uc?id=14_ft6Wiu-VaiU_5Ew2nS7EGGr3oLLQf8)

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[ ]:


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[ ]:


print(classification_report(y_test, y_predict_test))


# # # TASK #12: - TRAIN AND EVALUATE A LOGISTIC REGRESSION CLASSIFIER

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

print('Accuracy {} %'.format( 100 * accuracy_score(y_pred, y_test)))


# In[ ]:


cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)


# In[ ]:


print(classification_report(y_test, y_pred))

