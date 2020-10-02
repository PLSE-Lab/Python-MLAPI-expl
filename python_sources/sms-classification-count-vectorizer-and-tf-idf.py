#!/usr/bin/env python
# coding: utf-8

# # Classification of Text Messages

# ## Problem Statement

# The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

# We will be using this dataset to build a machine learning model to classify if a message is ham (Legitimate) or spam. We will be using some NLP techniques to preprocess the data and use seaborn to do some visualization.

# ## Importing the Libraries and the Dataset

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


email_data = pd.read_csv('../input/spam.csv', encoding = 'latin-1' )


# In[ ]:


email_data.head(20)


# Let's drop the columns with no values.

# In[ ]:


email_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis  = 1, inplace = True)


# In[ ]:


email_data.head(10)


# In[ ]:


email_data = email_data.rename(columns = {"v1": "label", "v2": "text"})


# In[ ]:


email_data.head()


# ## Visualizing the Dataset

# In[ ]:


sns.countplot(email_data['label'], label = "Count of the Labels")


# We can see that the dataset is not balanced. There are more data that are classified as ham other than spam. 

# Let's compute the length of the reviews to add as a new column.

# In[ ]:


email_data['length'] = email_data['text'].apply(len)


# In[ ]:


email_data.head()


# Taking a look at the distribution of the text lengths

# In[ ]:


email_data['length'].hist(bins = 50, color = 'g')


# From the histogram, we can see that most of the reviews are about have less than 50 words.

# ## Data Preprocessing / Data Cleaning
# 

# Before the data can be fed to a Machine Learning model,it is required that we clean the data first for the model to understand and process each review well.

# We will define a function that will remove all the punctuations and all the common words.

# In[ ]:


#Let's import the needed libraries
import string
from nltk.corpus import stopwords


# In[ ]:


def clean_review(review):
    remove_punctuation = [word for word in review if word not in string.punctuation]
    join_characters = ''.join(remove_punctuation)
    remove_stopwords = [word for word in join_characters.split() if word.lower() not in stopwords.words('english')]
    cleaned_review = remove_stopwords
    return cleaned_review


# Let's see if the function that we define is working.

# In[ ]:


#The original data
email_data['text'][4]


# In[ ]:


#Applying the cleaning function
clean_review(email_data['text'][4])


# We will use the function that we created later. 

# # Applying Count Vectorizer in the Dataset

# We will be applying count vectorizer to the dataset to convert the reviews into a matrix of token counts.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
#Define the clean_review function as an argument of the CountVectorizer.
count_vectorizer = CountVectorizer(analyzer = clean_review)
#Fit the countvectorizer to the dataset
email_countvec = count_vectorizer.fit_transform(email_data['text'])


# Taking a look at the Feature names

# In[ ]:


print(count_vectorizer.get_feature_names())


# Here's the shape of the new Vectorized Data.

# In[ ]:


email_countvec.shape


# From only 3 columns, we can see now that every word in the whole dataset have its own column.

# ## Training the Classification Model

# Before Initializing and Training the model. We will first split the data into training and testing data.

# In[ ]:


#Let's have a quick look at the data once again
email_data.head()


# Since we will be using the label column as our dependent variable. We need to encode it into binary variables.

# In[ ]:


#Using LabelEncoder from sklearn
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
email_data['label'] = le.fit_transform(email_data['label'])


# In[ ]:


#Assign X and y
X = email_countvec
y = email_data['label']


# ### Splitting the Data into Training and Testing

# We will be using 75% of the data for training and 25% for testing

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# ## Model Training

# We will be using Naive Bayes Classifier as our model.

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB() #Using the default parameters
classifier.fit(X_train, y_train)


# ## Evaluating the Model

# After training the Model we will proceed into evaluating its performance. Using confusion matrix and the classification report

# In[ ]:


#Making predictions to the Test Set
y_predictions = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_predictions)

#Taking a look at the Confusion Matrix with a Heatmap
sns.heatmap(cm, annot=True)


# In[ ]:


print(classification_report(y_test, y_predictions))


# ## The model actually produces a pretty good results with just using some of the basic techniques in NLP.

# # Using TF-IDF

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer()
tfidvec = tfid.fit_transform(email_data['text'])


# In[ ]:


tfidvec.shape


# In[ ]:


print(tfid.get_feature_names())


# In[ ]:


print(tfidvec[:,:])


# In[ ]:


#Assigning X2 and y2
X2 = tfidvec
y2 = email_data['label']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.25)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

classifier2 = MultinomialNB() #Using the default parameters
classifier2.fit(X_train2, y_train2)


# In[ ]:


#Making predictions to the Test Set
y_predictions2 = classifier2.predict(X_test2)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

cm2 = confusion_matrix(y_test2, y_predictions2)

#Taking a look at the Confusion Matrix with a Heatmap
sns.heatmap(cm2, annot=True)


# In[ ]:


print(classification_report(y_test2, y_predictions2))


# ### We can see that by using Term Frequency - Inverse Document Frequency, we can improve the performance of our model.
