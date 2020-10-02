#!/usr/bin/env python
# coding: utf-8

# ### Importing Necessary Libraries
# 
# 

# In[ ]:


import spacy
from spacy import displacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import LinearSVC
import string
# Importing Dataset from the Github Repository.
get_ipython().system('git clone https://github.com/laxmimerit/NLP-Tutorial-8---Sentiment-Classification-using-SpaCy-for-IMDB-and-Amazon-Review-Dataset')


# ## Loading SpaCy's small english model

# To get more details regarding SpaCy models check here : https://spacy.io/usage/models

# In[ ]:


# Loading Spacy small model as nlp
nlp = spacy.load("en_core_web_sm")


# ## Gathering all the Stop words which does not convey much meaning in the Sentiment

# In[ ]:


# Gathering all the stopwords
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
print(len(stopwords))


# In[ ]:


# Loading yelp dataset
data_yelp = pd.read_csv('../working/NLP-Tutorial-8---Sentiment-Classification-using-SpaCy-for-IMDB-and-Amazon-Review-Dataset/datasets/yelp_labelled.txt',
                        sep='\t', header= None)
data_yelp.head()


# In[ ]:


# Adding column names to the dataframe
columnName = ['Review','Sentiment']
data_yelp.columns = columnName
data_yelp.head()


# ## So here we can deduce that Sentiment 1 is Positive and 0 is negative

# In[ ]:


print(data_yelp.shape)


# In[ ]:


# Adding Amazon dataset and adding its column name
data_amz = pd.read_csv("../working/NLP-Tutorial-8---Sentiment-Classification-using-SpaCy-for-IMDB-and-Amazon-Review-Dataset/datasets/amazon_cells_labelled.txt",
                        sep='\t', header= None)
data_amz.columns = columnName
data_amz.head()


# In[ ]:


print(data_amz.shape)


# In[ ]:


# Adding IMdB dataset and adding its column name
data_imdb = pd.read_csv("../working/NLP-Tutorial-8---Sentiment-Classification-using-SpaCy-for-IMDB-and-Amazon-Review-Dataset/datasets/imdb_labelled.txt",
                        sep='\t', header= None)
data_imdb.columns = columnName
data_imdb.head()


# In[ ]:


print(data_imdb.shape)


# ## Appending all the Datasets

# In[ ]:


# Merging all the three dataframes
data = data_yelp.append([data_amz, data_imdb], ignore_index=True)
print(data.shape)


# In[ ]:


# Sentiment ditribution in the dataset
data.Sentiment.value_counts()


# In[ ]:


# Getting information regarding the null entries in the dataset
data.isnull().sum()


# In[ ]:


punct = string.punctuation
print(punct)


# 
# 
# ```
# Here in the reviews we will find many stop words which do not add any meaning to the review.
# Also punctuations will be encountered in the review which which will be considered as a seperate token by our model
# So removing all the stop words and punctuation so that our model can train efficiently
# ```
# 
# 

# In[ ]:


def dataCleaning(sentence):
  doc = nlp(sentence)
  tokens = []
  for token in doc:
    if token.lemma_ != '-PRON-':
      temp = token.lemma_.lower().strip()
    else:
      temp = token.lower_
    tokens.append(temp)
  clean_tokens = []
  for token in tokens:
    if token not in punct and token not in stopwords:
      clean_tokens.append(token)
  return clean_tokens


# ## Here after passing a particular sentence in dataCleaning method we are returned with relevant words which contribute to the sentiments

# In[ ]:


dataCleaning("Today we are having heavy rainfall, We recommend you to stay at your home and be safe, Do not start running here and there")
# All the useful words are returned, no punctuations no stop words and in the lemmatized form


# In[ ]:


# Spillting the train and test data
X = data['Review']
y = data['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train.shape,y_test.shape)


# ## Preparing Model

# In[ ]:


# Creating the model and pipeline
tfidf = TfidfVectorizer(tokenizer = dataCleaning)
svm = LinearSVC()
steps = [('tfidf',tfidf),('svm',svm)]
pipe = Pipeline(steps)


# In[ ]:


# Training the model
pipe.fit(X_train,y_train)


# In[ ]:


# Testing on the test dataset
y_pred = pipe.predict(X_test)


# In[ ]:


# Printing the classification report and the confusion matrix
print(classification_report(y_test,y_pred))
print("\n\n")
print(confusion_matrix(y_test,y_pred))


# ## Testing on the Random Manual Examples

# **Here '1' represent that the input is positive sentiment**

# In[ ]:


# Testing on random inputs
pipe.predict(["Wow you are an amazing person"])


# **Here '0' represent that input is negative sentiment**

# In[ ]:


pipe.predict(["you suck"])


# ### Footnotes
# https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
# 
# https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976
# 
