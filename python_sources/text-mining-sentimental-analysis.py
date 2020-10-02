#!/usr/bin/env python
# coding: utf-8

# # Movie Reviews - Rotten Tomatoes
# The dataset is comprised of tab-separated files with phrases from the **Rotten Tomatoes dataset**. The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. Each Sentence has been parsed into many phrases by the Stanford parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data.
# 
# train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.
# 
# test.tsv contains just phrases. You must assign a sentiment label to each phrase.
# 
# The sentiment labels are:
# 
# 0 - negative,1 - somewhat negative,2 - neutral,3 - somewhat positive, 4 - positive

# ## Task and Approach:
# * Like IMDb , Rotten Tomatoes is also one of the most popular website where people can find reviews and ratings for nearly any movie.            
# * We need to **predict the sentiment based on the each phrase** by training the clasification model .  

# # Loading important Libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# To make data visualisations display in Jupyter Notebooks 
import numpy as np   # linear algebra
import pandas as pd  # Data processing, Input & Output load

import matplotlib.pyplot as plt # Visuvalization & plotting
import seaborn as sns  #Data visualisation

import nltk # Natural Language Toolkit (statistical natural language processing (NLP) libraries )
from nltk.stem.porter import *   # Stemming 

from sklearn.model_selection import train_test_split, cross_val_score
                                    # train_test_split - Split arrays or matrices into random train and test subsets
                                    # cross_val_score - Evaluate a score by cross-validation

from sklearn.ensemble import RandomForestClassifier # RandomForestClassifier model to predict sentiment

from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer converts collection of text docs to a matrix of token counts

from sklearn.feature_extraction.text import TfidfTransformer # Converting occurrences to frequencies
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings   # To avoid warning messages in the code run
warnings.filterwarnings("ignore")


# ** Loading Data**

# In[2]:


train_MR = pd.read_csv("../input/train.tsv",sep="\t") # Train Moview Reviews 
test_MR = pd.read_csv("../input/test.tsv",sep="\t")


# **Describe the data**
# 

# In[3]:


print(train_MR.shape)
print(test_MR.shape)
#train_MR.head()
test_MR.head()


# In[4]:


train_MR.describe()


# * We can see from the mean sentiment showing as nearly 2 and max is 4 .
# 
# * Same thing we can visually see this using plots

# In[5]:


sns.countplot(data=train_MR,x='Sentiment')


# * Here We can see the distribution of the counts 

# In[6]:


dist = train_MR.groupby(["Sentiment"]).size()
print(dist)

dist_Percentage = round((dist / dist.sum())*100,2)
print(dist_Percentage)


# In[7]:


train_MR['Length'] = train_MR['Phrase'].apply(lambda x: len(str(x).split(' ')))   ## WIll get the length of each phrase 
test_MR['Length'] = test_MR['Phrase'].apply(lambda x: len(str(x).split(' '))) 

train_MR.head()


# *  **Checking the null values**

# In[8]:


train_MR.isnull().sum() 
test_MR.isnull().sum() 


# #### PreProcessing
# * Remove special characters, numbers, punctuations

# In[9]:


train_MR['PreProcess_Sentence'] = train_MR['Phrase'].str.replace("[^a-zA-Z#]", " ")
test_MR['PreProcess_Sentence'] = test_MR['Phrase'].str.replace("[^a-zA-Z#]", " ")
train_MR.head()


# #### Converting all lower letters

# In[10]:


train_MR['PreProcess_Sentence'] = train_MR['PreProcess_Sentence'].str.lower()
test_MR['PreProcess_Sentence'] = test_MR['PreProcess_Sentence'].str.lower()
test_MR['PreProcess_Sentence'].head()


# Tokens are usually individual words and "tokenization" is taking a text or set of text and breaking it up into its individual words
# 
# here in **vectorizer** Tokenizing text with scikit-learn so we are not applying extrnally for our phrases

# * CountVectorizer converts collection of text docs to a matrix of token counts

# In[11]:


count_vector = CountVectorizer()
train_counts = count_vector.fit_transform(train_MR['PreProcess_Sentence'])
train_counts.shape


# Preprocess sentence of traindata is converetd to 156060 X 15100 matrix 

# In[12]:


count_vector.get_feature_names()


# * Get index of some common words 

# In[13]:


count_vector.vocabulary_.get('abdul')


# #### TF-IDF usage 
# *  *Countvectorizer* gives equal weightage to all the words, i.e. a word is converted to a column
# * Tf idf is different from countvectorizer.                    
# * vocabulary_ just gives a dict of indexes of the words.

# #### TF

# It increases the weight of the terms (words) that occur more frequently in the document
# 

# In[14]:


## Term Frequencies (tf)
tf_transformer = TfidfTransformer(use_idf = False).fit(train_counts)  # Use fit() method to fit estimator to the data
train_tf = tf_transformer.transform(train_counts) # Use transform() method to transform count-matrix to 'tf' representation


# #### IDF

# It diminishes the weight of the terms that occur in all the documents of corpus and similarly increases the weight of the terms that occur in rare documents across the corpus.

# In[15]:


## Term Frequency times Inverse Document Frequency (tf-idf)
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts) # Use transform() method to transform count-matrix to 'tf-idf' representation


# ### Model
# * We can try to predict now by keeping the most words and their respective Sentiment scores

# In[16]:


## Training a classifier to predict sentiment label of a phrase
# RandomForestClassifier model to predict sentiment

model = RandomForestClassifier()
Final_Model = model.fit(train_tfidf, train_MR['Sentiment'])


# #### Tokenizing test phrase

# In[17]:


test_counts = count_vector.transform(test_MR['PreProcess_Sentence'])
# Use transform() method to transform test count-matrix to 'tf-idf' representation
test_tfidf = tfidf_transformer.transform(test_counts)
test_tfidf.shape


# In[18]:


## Prediction on test data
predicted = Final_Model.predict(test_tfidf)


# In[19]:


#for i, j in zip(testdata['PhraseId'], predicted):print(i, predicted[j])

for i, j in zip(test_MR['PhraseId'], predicted):
    print(i, predicted[j])
    
#testdata.head()
#testdata['PhraseId']
#predicted


# In[21]:


# Writing *csv file for submission
import csv
with open('Movie_Sentiment.csv', 'w') as csvfile:
    csvfile.write('PhraseId,Sentiment\n')
    for i, j in zip(test_MR['PhraseId'], predicted):
         csvfile.write('{}, {}\n'.format(i, j))


# # END OF NOTEBOOK
