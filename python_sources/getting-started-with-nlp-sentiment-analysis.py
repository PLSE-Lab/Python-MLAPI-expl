#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing

# ## Sentiment Analysis

# **----: Problem Statement :----**
# - **About Practice Problem: Identify the Sentiments**
#    - Sentiment analysis is contextual mining of text which identifies and extracts subjective information in source material, and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations. Brands can use this data to measure the success of their products in an objective manner. In this challenge, you are provided with tweet data to predict sentiment on electronic products of netizens.
# 
#   - Sentiment analysis remains one of the key problems that has seen extensive application of natural language processing. This time around, given the tweets from customers about various tech firms who manufacture and sell mobiles, computers, laptops, etc, the task is to identify if the tweets have a negative sentiment towards such companies or products.

# #### ----: Importing necessary liabraries :----

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
import re 
pd.set_option("display.max_colwidth",200)
import nltk
import string
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import KFold, cross_val_score
plt.style.use('seaborn')


# **----: Importing Word Net Lemmatizer and stopwords :----**

# In[ ]:


wn = nltk.WordNetLemmatizer()
stopwords=nltk.corpus.stopwords.words("english")


# #### ----: Reading Datasets :----

# In[ ]:


# training dataset
train_dataset=pd.read_csv("../input/train_sentiment.csv") 
#training labels
label=pd.read_csv("../input/train_sentiment.csv") 
# testing dataset
test_dataset=pd.read_csv("../input/test_sentiment.csv")


# **----: Exploring Datasets :----**

# In[ ]:


# printing the size of the dataset
print("train_dataset--->",train_dataset.shape)
print("test_dataset--->",test_dataset.shape)


# In[ ]:


train_dataset=train_dataset.drop("label",axis=1)
## convertinig each text into lower case
train_dataset["tweet"]=train_dataset["tweet"].str.lower()


# In[ ]:


# copying the train_dataset to train.
train=train_dataset


# In[ ]:


train.head()


# In[ ]:


# copying the test_dataset to test.
test=test_dataset


# In[ ]:


test.head()


# In[ ]:


# conctinating the test and train dataset in order to avoid the repetation of cleaning process on this datasets.
# we will again split the datasets into train and test after the preprocessing of the datasets.
final_dataset=pd.concat([train,test],axis=0)


# In[ ]:


final_dataset.head()


# In[ ]:


## adding a new feature teweet length to the dataset
## it shows the length of the tweets
final_dataset["tweet_length"]=final_dataset["tweet"].apply(lambda x:len(x)-x.count(" "))


# In[ ]:


# adding new column : punct%
# punct% shows the percentage of punctuations used in the tweets

# function for calculating punctuation percentage in each tweet
def punct(text):
    count=sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text)-text.count(" ")),3)*100
final_dataset['punct%']=final_dataset["tweet"].apply(lambda x: punct(x))
final_dataset.head()


# In[ ]:


## remove the non-alphabets
final_dataset["tweet"]=final_dataset["tweet"].str.replace("[^a-z ]","")


# In[ ]:


# visualising the distribution of tweet length in the datasets
fig,ax=plt.subplots(1,3,figsize=(15,5))
sns.distplot(final_dataset["tweet_length"],ax=ax[0],color="mediumturquoise")
sns.boxplot(final_dataset["tweet_length"],hue=label["label"],ax=ax[1],color="turquoise")
sns.violinplot(final_dataset["tweet_length"],hue=label["label"],ax=ax[2],orient="v",color="lightgreen")
plt.grid(True)
# visualising the distribution of punct% in the datasets
fig,ax=plt.subplots(1,3,figsize=(15,5))
sns.distplot(final_dataset["punct%"],ax=ax[0],color="lightgreen")
sns.boxplot(final_dataset["punct%"],ax=ax[1],color="turquoise")
sns.violinplot(final_dataset["punct%"],ax=ax[2],orient="v",color="mediumturquoise")
plt.grid(True)


# In[ ]:


# function for converting raw tweets into tokenized tweets with no stopwords.

def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    return text
# adding new column tweet_nostopwords which consists of tokenized tweets with no stopwords.
final_dataset['tweet_nostopwords'] = final_dataset['tweet'].apply(lambda x: clean_text(x.lower()))
final_dataset.head()


# In[ ]:


# function for converting tokenized tweets into lemmatized tweets.
def lemmatizing(tokenized_text):
    text =" ".join([wn.lemmatize(word) for word in tokenized_text])
    return text

# adding new column tweet_lemmatized which consists of lemmatized tweets.
final_dataset['tweet_lemmatized'] = final_dataset['tweet_nostopwords'].apply(lambda x: lemmatizing(x))

final_dataset.head()


# ### ***Random Forest Model Using Count Vectorizer***

# In[ ]:


# imporitng count vectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


# converting the finsl dataset into vector matrix
count_vect=CountVectorizer(analyzer=clean_text)
X_counts=count_vect.fit_transform(final_dataset["tweet_lemmatized"])
print(X_counts.shape)


# In[ ]:


X_counts # we cannot print the matrix because it is a sparse matrix


# In[ ]:


# in order to print the matrix we have to convert it to array
X_counts_df=pd.DataFrame(X_counts.toarray())
#X_counts_df.columns=count_vect.get_feature_names()
X_counts_df.head(10)


# In[ ]:


# adding the tweet length and punct% column to the  vectorized dataset
a=pd.DataFrame(np.array(final_dataset['tweet_length']).reshape(-1,1),columns=["tweet_length"])
b=pd.DataFrame(np.array(final_dataset['punct%']).reshape(-1,1),columns=["punct%"])
final_dataset1= pd.concat([a,b,X_counts_df],axis=1)


# In[ ]:


# final dataset for fitting ML model
final_dataset1.head()


# In[ ]:


# splitting the dataset into train and test as we specified above.
final_train=final_dataset1.iloc[0:7920,:]
final_test=final_dataset1.iloc[7920:,:]


# In[ ]:


# creating object of the model
rf = RandomForestClassifier(n_jobs=-1,n_estimators=300)


# In[ ]:


# fitting the model
rf.fit(final_train,label['label'])


# In[ ]:


# predicting on test dataset
y_pred=rf.predict(final_test)
y_pred=pd.DataFrame(y_pred)
y_pred.head()


# ### ***Random Forest Model Using TF-IDF Vectorizer***

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(analyzer=clean_text)
x_tfidf=tfidf.fit_transform(final_dataset["tweet_lemmatized"])
print(x_tfidf.shape)


# In[ ]:


# in order to print the matrix we have to convert it to array
x_tfidf_df=pd.DataFrame(x_tfidf.toarray())
#x_tfidf_df.columns=tfidf.get_feature_names()
x_tfidf_df.head()


# In[ ]:


# adding the tweet length and punct% column to the  vectorized dataset
a=pd.DataFrame(np.array(final_dataset['tweet_length']).reshape(-1,1))
b=pd.DataFrame(np.array(final_dataset['punct%']).reshape(-1,1))


# In[ ]:


final_dataset1= pd.concat([a,b,x_tfidf_df],axis=1)


# In[ ]:


# final dataset for fitting ML model
final_dataset1.head()


# In[ ]:


# splitting the dataset into train and test as we specified above.
final_train=final_dataset1.iloc[0:7920,:]
final_test=final_dataset1.iloc[7920:,:]


# In[ ]:


# creating object of the model
rf = RandomForestClassifier(n_jobs=-1,n_estimators=300)


# In[ ]:


# fitting the model
rf.fit(final_train,label['label'])


# In[ ]:


# predicting on test dataset
y_pred=rf.predict(final_test)
y_pred=pd.DataFrame(y_pred)
y_pred.head()

