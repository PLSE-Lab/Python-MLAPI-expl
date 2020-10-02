#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


def review_to_wordlist( review, remove_stopwords=True):
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()

    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)

    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    
    # 4. Optionally remove stop words (True by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    b=[]
    stemmer = english_stemmer #PorterStemmer()
    for word in words:
        b.append(stemmer.stem(word))

    # 5. Return a list of words
    return(b)


# In[3]:


data_file = '../input/Amazon_Unlocked_Mobile.csv'

n = 413000  
s = 20000 
skip = sorted(random.sample(range(1,n),n-s))


df = pd.read_csv( data_file, delimiter = ",", skiprows = skip)
#print(df)
# Any results you write to the current directory are saved as output.


# In[4]:


# Drop missing values
df.dropna(inplace=True)
# Remove any 'neutral' ratings equal to 3
df = df[df['Rating'] != 3]
# Encode 4s and 5s as 1 (rated positively)
# Encode 1s and 2s as 0 (rated poorly)
df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)
df.head(10)


# In[5]:


print(len(df['Positively Rated']))
#df['Reviews']

#for i in range(0,len(df['Positively Rated'])-1,1):
    #review_to_wordlist(df['Positively Rated'][i])
# Most ratings are positive


# In[6]:


#data = df[df['Rating'].isnull()==False]
df['Positively Rated'].mean()


# In[7]:


#train, test = train_test_split(data, test_size = 0.3)
#X_train, X_test, y_train, y_test = train_test_split(df['Rating'], df['Positively Rated'], random_state=0)


# In[ ]:





# In[8]:


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], df['Positively Rated'], random_state=0)


# In[ ]:





# In[9]:


print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)


# In[10]:


# Fit the CountVectorizer to the training data

vect = CountVectorizer().fit(X_train)
#print(vect)


# In[13]:



vect.get_feature_names()[::2000]


# In[ ]:



#print(vect.get_feature_names()[0])
#print(len(vect.get_feature_names()))
for i in range(0,len(vect.get_feature_names())-1,1):
    if vect.get_feature_names()[i].isalpha and len(vect.get_feature_names()[i])>2:
        #if remove_stopwords==True:
         #   stops = set(stopwords.words("english"))
          #  words = [w for w in words if not w in stops]
        #[x.lower() for x in vect.get_feature_names()]
        vect.get_feature_names()
#vect.get_feature_names()


# In[ ]:


vect.get_feature_names()[::2000]


# In[ ]:


# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(X_train)

X_train_vectorized


# In[ ]:


#data.reshape((999,1))
# Train the model
#X = X_train_vectorized.reshape(X_train_vectorized.shape[1:])
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[ ]:


#Predict the transformed test documents
predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[ ]:


# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[ ]:


# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
for i in range(0,len(vect.get_feature_names())-1,1):
    if vect.get_feature_names()[i].isalpha:
        vect = TfidfVectorizer(min_df=5).fit(X_train)
        #len(vect.get_feature_names())
        vect.get_feature_names()
        #[x.lower() for x in vect.get_feature_names()]


# In[ ]:


X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[ ]:


feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))


# In[ ]:


sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[ ]:


# These reviews are treated the same by our current model
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


# In[ ]:


# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())


# In[ ]:


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[ ]:


feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[ ]:


# These reviews are now correctly identified
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))

