#!/usr/bin/env python
# coding: utf-8

# In[7]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
"../input/NAME_OF_DATAFILE.csv"


# **Using sentiment analysis dictionary as training data set**

# In[14]:


train_ds = pd.read_csv( "../input/dictionary-for-sentiment-analysis/dict.csv", 
                       delimiter=",", encoding = "ISO-8859-1",
                       names = ["Text", "Sentiment"] )


# In[15]:


train_ds.head(10)


# **Replacing positive with 1 and negative with 0**

# In[16]:


train_ds['Sentiment'] = train_ds['Sentiment'].map({'positive': 1, 'negative': 0})


# In[17]:


train_ds.head(10)


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer


# In[19]:


count_vectorizer = CountVectorizer( max_features = 5000 )


# In[21]:


feature_vector = count_vectorizer.fit( train_ds.Text )
train_ds_features = count_vectorizer.transform( train_ds.Text )


# In[22]:


features = feature_vector.get_feature_names()


# In[23]:


features[0:20]


# In[24]:


features_counts = np.sum( train_ds_features.toarray(), axis = 0 )


# In[25]:


feature_counts = pd.DataFrame( dict( features = features,
                                  counts = features_counts ) )


# In[26]:


feature_counts.head(5)


# **Removing stop words**

# In[27]:


count_vectorizer = CountVectorizer( stop_words = "english",
                                 max_features = 5000 )
feature_vector = count_vectorizer.fit( train_ds.Text )
train_ds_features = count_vectorizer.transform( train_ds.Text )


# In[29]:


features = feature_vector.get_feature_names()
features_counts = np.sum( train_ds_features.toarray(), axis = 0 )
feature_counts = pd.DataFrame( dict( features = features,
                                  counts = features_counts ) )
feature_counts.sort_values( "counts", ascending = False )[0:20]


# **Loading test dataset**

# In[ ]:


test_ds = pd.read_csv( "../input/million-headlines/abcnews-date-text.csv", 
                       delimiter=",", encoding = "ISO-8859-1")


# In[34]:


test_ds.head(10)


# In[35]:


test_ds.count()


# In[36]:


test_ds.dropna()


# In[38]:


test_ds.count()


# In[ ]:


headline_text = count_vectorizer.transform( test_ds.headline_t )


# In[40]:


headline_text = count_vectorizer.transform( test_ds.headline_text.astype('U') )


# In[41]:


headline_text[1]


# In[46]:


from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
clf = GaussianNB()


# **Splitting training dataset features into training and test datasets**

# In[48]:


train_x, test_x, train_y, test_y = train_test_split( train_ds_features,
                                                  train_ds.Sentiment,
                                                  test_size = 0.3,
                                                  random_state = 42 )


# In[49]:


clf.fit( train_x.toarray(), train_y )


# In[54]:


test_ds_predicted = clf.predict( test_x.toarray() )


# In[ ]:


from sklearn import metrics
cm = metrics.confusion_matrix( test_y, test_ds_predicted )


# In[56]:


cm


# In[58]:


import matplotlib as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', '')
sn.heatmap(cm, annot=True,  fmt='.2f' )


# In[59]:


score = metrics.accuracy_score( test_y, test_ds_predicted )


# **Poor accuracy!**

# In[60]:


score


# **Fitting the above model to test data and assigning sentiment value**

# In[50]:


test_ds["sentiment"] = clf.predict( headline_text.toarray() )


# In[52]:


test_ds[0:100]


# In[ ]:




