#!/usr/bin/env python
# coding: utf-8

# **Hello There**
# 
# In this notebook I am going to create a linear SVM model to predict the authenticity of a news. If you found any mistake feel free to drop a comment. Cheers

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# After we create the fake and true dataset, we need to add a "status" column to both of the datasets. Where the value will be 1 if it belongs to the fake class and 0 otherwise
# 
# Then we combine both as one dataframe, called it df

# In[ ]:


true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true['status'] = 0
fake['status'] = 1

df = pd.concat([true, fake])
df = df.sample(frac = 0.5)


# Notice at the end of the line, i do a sampling by fraction of 0.5. This is due to the ram capacity. When i tried this locally on my computer just use the original dataframe is fine, however here i need to scale it down by half so that the ram can finish the computation

# In[ ]:


df


# Next, we are going to convert all capitals in both title and text columns into lower case characters
# And drop the subject and date, as we are going to make use only the words fron the text and title

# In[ ]:


df.title = df.title.str.lower()
df.text = df.text.str.lower()
df = df.drop(columns = ['subject','date'])


# Next, I merge together the title and text and add a spcae between these two

# In[ ]:


df.text = df.title + ' ' + df.text


# In[ ]:


df = df.drop(columns = ['title'])


# In[ ]:


df.head()


# Now I created two dataframe, one for training and one for testing with test size 0.3

# In[ ]:


train, test = train_test_split(df, test_size = 0.3, random_state = 7)


# In[ ]:


train


# In[ ]:


test


# In order to extract the words from all the news, I am going to make use of count vectorizer function

# In[ ]:


cv = CountVectorizer(stop_words = 'english')
fitting = list(train.text)
cv.fit(fitting)


# In[ ]:


features = cv.transform(fitting).toarray()


# In[ ]:


inv_vocab = {v: k for k, v in cv.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]


# In[ ]:


new_train = pd.DataFrame(features, columns = vocabulary)


# In[ ]:


new_train


# In[ ]:


#Finding the least 80000 used words to be removed from the training dataset
to_remove = list(new_train.sum(axis = 0).sort_values()[:65000].index)


# The majority of the words appear less than 5 times in the whole news and hence it is un necessary for the training model, thus I decided to remove the least 65000 keywords appeared in the whole news. This will also fasten the training time

# In[ ]:


new_train = new_train.drop(columns = to_remove)


# In[ ]:


new_train


# In[ ]:


svc = LinearSVC()
svc.fit(new_train, train.status)


# For the test dataframe, we need to create the exact features as the train dataframe

# In[ ]:


test_features = cv.transform(list(test.text)).toarray()
new_test = pd.DataFrame(test_features, columns = vocabulary)
new_test = new_test.drop(columns = to_remove)
new_test


# In[ ]:


ans = svc.predict(new_test)


# And Voila! We achieved 99.4% of accuracy in the test set. This is because SVM model is a very good model when we are working with data that has a lot of features such as in this case

# In[ ]:


accuracy_score(ans, test.status)

