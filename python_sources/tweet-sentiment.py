#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.columns


# In[ ]:


train.describe()


# In[ ]:


train.isnull().sum()


# In[ ]:


train["text"].fillna(method = "ffill" , limit = 1 , inplace = True)
train["selected_text"].fillna(method = "ffill" , limit = 1 , inplace = True)


# In[ ]:


train.isnull().sum()


# In[ ]:


sentiments = train.sentiment.value_counts()
sentiments = pd.DataFrame(sentiments)
sentiments.head()


# In[ ]:


sentiments.shape


# In[ ]:


px.bar(sentiments , x = sentiments.index , y = "sentiment" , color = "sentiment")


# In[ ]:


freq_words = train["text"].str.split(expand = True).unstack().value_counts()


# In[ ]:


px.bar(freq_words , x = freq_words.index.values[2:40] , y = freq_words.values[2:40] , color = freq_words.values[2:40])


# In[ ]:


all_words1 = train["selected_text"].str.split(expand = True).unstack().value_counts()
all_words1.head()


# In[ ]:


px.bar(all_words1 , x = all_words1.index.values[1:40] , y = all_words1.values[1:40] , color = all_words1.values[1:40])


# In[ ]:


df = pd.DataFrame(train.text)
df.head(40)


# In[ ]:


df1 = pd.DataFrame(train.selected_text)
df1.head(20)


# In[ ]:


train.head()


# In[ ]:


train["text"] = train.text.str.replace("," , " ")


# In[ ]:


train["selected_text"] = train.selected_text.str.replace("," , " ")


# In[ ]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


# In[ ]:


stop = set(stopwords.words('english')) 
  


# In[ ]:


train['text_without_stopwords'] = train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[ ]:


train['selected_text_without_stopwords'] = train['selected_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[ ]:


train.head()


# In[ ]:


train["test"]=train['selected_text'].apply(lambda x: [item for item in x.split() if item not in stop])


# In[ ]:


train.drop("text" , axis = 1 , inplace = True)


# In[ ]:


imp_words = train.selected_text_without_stopwords.str.split(expand = True).unstack().value_counts()


# In[ ]:


px.bar(imp_words , x = imp_words.index.values[1:40] , y = imp_words.values[1:40] , color = imp_words.values[1:40])


# In[ ]:


px.bar(imp_words , x = imp_words.index.values[-100:] , y = imp_words.values[-100:] , color = imp_words.values[-100:])


# In[ ]:




