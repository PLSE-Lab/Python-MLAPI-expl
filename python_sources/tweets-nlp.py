#!/usr/bin/env python
# coding: utf-8

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
import numpy as np
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))
stop_words.extend(['CA','ca','!','@','#','$','%','&','*','(',')',',','.','?','/','{','}','[',']'])


# In[ ]:


train = pd.read_csv("../input/nlp-getting-started/train.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")


# In[ ]:


train.head()


# In[ ]:


train['target'].value_counts()


# In[ ]:


train['text'][2]


# In[ ]:


train[train.keyword.notna()]['keyword'].values
# for x in train[train.keyword.notna()]['keyword'].values:
#     print(x.replace('%20',' '))


# In[ ]:


train['keyword'].isna().sum()


# In[ ]:


kws= list(train[train.keyword.notna()]['keyword'].values)

for x,y in zip(train.loc[train.keyword.isna(),'text'].head(10),train.loc[train.keyword.isna(),'keyword']):
#     for y in kws:
#         if y in x:
#             print(y)
#             train[train.text==x]['keyword'] = y
    print(x,y)    
    


# In[ ]:


for x in train.loc[train.keyword.isna(),'text']:
    for y in kws:
        if y.lower() in x.lower():
            print(y)
            train.loc[train.text==x,'keyword'] = y
            break


# In[ ]:


train['keyword'].isna().sum()


# In[ ]:


for x in train.loc[train.keyword.isna(),'text']:
    print(x.lower())


# Now that we've filled the keywords of most of disaster tweets,the others don't really matter
# 
# Still we've countable number(3) of disaster tweets left without keywords of which we can manually fill 2 of them with heat wave ;)

# In[ ]:


for x in train.loc[train.keyword.isna(),'text']:
    if 'heat wave' in x.lower():
        train.loc[train.text==x,'keyword'] = 'heat wave'
    elif 'oil spill' in x.lower():
        train.loc[train.text==x,'keyword'] = 'oil spill'
        


# In[ ]:


train.keyword.isna().sum()


# In[ ]:


for x in train.loc[train.keyword.isna(),'text']:
    print(x.lower(),train.loc[train.text==x,'target'])


# In[ ]:


train['keyword'] = train['keyword'].fillna('general')


# In[ ]:


train['keyword'].isna().sum()


# Yeah ! We've filled NaN values with appropraite values in keywords

# In[ ]:


train.head()


# In[ ]:


train.loc[train['location'].isna(),'text']


# In[ ]:


train['location'].isna().sum()


# In[ ]:


train.shape


# So around 5000 locations are identified. We'll use them to try fill the unknown locations

# Since we've extracted the keywords,let's remove the #,@ from text so that few locations in #,@ can be extracted

# In[ ]:


train['text'][0][0]


# In[ ]:


locs = list(train.loc[train.location.notna(),'location'].values)


for x in train.loc[train['location'].isna(),'text']:
    for y in locs:
        if y.lower() in word_tokenize(x.lower()) and y.lower() not in stop_words:
            train.loc[train['text']==x,'location'] = y
            
            break
        
    


# In[ ]:


train.location.isna().sum()


# In[ ]:


train.to_csv('modified1.csv')


# Obviously, the above method wasn't accurate as many locations weren't appropriate. But most of the irrelevant entries we've made are for non disaster tweets while disastrous tweets got proper location entries. So,it won't degrade our model performance

# In[ ]:


train.loc[train['location'].isna(),'text']


# In[ ]:


train['location'] = train['location'].fillna('unknown')


# In[ ]:


train.location.isna().any()


# In[ ]:


train.head()


# In[ ]:


ids = train['id']
train.drop(['id'],axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


target = train['target']
train.drop(['target'],axis=1,inplace=True)


# In[ ]:


train.head()

