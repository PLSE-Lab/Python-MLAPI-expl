#!/usr/bin/env python
# coding: utf-8

# In this kernel we will study how to do a topic modelling on a Given Dataset.Please Vote if you like the work

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


# In[ ]:


npr=pd.read_csv('../input/topic-modeling/npr.csv')
npr.head()


# In[ ]:


npr.shape


# We can see that our dataset has 11992 documents.Our task will be to segegrate this documents into various topics

# **Displaying the first document **

# In[ ]:


npr['Article'][0]


# In[ ]:


len(npr)


# **Preprocessing of the data**

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


cv=CountVectorizer(max_df=0.9,min_df=2,stop_words='english')


# In[ ]:


dtm=cv.fit_transform(npr['Article'])


# In[ ]:


dtm


# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation


# In[ ]:


LDA=LatentDirichletAllocation(n_components=7,random_state=42)


# In[ ]:


LDA.fit(dtm)


# **Grab the vocubalary of words**

# In[ ]:


len(cv.get_feature_names())


# In[ ]:


type(cv.get_feature_names())


# In[ ]:


cv.get_feature_names()[41000]


# In[ ]:


import random
random_word_id=random.randint(0,54777)
cv.get_feature_names()[random_word_id]


# **Grab the topics**

# In[ ]:


len(LDA.components_)


# In[ ]:


LDA.components_.shape


# In[ ]:


LDA.components_


# **Grab Highest Probability word for each topic**

# In[ ]:


single_topic=LDA.components_[0]


# In[ ]:


single_topic.argsort()


# In[ ]:


import numpy as np
arr=np.array([10,200,1])


# In[ ]:


arr


# In[ ]:


arr.argsort()


# In[ ]:


# Argsort gives index in ascending order 
single_topic.argsort()[-10:]


# In[ ]:


top_ten_words=single_topic.argsort()[-10:]


# In[ ]:


for index in top_ten_words:
    print(cv.get_feature_names()[index])


# In[ ]:


for i,topic in enumerate(LDA.components_):
    print(f"THE TOP 15 WORDS FOR THE TOPIC #{i}")
    print([cv.get_feature_names()[index] for index in topic.argsort()[-15:]])
    print('\n')
    print('\n')


# In[ ]:


topic_results=LDA.transform(dtm)


# In[ ]:


topic_results.shape


# In[ ]:


topic_results[0]


# In[ ]:


topic_results[0].round(2)


# In[ ]:


npr['Article'][0]


# In[ ]:


topic_results[0].argmax()


# In[ ]:


npr['Topic']=topic_results.argmax(axis=1)


# In[ ]:


npr


# In[ ]:


mytopic_dict={0:'Health',1:'Election',2:'Family',3:'Politics',4:'Election',5:'Music',6:'Education'}
npr['Topic Label']=npr['Topic'].map(mytopic_dict)


# In[ ]:


npr

