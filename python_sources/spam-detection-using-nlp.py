#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')


# In[ ]:


data.head()


# In[ ]:


data=data.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)


# In[ ]:


data.columns=['type','message']


# In[ ]:


data.head()


# In[ ]:


data.groupby('type').describe()


# In[ ]:


data['length'] = data['message'].apply(len)
data.head()


# In[ ]:


data.length.describe()


# In[ ]:


sns.set_style('darkgrid')
data['length'].plot(bins=50, kind='hist') 


# In[ ]:


import string 
from nltk.corpus import stopwords


# In[ ]:


def clean_text(message):
        no_punc=[char for char in message if char not in string.punctuation]
        no_punc=''.join(no_punc)
        return[word for word in no_punc.split() if word.lower() not in stopwords.words('english')]


# In[ ]:


data.head()


# In[ ]:


data['message'].head(5).apply(clean_text)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


msg_train,msg_test,type_train,type_test=train_test_split(data['message'],data['type'],test_size=.2)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[ ]:


from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bag_of_words', CountVectorizer(analyzer=clean_text)),  # strings to token integer counts
    ('tf-idf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),
])


# In[ ]:


pipeline.fit(msg_train,type_train)


# In[ ]:


predictions = pipeline.predict(msg_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print('classification matrix is',classification_report(predictions,type_test))
print('confusion matrix is',confusion_matrix(predictions,type_test))


# In[ ]:




