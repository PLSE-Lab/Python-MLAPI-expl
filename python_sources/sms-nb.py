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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk


# In[ ]:


sms = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding='latin')


# In[ ]:


sms.head()


# In[ ]:


sms.columns[2]


# In[ ]:


# Most of the 2,3,4 columns have null values
print(sms.iloc[:,2].isna().sum(),
      sms.iloc[:,3].isna().sum(),
      sms.iloc[:,4].isna().sum())


# In[ ]:


sms = sms.drop([sms.columns[2],sms.columns[3],sms.columns[4]],axis=1)


# Now we only have 2 columns in sms

# In[ ]:


sms.head()


# In[ ]:


sms.v1.value_counts()


# In[ ]:


sns.countplot(sms["v1"])


# In[ ]:


sms.describe()


# In[ ]:


sms.groupby('v1').describe().T


# In[ ]:


sms['length'] = sms['v2'].apply(len)


# In[ ]:


sms.head()


# In[ ]:


sns.set()


# In[ ]:


sms['length'].plot(bins=50,kind="hist")


# In[ ]:


sms.length.describe()


# In[ ]:


sms.hist(column='length',by='v1',bins=50,figsize=(12,3))


# In[ ]:


# The spam messages are longer


# Pre Processing

# In[ ]:


import string

test = "We can turn, the world to go !"

puncless = [c for c in test if c not in string.punctuation]

puncless = "".join(puncless)


# In[ ]:


puncless


# In[ ]:


from nltk.corpus import stopwords
stopwords.words('english')[:20]


# In[ ]:


list(puncless.split())


# In[ ]:


stopless = [w for w in list(puncless.split()) if w not in stopwords.words('english')]


# In[ ]:


stopless


# Now we process all the entries

# In[ ]:


from nltk.corpus import stopwords
def text_process(msg):
    puncless = [c for c in msg if c not in string.punctuation]
    
    puncless = "".join(puncless)
    
    return [w for w in list(puncless.split()) if w.lower() not in stopwords.words('english')]


# In[ ]:


sms['v2'].head(5).apply(text_process)


# In[ ]:





# In[ ]:


#Vectorization


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


bow_transformer = CountVectorizer(analyzer = text_process).fit(sms['v2'])


# In[ ]:


print(len(bow_transformer.vocabulary_))


# In[ ]:


sms_bow = bow_transformer.transform(sms['v2'])
print("Sparse shape",sms_bow.shape)
print("Non zero",sms_bow.nnz)


# In[ ]:


sparsity = (100.0 * sms_bow.nnz / (sms_bow.shape[0] * sms_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity,4)))


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(sms_bow)
sms_tfidf = tfidf_transformer.transform(sms_bow)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(sms_tfidf,sms['v1'])


# In[ ]:


#Testing
print("Predicted:",model.predict(sms_tfidf)[0])
print("expected:",sms.v1[3])


# In[ ]:


pred = model.predict(sms_tfidf)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(sms['v1'],pred))


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


print(accuracy_score(pred,sms['v1']))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
model2 = neigh.fit(sms_tfidf,sms['v1'])
pred2 = model2.predict(sms_tfidf)

print("KNeighbors Classifier accuracy : ",accuracy_score(pred2,sms['v1']))


# In[ ]:


from sklearn.svm import LinearSVC
model3 = LinearSVC(random_state=0).fit(sms_tfidf,sms['v1'])
pred3 = model3.predict(sms_tfidf)
print("SVC accuracy : ",accuracy_score(pred3,sms['v1']))


# In[ ]:


from sklearn.model_selection import train_test_split

