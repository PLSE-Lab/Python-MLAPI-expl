#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[ ]:


import itertools


# In[ ]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[ ]:


df_X = pd.read_csv('../input/nlp-getting-started/train.csv')
labels = df_X.target
labels.head()
df_Xtrain, df_Xtest, df_Ytrain, df_Ytest = train_test_split(df_X['text'], labels, test_size=0.55950, random_state=5)



# In[ ]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


# In[ ]:


tfidf_train = tfidf_vectorizer.fit_transform(df_Xtrain)


# In[ ]:


tfidf_test = tfidf_vectorizer.transform(df_Xtest)


# In[ ]:


pac = PassiveAggressiveClassifier (max_iter=120)


# In[ ]:


pac.fit(tfidf_train, df_Ytrain)


# In[ ]:


Y_pred = pac.predict(tfidf_test)


# In[ ]:


Score = accuracy_score(df_Ytest, Y_pred)


# In[ ]:


print(Score)
import pickle
saved_model = pickle.dumps(pac)
pac_from_pickle = pickle.loads(saved_model)


# In[ ]:


z = pd.read_csv('../input/nlp-getting-started/test.csv')
Z = z['text']
tfidf_real = tfidf_vectorizer.transform(Z)
tfidf_real.shape


# In[ ]:


final_result = pac_from_pickle.predict(tfidf_real)


# In[ ]:


sub_z = pd.DataFrame({"id" : z['id'], "target" : final_result})


# In[ ]:


sub_z.to_csv("sub_z.csv", index = None)


# In[ ]:




