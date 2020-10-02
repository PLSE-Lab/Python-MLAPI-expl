#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df=pd.read_csv("/kaggle/input/amazon-e-commerce-data-set/ecommerceDataset.csv",names=['Type','Text'])


# In[ ]:


df.head()


# In[ ]:


df['Type'].value_counts()


# In[ ]:


df.shape


# In[ ]:


df.Text.nunique()


# In[ ]:


df.groupby('Type').describe()


# In[ ]:


df.columns


# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


df.dropna(axis=0,inplace=True)


# In[ ]:


import nltk
import string


# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


string.punctuation


# In[ ]:


df['Text'][10] #Original text


# In[ ]:


kk="".join([r for r in df['Text'][10] if r not in string.punctuation]) #depunctuated text


# In[ ]:


kk


# In[ ]:


stopwords.words("english")


# In[ ]:


kk.split(" ") #orginal text


# In[ ]:


[w for w in kk.split(" ") if w not in stopwords.words("english")] # after removing stopwords


# In[ ]:


def fun(w):
    w=[r for r in w if r not in string.punctuation] #removes punctuation
    w="".join(w)
    return [ x for x in w.split(" ") if x.lower() not in stopwords.words("english")]


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer 


# In[ ]:


vec=CountVectorizer(analyzer=fun).fit(df['Text'])


# In[ ]:


vec=vec.transform(df['Text'])


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
vec_tfidf=TfidfTransformer().fit(vec)
vec_tfidf=vec_tfidf.transform(vec)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X=df['Text']
y=df['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
r=RandomForestClassifier(n_estimators=100)


# In[ ]:


from sklearn.pipeline import Pipeline
pp=Pipeline([
    ('b',CountVectorizer()),
    ('C',TfidfTransformer()),
    ('r',RandomForestClassifier(n_estimators=100))
])


# In[ ]:


pp.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(pp.predict(X_test),y_test))


# In[ ]:


df.loc[101]['Type'] #Actual Value


# In[ ]:


pp.predict(['AsianHobbyCrafts Wooden Embroidery Hoop Ring Frame (3 Pieces) Style Name:Assorted A   Asian Hobby Crafts embroidery collection comprises of embroidery frames (in various sizes), cross stitch fabric, embroidery tools, embroidery wool. This embroidery hoop frame is made of well finished wood with a easy-to-adjust screw mounted on the frame to tighten the fabric. Cross stitch art is a phenomenal art form which involves intricate stitching techniques to form beautiful designs on fabric.'])[0]
#Original value


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




