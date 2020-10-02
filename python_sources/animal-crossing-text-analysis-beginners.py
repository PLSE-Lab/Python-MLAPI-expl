#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


#Lets Take our Critics Data!
df=pd.read_csv('../input/animal-crossing/critic.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


plt.hist(df['date'],bins=10)
plt.xticks(rotation=70)


# In[ ]:


df['text'][6]


# In[ ]:


plt.hist(df['grade'],bins=10)


# In[ ]:


#This will be our preprocessing or Data-Cleaning State, 
#Where we'll import the PorterStemmer for Stemming, and StopWords to avoid the words like (he, a, the, or..etc).
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
text=[]


# In[ ]:


len(df)


# In[ ]:


for i in range(0,len(df)):
    review=re.sub('[^a-zA-Z]',' ',df['text'][i]) #Here Except a-z and A-Z, we'll replace all the punctuations,etc, with a ''.
    review=review.lower() #Here we lower the Uppercase characters to Lower one!.
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')] #We'll Stem all the words, except those which are StopWords!
    review=' '.join(review)
    text.append(review)


# In[ ]:


text


# So, This is our data, which is purified and Stemmed!
# 

# In[ ]:


#Creating a TF-IDF Model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(ngram_range=(1,2))


# I selected Tf-Idf, over bag-of-words, for the semantation sake!

# In[ ]:


#Converting our text to array!
X=tfidf.fit_transform(text).toarray()
X.shape


# In[ ]:


y=df['grade']
y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


tfidf.get_feature_names()[:20]


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RR = RandomForestRegressor(random_state=0)
RR.fit(X_train,y_train)


# In[ ]:


rslt = RR.predict(X_test)
rslt


# In[ ]:




