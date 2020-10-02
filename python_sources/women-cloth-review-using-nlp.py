#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
print(os.listdir("../input/womens-ecommerce-clothing-reviews"))


# In[ ]:


df = pd.read_csv("..//input//womens-ecommerce-clothing-reviews//Womens Clothing E-Commerce Reviews.csv")
df.head()


# In[ ]:


df.drop(df.columns[0], axis = 1,inplace=True)
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


#type(df['Review Text'])
df['Review Text']=df['Review Text'].astype(str)
df['Review Length']=df['Review Text'].apply(len)
df.head(10)


# In[ ]:


g = sns.FacetGrid(df,col='Rating')
g.map(plt.hist,'Review Length')


# In[ ]:


sns.boxplot(x='Rating',y='Review Length',data=df,palette='rainbow')


# In[ ]:


sns.countplot(x='Rating',data=df,palette='rainbow')


# In[ ]:


ratings = df.groupby('Rating').mean()
ratings


# In[ ]:


ratings.corr()


# In[ ]:


sns.heatmap(ratings.corr(),cmap='coolwarm',annot=True)


# In[ ]:


df_part = df[(df.Rating==1) | (df.Rating==5)]


# In[ ]:


X = df_part['Review Text'].astype(str)

y = df_part['Rating']


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[ ]:


X = cv.fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[ ]:


nb.fit(X_train,y_train)


# In[ ]:


predictions = nb.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# Found good results using Naive Bayes NLP method

# In[ ]:


from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline


# In[ ]:


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[ ]:


X = df_part['Review Text']
y = df_part['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[ ]:


pipeline.fit(X_train,y_train)


# In[ ]:


predictions = pipeline.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# **Using TFIDF receive worse results.**
