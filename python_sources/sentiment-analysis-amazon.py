#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np

df=pd.read_csv('../input/Amazon_Unlocked_Mobile.csv')

df=df.sample(frac=0.1,random_state=10)
df.head()


# In[25]:


df.shape


# In[26]:


#Droping missing values
df.dropna(inplace=True)


#Removing any neutral rating =3
df=df[df['Rating']!=3]

#encode 4 an 5 as 1
#1 and 2 as 0
df['Positively Rated']=np.where(df['Rating']>3,1,0)
df.head()


# In[27]:


df['Positively Rated'].mean()


# In[28]:


from sklearn.model_selection import train_test_split

#spliting data into training and test 
X_train,X_test,y_train,y_test=train_test_split(df['Reviews'],df['Positively Rated'],random_state=0)


# In[29]:


X_train.iloc[0]


# In[30]:


X_train.shape


# looking at extreme we have a series over 23052 reviews or documents we need to convert this into numerical representation with sklearn, the bag of words approch is simple and commonly used way to represent text for use in machine learning it ignores sructure and only counts how often each word occurs 

# # count vectorizer
# court vectorizer allows us to use bag of words approch by converting collection of text documents in to a matrix of token counts 
# 
# First we initiate the count vectorizer and fit it to our training data. Fitting the count vectorizer consists of the tokens of the training data and building of the vocabulary 
# 
# Fitting the count vectorizer tokenizes each document by finding all sequences of characters that is numbers or letters seperated by word boundaries converts every thing to lower case and builds a vocabulary using these tokens 

# In[31]:


from sklearn.feature_extraction.text import CountVectorizer

vect=CountVectorizer().fit(X_train)


# In[32]:


vect.get_feature_names()[::2000]


# In[33]:


len(vect.get_feature_names())


# In[34]:


X_train_vectorized = vect.transform(X_train)

X_train_vectorized


# In[35]:


from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X_train_vectorized,y_train)


# In[36]:


from sklearn.metrics import roc_auc_score

predictions=model.predict(vect.transform(X_test))
print('AUC:',roc_auc_score(y_test,predictions))


# In[37]:


# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# 
# # tfidf(term frequency inverse document frequency(allows us to rescale features)
#  
# tfidf allows us to weight terms how imp they are to the document
# high weight are given to the terms that apper to the document but dont appear often in a corpus 
# features with low tfidf are come in use in all documents 

# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())


# In[39]:


X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[40]:


#features with smallest and largest tfidf
feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))


# In[41]:


sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[42]:


# These reviews are treated the same by our current model
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


# # NGRAM
# 

# In[43]:


# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())


# In[44]:


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[45]:


feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[ ]:





# In[ ]:




