#!/usr/bin/env python
# coding: utf-8

# ## Fire up pandas

# In[ ]:


import pandas as pd


# ## Read some product review data

# In[ ]:


products = pd.read_csv('../input/amazon_baby.csv')


# In[ ]:


len(products)


# In[ ]:


products.head()


# ## Build the word count vector for each review

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
word_count = vectorizer.fit_transform(products['review'].values.astype('U'))


# In[ ]:


print (word_count.shape)


# ## Build a sentiment classifier

# ## Define what's a positive and a negative sentiment

# ## positive sentiment = 4 or 5 * rating

# In[ ]:


products['sentiment'] = products['rating'] >= 4


# In[ ]:


products.head()


# ## Let's train  the sentiment classifier

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(word_count, products['sentiment'])


# ## Evaluate the sentiment model

# In[ ]:


clf.coef_.shape


# In[ ]:


clf.coef_


# ## Building a pipeline

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()), ])


# In[ ]:


text_clf.fit(products.review.values.astype('U'), products.sentiment)


# In[ ]:


products['prediction'] = text_clf.predict_proba(products.review.values.astype('U'))[:, [True, False]]


# ## Sort the giraffe reviews based on the predicted sentiment and explore

# In[ ]:


giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[ ]:


giraffe_reviews = giraffe_reviews.sort_values('prediction')


# In[ ]:


giraffe_reviews.head()


# ## Most positive reviews

# In[ ]:


pd.options.display.max_colwidth = 1000
giraffe_reviews.iloc[0]['review']


# In[ ]:


giraffe_reviews.iloc[1]['review']


# ## Show most negative reviews

# In[ ]:


giraffe_reviews.iloc[-1]['review']


# In[ ]:


giraffe_reviews.iloc[-2]['review']


# In[ ]:




