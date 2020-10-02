#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import spacy
from sklearn.svm import LinearSVC


# # Data Analysis

# In[ ]:


data = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')
data.head()


# In[ ]:


(data.overall.value_counts() / data.overall.count()) * 100


# Value counts show that 5 star ratings account for 67.6% of the data set while the combination of 4,3,2,1 only account for 32.4%. My goal is to classify positive and negative samples, so I will be considering 5, 4 to be positive while 3, 2, 1 are negative

# In[ ]:


data.loc[data.overall < 4, 'overall'] = 0
data.loc[data.overall >= 4, 'overall'] = 1
(data.overall.value_counts() / data.overall.count()) * 100


# In[ ]:


(data.isnull().sum() / data.count()) * 100


# Reviewer name is not used for training. Review text is though. Since less than 0.07% of data is missing, I will just be dropping those rows.

# In[ ]:


data.dropna(subset=['reviewText'], inplace=True)


# In[ ]:


(data.isnull().sum() / data.count()) * 100


# # Vectorization and Model
# Model uses the core web lg model to vectorize dataset for Linear SVC classification

# In[ ]:


nlp = spacy.load('en_core_web_lg')


# In[ ]:


with nlp.disable_pipes():
    doc_vectors = np.array([nlp(text).vector for text in data.reviewText])
    
doc_vectors.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(doc_vectors, data.overall, test_size=0.1)


# In[ ]:


svc = LinearSVC(dual=False, max_iter=10000)
svc.fit(X_train, y_train)
print(f"Accuracy {svc.score(X_test, y_test) * 100}")


# # Test Examples
# Few examples taken from amazon reviews.

# In[ ]:


def predict(text):
    with nlp.disable_pipes():
        test_vector = nlp(text).vector
    return svc.predict([test_vector])[0]


# In[ ]:


print(predict("I really enjoyed this product."))
print(predict("The sound was terrible. Product is overpriced."))
print(predict("The reed was very loose, but overall you get what you pay for."))
print(predict("You get so much for the price: the thousands of sounds/software that come with it are really useful, especially for a starting producer . The product is well designed and the built is solid. Would totally recommend !"))
print(predict("Software that came in the bundle took nearly 3 hours to upload, then found out it was corrupt!"))

