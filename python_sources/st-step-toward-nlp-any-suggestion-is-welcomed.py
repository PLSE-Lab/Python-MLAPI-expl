#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


dataset=pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter='\t' )


# In[ ]:


dataset.sample(5)


# In[ ]:


import re


# In[ ]:


review=re.sub('[^a-zA-Z]',' ', dataset.Review.iloc[5])


# In[ ]:


review


# In[ ]:


review=review.lower()


# In[ ]:


review


# In[ ]:


import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# In[ ]:


review=review.lower()
review=review.split()
review=[word for word in review]


# In[ ]:


corpus=[]


# In[ ]:


dataset.shape


# In[ ]:


for i in range(0,1000):
    review=re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)
    


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


cv=CountVectorizer(max_features=1500)


# In[ ]:


X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values


# In[ ]:


y.shape


# In[ ]:


X.shape


# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[ ]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


clf=GaussianNB()
clf.fit(X_train, y_train)


# In[ ]:


y_pred=clf.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cm=confusion_matrix(y_test, y_pred)


# In[ ]:


cm


# In[ ]:


x=sum(sum(cm))


# In[ ]:


acc=(cm[0][0]+cm[0][1])/x


# In[ ]:


sum(cm)[0]


# In[ ]:


sum(cm)[1]


# In[ ]:


sum(cm)


# In[ ]:


pre=cm[0][0]/sum(cm)[0]


# In[ ]:


rec=cm[0][1]/sum(cm)[1]


# In[ ]:


F_score=2*pre*rec/(pre+rec)


# In[ ]:


F_score

