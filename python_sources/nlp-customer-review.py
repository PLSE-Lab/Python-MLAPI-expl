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


# **Importing Modules** 

# In[ ]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer


# In[ ]:


# Importing the dataset
df = pd.read_csv('../input/restaurantreviews/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# In[ ]:


df.head()


# In[ ]:


#df['Review'][0]


# **We will be keeping only letters and Removing numbers and Puncuations **

# In[ ]:


#review=re.sub('[^a-zA-Z]',' ',df['Review'][0])


# In[ ]:


#review


# We can see now that we have removed puncuation from the first review

# **Converting all the letters in the review into small case**

# In[ ]:


#review=review.lower()


# In[ ]:


#review


# So we can see that the all the letters of the review column are converted to small case 

# **Removing non Significant words (Stop Words)**

# In[ ]:


nltk.download('stopwords')
from nltk.corpus import stopwords


# Converting the review into list of words 

# In[ ]:


#review=review.split()


# In[ ]:


#ps=PorterStemmer()


# In[ ]:


#review


# We Can see now that review is list of four words

# **Stemming to keep only the root word**

# In[ ]:


#review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#review


# We can see that after stemming Loved was converted to Love in out list of words review

# **Joining the words in the list **

# In[ ]:


#review=' '.join(review)


# In[ ]:


#review


# We can see that all the words in the list are joined and we have used space between the words

# **Now lets clean data for whole data set using a loop**

# In[ ]:


corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',df['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)


# In[ ]:


corpus[0:5]


# We have the new list corpus which has all the words from the 1000 reviews.Using a for loop all the process like removing puncuations,making all words lower case and removing stop words and joining done in one stage 

# **Creating Bag of Words Model**

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()


# In[ ]:


X.shape


# We can remove to non frequent words by using max_features.In this case there are 1565 words in the word vector matrix.We can use word max as 1500 which would effectively remove 65 words from the spare word vector matrix

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()


# In[ ]:


X.shape


# In[ ]:


y=df.iloc[:,1].values


# In[ ]:


y.shape


# **Splitting data into train and test Data**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15)


# **Scaling the data **

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)


# **Fitting the Naive Bayes model to the training data**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)


# **Predicting the test results **

# In[ ]:


y_pred=classifier.predict(X_test)


# * **Making the confusion matrix,Classification report and Accuracy Score**

# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(classification_report(y_test,y_pred))


# In[ ]:


print(accuracy_score(y_test,y_pred))

