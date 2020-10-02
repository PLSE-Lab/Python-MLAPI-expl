#!/usr/bin/env python
# coding: utf-8

# # Multinomial Naive Bayes classifier

# In[ ]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load data set
df = pd.read_csv("/kaggle/input/mydata/restaurantreviews.csv", sep="\t", names=['Review','Liked'], encoding="latin-1")


# In[ ]:


# Print all data
df = df[1:]
df


# In[ ]:


# Rows and columns
df.shape


# In[ ]:


# Rows
df.shape[0]


# In[ ]:


# columns
df.shape[1]


# In[ ]:


#First 5 rows
df.head(5)


# In[ ]:


#Last 5 records
df.tail(5)


# In[ ]:


# filter data
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
wordnet=WordNetLemmatizer()
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'].values[i])
    review = review.lower()
    review = review.split()
    
    review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[ ]:


# Vectorize setences and define x and y
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(corpus).toarray()
y = df['Liked'].values


# In[ ]:


#Split test train data into 70:30
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# In[ ]:


#Prepare Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


#fit the data
detect_model = MultinomialNB().fit(x_train, y_train)


# In[ ]:


#Predict result
y_pred=detect_model.predict(x_test)


# In[ ]:


#classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:


#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[ ]:


#Accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:




