#!/usr/bin/env python
# coding: utf-8

# ## Project on predicting if a news is Real or Fake

# ##### Importing Necessary Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns


# #### Loading the Dataset

# In[ ]:


df=pd.read_csv('../input/textdb3/fake_or_real_news.csv')
df.head()


# In[ ]:


df.shape


# ##### Checking the Class imbalance

# In[ ]:


df['label'].value_counts(normalize=True)


# In[ ]:


sns.barplot(x=df['label'].value_counts().index,y=df['label'].value_counts().values,data=df)
plt.show()


# From the above value counts and the barplot we can see that the class labels are balanced and hence we can proceed further without any over/under sampling techniques.

# In[ ]:


x=df['text']
y=df['label']


# #### Splitting the data into training and test set

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# #### Initializing  TF-IDF Vectorizer to convert text into a TF-IDF matrix

# In[ ]:


vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
train=vectorizer.fit_transform(x_train)
test=vectorizer.transform(x_test)


# ##### Building Models and checking Accuracy And Misclassification

# In[ ]:


## Logistic Regression
lr=LogisticRegression()
lr.fit(train,y_train)
y_pred=lr.predict(test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


## Naive -Bayes Classifier (Used widely for text classification)
nb=BernoulliNB()
nb.fit(train,y_train)
y_pred=nb.predict(test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


### Passive Aggressive Classifier 
pac= PassiveAggressiveClassifier()
pac.fit(train,y_train)
y_pred= pac.predict(test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


confusion_matrix(y_test,y_pred)


# ###### Observation 

# From the above Three classification models we can see that Passive Aggressive Classifier outperforms the other two by the accuracy score and also does less misclassification i.e. less False Negative and Positive with an accuracy of 93.58%.

# In[ ]:





# In[ ]:




