#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from tqdm import tqdm


# In[2]:


data = pd.read_csv("../input/train.csv")


# In[3]:


def pre_process(text):
 
    text = text.translate(str.maketrans('', '', string.punctuation))
   
#     text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
   
    return text


# In[4]:


y= data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]


# In[ ]:


len(y)


# In[5]:


x=  data['comment_text'].apply(pre_process)


# In[6]:


x= x.tolist()


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=300)
vectorizer.fit(x)


# In[8]:


X= vectorizer.transform(x)


# In[9]:


Y= y.values


# In[10]:


from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import MultinomialNB
classifier = ClassifierChain(MultinomialNB(alpha=0.7))


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.08, random_state=42)


# In[12]:


classifier.fit(X_train, y_train)


# In[13]:


predictions = classifier.predict(X_test)


# In[14]:


from sklearn.metrics import accuracy_score


# In[15]:


accuracy_score(y_test,predictions)


# In[ ]:


# from skmultilearn.adapt import MLkNN
# classifier1 = MLkNN(k=6)
# classifier1.fit(X_train, y_train)
# predictions1 = classifier1.predict(X_test)



# In[ ]:


# accuracy_score(y_test,predictions1)


# In[ ]:


# from sklearn.ensemble import RandomForestClassifier
# randclf = RandomForestClassifier(n_estimators=100,random_state=2)


# In[ ]:


# randclf.fit(X_train, y_train)


# In[ ]:


# randclf.score(X_test, y_test)

