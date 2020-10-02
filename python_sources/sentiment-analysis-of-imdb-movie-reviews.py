#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:




df=pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df['review'][0]


# In[ ]:





# ## Text Cleaning
# 
# - sample 10000 rows
# - remove html tags
# - remove special characters
# - converting specialcharecters
# - converting to lower case 
# - removing stop words
# - stemming
# 

# In[ ]:


df.iloc[0:25000,:]['sentiment'].value_counts()


# In[ ]:


df=df.iloc[0:5000,:]


# In[ ]:


df.shape


# In[ ]:


df['sentiment'].value_counts()


# In[ ]:


df.info()


# In[ ]:


df['sentiment'].replace({'positive':1,'negative':0},inplace=True)


# In[ ]:


df.head()


# In[ ]:


import re
def clean_html(text):
    
    clean = re.compile('<.*?>')
    return re.sub(clean, '',text)
    


# In[ ]:


df['review']=df['review'].apply(clean_html)


# In[ ]:



df['review']


# In[ ]:




def convert_lower(text):
    return text.lower()


# In[ ]:


df['review']=df['review'].apply(convert_lower)


# In[ ]:





# In[ ]:


df['review']


# In[ ]:



def remove_special(text):
        x=''
        for i in text:
            if i.isalnum():
                x=x+i
            else:
                x=x+' '
        return x


# In[ ]:


df['review'] = df['review'].apply(remove_special)


# In[ ]:


df['review']


# In[ ]:


import nltk

from nltk.corpus import stopwords


# In[ ]:



def remove_stopwords(text):
    x=[]
    for i in text.split():
        
        if i not in stopwords.words('english'):
            x.append(i)
    y=x[:]
    x.clear()
    return y


# In[ ]:


df['review']=df['review'].apply(remove_stopwords)


# In[ ]:


df.head()


# In[ ]:


from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()


# In[ ]:


y=[]

def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z


# In[ ]:


df['review']=df['review'].apply(stem_words)


# In[ ]:


## join

def join_back(list_input):
    return " ".join(list_input)
    


# In[ ]:


df['review']=df['review'].apply(join_back)


# In[ ]:


df


# In[ ]:



from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=800)


# In[ ]:


X=cv.fit_transform(df['review']).toarray()


# In[ ]:


X.shape


# In[ ]:


X[1].max()


# In[ ]:


y=df.iloc[:,-1].values


# In[ ]:



y.shape


# In[ ]:


#X,y 

# Training set
# Test set




# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.5) 


# In[ ]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


# In[ ]:




clf1=GaussianNB()
clf2=MultinomialNB()
clf3=BernoulliNB()


# In[ ]:




clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)


# In[ ]:




y_pred1=clf1.predict(X_test)
y_pred2=clf2.predict(X_test)
y_pred3=clf3.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:




print("Gaussian",accuracy_score(y_test,y_pred1))
print("Multinomial",accuracy_score(y_test,y_pred2))
print("Bernaulli",accuracy_score(y_test,y_pred3))


# In[ ]:





# In[ ]:




