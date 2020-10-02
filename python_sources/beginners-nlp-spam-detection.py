#!/usr/bin/env python
# coding: utf-8

# Hello, our goal in this notebook is to build  a model which can accurately predict if the SMS is spam or not spam.
# 
# We will be converting Text to vectors.
# We will be using classifiers to differentiate between spam and not spam.

# ## Importing Libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 


# In[ ]:


df = pd.read_csv(r'/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')


# In[ ]:


df.head()


# In[ ]:


df.info()


# ### Since we have many **NULL** values in column 3,4,5 we will be removing the columns

# In[ ]:


df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1) 


# In[ ]:


df.head()


# ### Will Check for the null value if there is any in the second columns

# In[ ]:


df['v2'].isnull().sum()


# ### We will rename columns

# In[ ]:


df = df.rename(columns={"v1": "status", "v2": "content"})


# In[ ]:


df.shape


# We need to preprocess the data. Preprocessing data usually helps in increasing accuracy of the model.
# In preprocess we will.
# * Remove emails
# * Remove Urls
# * Remove Numbers
# * Remove non alph-numeric
# * Remove stop words if necessary

# In[ ]:


from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))


# In[ ]:


def stop_words(text):
    words = text.split(" ")
    # Lemmatization 
    lemmatizer = WordNetLemmatizer()
    words_lemmatizer = [lemmatizer.lemmatize(ele) for ele in words]
    nltk_remove_sw = [ele for ele in words_lemmatizer if ele not in stopWords]

    sentence_lemmatized = " ".join(nltk_remove_sw)
    
    return sentence_lemmatized


# In[ ]:


def pre_process(text):
    tx1 = re.sub('(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', '', text) # URL's without http
    tx2 = re.sub('[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', tx1)# URL's with HTTP
    tx3 = re.sub('[\w\.-]+@[\w\.-]+', '', tx2)  # Emails
    tx4 = re.sub(r'[0-9]+', '', tx3) # Numbers
    tx5 = re.sub(r"[^A-Za-z0-9 ]", ' ', tx4) # Non aplha-numerics
    tx6 = re.sub("\s\s+", " ", tx5) 
    tx7 = tx6.lower() 
    sentence_lemmatized  = stop_words(tx7)
    return sentence_lemmatized


# In[ ]:


pre_processed_text = []
for ele in df['content']:
    pre_processed_text.append(pre_process(ele))
    


# In[ ]:


df.insert(loc=2, column='Pre_processed', value=pre_processed_text)


# In[ ]:


df.head()


# In[ ]:


df.info()


# ### We will split data to training and testing
# 
# From Data Frame the content is the input and status is the output.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df['Pre_processed'], df['status'], test_size=0.35, random_state=42)


# In the above code we have treated the df['content'] has input and df['status'] as output. We have set the test_data to be  35% of the main dataset and random_state is 42 by convention, random_state shuffels the rows and create datframe.

# ## For System to process we need to convert **Text** to **Numbers**
# Used to transform text to a vector of term / token counts.

# In[ ]:


vector = CountVectorizer(max_df=0.7)
X_train_vector = vector.fit_transform(X_train)
X_test_vector = vector.transform(X_test)


# In[ ]:


pd.DataFrame(X_train_vector.toarray(), columns=vector.get_feature_names())


# ## We will  check with a few classifier

# ## Gaussian Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
clf_GNB = GaussianNB()
clf_GNB.fit(X_train_vector.toarray(), y_train)
print(f'accuracy: {clf_GNB.score(X_test_vector.toarray(), y_test )}')


# ## Decision Tree Classification

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf_DTC = DecisionTreeClassifier()
clf_DTC.fit(X_train_vector, y_train)
print(f'accuracy: {clf_DTC.score(X_test_vector, y_test)}')


# ## Support Vector Classifier

# In[ ]:


from sklearn.svm import SVC
clf_SVC = SVC()
clf_SVC.fit(X_train_vector, y_train)
print(f'accuracy: {clf_SVC.score(X_test_vector, y_test)}')


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression()
clf_LR.fit(X_train_vector, y_train)
print(f'accuracy: {clf_LR.score(X_test_vector, y_test)}')


# ## Summary
# 
# * We started with checking the NULL values, and droped few columns
# * We did few pre-processing test
#     * This included removal of emails, Urls, Numbers etc...    
# * We then converted text to a vector
# * We applied this vectors to different classifier, and found accuracy of the model. We performed this on the test data set.
# 
# ### Observation
# * Even without pre-processing the text, their was no big difference in the model accuracy.
# * Their was a slight increase in model accuracy with **max_df** set to 0.7 in CountVectorizer
# * Tried the model with bigram, accuracy went down a bit on few models. No major improvements
# 
