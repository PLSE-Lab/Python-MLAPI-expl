#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[ ]:


messages = pd.read_csv("../input/spam.csv",encoding='latin-1')
messages.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1,inplace=True)
messages.rename(columns={"v1":"label", "v2":"message"},inplace=True)
messages.head()


# ## Exploratory Data Analysis

# In[ ]:


messages.describe()


# Let's look at labels closely.

# In[ ]:


messages.groupby('label').describe()


# **To better understand the data and improve the accuracy with our ML model, lets add a synthetic feature 'length'**

# In[ ]:


messages['length'] = messages['message'].apply(len)
messages.head()


# ### Data Visualization
# 

# In[ ]:


messages['length'].plot(bins=10, kind='hist') 


# **The new feature looks normally distributed and hence can act as a pretty good feature**

# In[ ]:


messages.length.describe()


# Woah! 910 characters, let's use masking to find this message:

# In[ ]:


messages[messages['length'] == 910]['message'].iloc[0]


# **Looks like some sort of Love letter! But let's focus back on the idea of trying to see if message length can act as a good feature to disting between ham and spam:**

# In[ ]:


messages.hist(column='length', by='label', bins=50,figsize=(12,4))


# **We can clearly see there IS a trend in the the text length of 'spam' and 'ham messages' **

# ## Text Pre-processing

# **Let's create the function that tockenize's our datset by removing unwanted punctuations and stopwords.**

# In[ ]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# 
# ## Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# The test size is 20% of the entire dataset (1115 messages out of total 5572), and the training is the rest (4457 out of 5572).
# 
# ## Creating a Data Pipelines
# **Let's create data pipelines that vectorizes and converts to bag-of-words(BOW), applies TF-IDF, and fits a classifier to the data**
# 

# ## Naive Bayes

# In[ ]:


from sklearn.pipeline import Pipeline

pipeline1 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[ ]:


pipeline1.fit(msg_train,label_train)


# In[ ]:


predictions1 = pipeline1.predict(msg_test)


# In[ ]:


print(classification_report(predictions1,label_test))
nbscore=accuracy_score(predictions1,label_test)
print(nbscore)


# ## Logistic regression

# In[ ]:


pipeline2 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', LogisticRegression()),  # train on TF-IDF vectors w/ Logistic regression
])


# In[ ]:


pipeline2.fit(msg_train,label_train)
predictions2=pipeline2.predict(msg_test)
print(classification_report(predictions2,label_test))
lrscore=accuracy_score(predictions2,label_test)
print(lrscore)


# ## K-Nearest Neigbors

# In[ ]:


pipeline3 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', KNeighborsClassifier()),  # train on TF-IDF vectors w/ K-nn
])


# In[ ]:


pipeline3.fit(msg_train,label_train)
predictions3=pipeline3.predict(msg_test)
print(classification_report(predictions3,label_test))
knnscore=accuracy_score(predictions3,label_test)
print(knnscore)


# ## Support Vector Machine
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


pipeline4 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', SVC()),  # train on TF-IDF vectors w/ SVM
])


# In[ ]:


pipeline4.fit(msg_train,label_train)
predictions4=pipeline4.predict(msg_test)
print(classification_report(predictions4,label_test))
svmscore=accuracy_score(predictions4,label_test)
print(svmscore)


# ## Decision Tree

# In[ ]:


pipeline5 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', DecisionTreeClassifier()),  # train on TF-IDF vectors w/ DecisionTreeClassifier
])


# In[ ]:


pipeline5.fit(msg_train,label_train)
predictions5=pipeline5.predict(msg_test)
print(classification_report(predictions5,label_test))
dtscore=accuracy_score(predictions5,label_test)
print(dtscore)


# In[ ]:


results=[nbscore,lrscore,knnscore,svmscore,dtscore]
n=['Naive-B','Log. Reg.','KNN','SVM','Dtree']


# In[ ]:


ndf=pd.DataFrame(n)
rdf=pd.DataFrame(results)
rdf[1]=n


# In[ ]:


print('Accuracy')
rdf


# **It is very clear that Naive-bayes, Logistic regression, and Decision tree are the best fit for this dataset with accuracies around 95%**

# ## Thank You!
