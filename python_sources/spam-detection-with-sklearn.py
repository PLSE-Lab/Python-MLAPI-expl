#!/usr/bin/env python
# coding: utf-8

# In this Tutorial, we will try to analyse the **SMS SPAM COLLECTION DATASET** (a pre-labelled Spam/Not-Spam message dataset). We will be building a model to classify the messages, so lets see how well our model would perform.
# 
# This exercise is a Text mining exerise. Text mining is usually is process in which we try to analyse a set of Text to find meaningful information. This is achieved by attempting to automatically identify themes, patterns and keywords which could lead to the discovery of helpful information instead of having to go through the dataset manually.
# 
# # Let's build a Spam Detection Model
# 
# 1. **Load the Data**
# 
# We will remove every other column and leave only columns we will be working with *"V1" & "V2"*
# 

# In[ ]:


import pandas as pd
import warnings; warnings.simplefilter('ignore')

dataset= pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='ISO-8859-1')
to_drop=['Unnamed: 2','Unnamed: 3','Unnamed: 4']
dataset.drop(columns=to_drop,inplace=True)

dataset.head()


# Here we use Label Encoding, and since we have only tow possiblities in the Label Column (Ham, Spam); Our resulting encoded label would hold(0,1).

# In[ ]:


dataset['encoded_labels']=dataset['v1'].map({'spam':0,'ham':1})
dataset.head()


# 2. **Split Dataset and extract Features**
# 
# We can see that our Dataset is composed of sentences. We need to extract the features in these sentences. In this example, we achieve this by using the Count Vectorizer which generates a Vector holding the frequency values for words in the sentence.

# In[ ]:


from sklearn.model_selection import train_test_split as split_data

labels=dataset.pop('encoded_labels')

train_data,test_data,train_label,test_label=split_data(dataset,labels, test_size=0.3)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

c_v = CountVectorizer(decode_error='ignore')
train_data = c_v.fit_transform(train_data['v2'])
test_data = c_v.transform(test_data['v2'])


# 3. **Build Classifier & Evaluate Accuracy**

# In[ ]:


from sklearn import naive_bayes as nb
from sklearn.metrics import accuracy_score



clf=nb.MultinomialNB()
model=clf.fit(train_data, train_label)
predicted_label=model.predict(test_data)
print("train score:", clf.score(train_data, train_label))
print("test score:", clf.score(test_data, test_label))
print("Classifier Accuracy",accuracy_score(test_label, predicted_label))

