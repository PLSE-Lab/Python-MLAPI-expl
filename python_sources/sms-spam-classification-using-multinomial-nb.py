#!/usr/bin/env python
# coding: utf-8

# ## Import the libraries

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


# Read the data set

data = pd.read_table('....input/SMSSpamCollection', header=None, names=['Class', 'sms'])
data.head(10)


# In[ ]:


len(data)


# ## Exploratory Data Analysis

# In[ ]:


# Get the count, unique and frequency of the data

data.describe()


# In[ ]:


# Now groupby the class column & describe it

data.groupby('Class').describe()


# In[ ]:


# Check the length of the each SMS

data['length']=data['sms'].apply(len)
data.head(10)


# In[ ]:


# Convert the class into numerical value

data['labels'] = data.Class.map({'ham':0, 'spam':1})
data.head(10)


# In[ ]:


# Now drop the class column

data = data.drop('Class', axis = 1)
data.head(10)


# ## Visualize the Data

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data['length'].plot(bins=50,kind='hist')


# In[ ]:


data.length.describe()


# In[ ]:


# A message with 910 words

data[data['length']==910]['sms'].iloc[0]


# ## Text Pre - Processing

# In[ ]:


# Example using to create a function which will be use in later part of the code

import string
mess = 'sample message!...'
nopunc=[char for char in mess if char not in string.punctuation]
nopunc=''.join(nopunc)
print(nopunc)


# In[ ]:


# Now import the stopwords

from nltk.corpus import stopwords
stopwords.words('english')


# In[ ]:


nopunc.split()


# In[ ]:


# Check whether the word nopunc.split() is present in stopwords.words('english') or not

clean_mess=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess


# In[ ]:


# Now let's put both of these together in a function to apply it to our DataFrame later on:

def text_process(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# ### Text Pre - Processing - Tokenization

# In[ ]:


# Tokenization can be done but by not removing puctuation

from nltk.tokenize import sent_tokenize, word_tokenize
data['sms'].apply(word_tokenize).head(10)


# In[ ]:


data['sms'].apply(sent_tokenize).head(10)


# ### Or 

# In[ ]:


# By removing puctuation

data['sms'].head(10).apply(text_process)


# ### Continuing Normalization - Stemming

# In[ ]:


#create an object of class PorterStemmer

from nltk.stem import PorterStemmer
porter = PorterStemmer()


# In[ ]:


data['sms'] = data['sms'].str.lower()


# In[ ]:


print ([porter.stem(word) for word in data['sms']])


# In[ ]:


# Convert to X & y

X = data.sms
y = data.labels


# In[ ]:


print(X.shape)


# In[ ]:


print(y.shape)


# ## Split the data into train & test

# In[ ]:


from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# ### Vectorization

# In[ ]:


# Vectorizing the sentence & removing the stopwords

from sklearn.feature_extraction.text import CountVectorizer
#vect = CountVectorizer(stop_words='english')

# or we can use the fuction text_process in place of stop_words = 'english'
vect = CountVectorizer(analyzer=text_process)


# In[ ]:


vect.fit(X_train)


# In[ ]:


# Printing the vocabulary

vect.vocabulary_


# In[ ]:


# vocab size
len(vect.vocabulary_.keys())


# In[ ]:


# transforming the train and test datasets

X_train_transformed = vect.transform(X_train)
X_test_transformed = vect.transform(X_test)


# In[ ]:


# note that the type is transformed (sparse) matrix

print(type(X_train_transformed))
print(X_train_transformed)


# ### Building and Evaluating the Model

# In[ ]:


# training the NB model and making predictions
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

# fit
mnb.fit(X_train_transformed,y_train)

# predict class
y_pred_class = mnb.predict(X_test_transformed)

# predict probabilities
y_pred_proba = mnb.predict_proba(X_test_transformed)


# In[ ]:


# note that alpha=1 is used by default for smoothing
mnb


# ### Model Evaluation

# In[ ]:


# printing the overall accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[ ]:


# confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)
# help(metrics.confusion_matrix)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_pred_class))
print(confusion_matrix(y_test, y_pred_class))


# In[ ]:


confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]


# In[ ]:


sensitivity = TP / float(FN + TP)
print("sensitivity",sensitivity)


# In[ ]:


specificity = TN / float(TN + FP)
print("specificity",specificity)


# In[ ]:


precision = TP / float(TP + FP)
print("precision",precision)
print(metrics.precision_score(y_test, y_pred_class))


# In[ ]:


print("precision",precision)
print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class))
print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))
print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class))


# In[ ]:


y_pred_class


# In[ ]:


y_pred_proba


# ## ROC Curve

# In[ ]:


# creating an ROC curve
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[ ]:


# area under the curve
print (roc_auc)


# In[ ]:


# matrix of thresholds, tpr, fpr
pd.DataFrame({'Threshold': thresholds, 'TPR': true_positive_rate, 'FPR':false_positive_rate})


# In[ ]:


# plotting the ROC curve
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)


# In[ ]:


print(len(X_train),len(X_test),len(y_train),len(y_test))


# ### Creating a Data Pipeline

# In[ ]:


from sklearn.pipeline import Pipeline
pipeline = Pipeline([
   ( 'bow',CountVectorizer(analyzer=text_process)),
    ('classifier',MultinomialNB()),
])


# In[ ]:


pipeline.fit(X_train,y_train)


# In[ ]:


predictions = pipeline.predict(X_test)


# In[ ]:


print(classification_report(predictions,y_test))


# In[ ]:




