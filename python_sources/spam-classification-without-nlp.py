#!/usr/bin/env python
# coding: utf-8

# # Spam Classification
# 

# Let's Get started, I have used datasets from
# UCI Spam dataset : https://www.kaggle.com/uciml/sms-spam-collection-dataset

# Now, here we start with spam classification so we will allotting binary values to labels so that Machine Learning model can work efficiently in predicting the results

# # Steps taken
# * Load the libraries
# * Data Cleaning
# * Assigning Binary Values to Labels
# * Data Visualization (Part-1)
# * LowerCasing, Punctuation removing and Vocabulary modifications
# * Counting The Occurence of Words
# * Training, Testing Part of the model
# * Data Visualization (Part-2)

# # Loading the libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


filepath='../input/sms-spam-collection-dataset/spam.csv'
df=pd.read_csv(filepath, encoding='latin-1')
df.head()


# # Data Cleaning
# We, start with dropping columns with missing values

# In[ ]:


df=df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
df=df.rename(columns={'v1':'labels','v2': 'sms'})
df.head()


# # Assigning Binary Values

# We fix our response values for spam and ham

# In[ ]:


df['labels']=df.labels.map({'spam':0, 'ham':1})
df.head()


# In[ ]:


df.shape


# In[ ]:


df['length']=df['sms'].apply(len)
df.head()


# # Data Visualization (Part-1)

# In[ ]:


plt.figure(figsize=(16,6))
sns.distplot(a=df['length'],kde=False)
plt.legend()


# In[ ]:


message=df[df['length']==910]['sms'].iloc[0]
message


# # LowerCasing, Punctuations and Vocab. modifications

# Now we will implement Bag of Words which will count the number of words based on their frequency distribution and that binary number will be fed for Machine Learning model
# 

# We start with using lowercase for all the words in the above sentence

# In[ ]:


message={"""
For me the love should start 
         with attraction.i should feel that 
         I need her every time around me.she should be the first thing which comes in my thoughts.
         I would start the day and end it with her.she should be there every time I dream.love will be 
         then when my every breath has her name.my life should happen around her.my life will be named to her.
         I would cry for her.will give all my happiness and take all her sorrows.I will be ready to fight with anyone for her.
         I will be in love when I will be doing the craziest things for her.love will be when I don't have to proove anyone that 
         my girl is the most beautiful lady on the whole planet.I will always be singing praises for her.love will be when 
         I start up making chicken curry and end up makiing sambar.life will be the most beautiful then.
         will get every morning and thank god for the day because she is with me.I would like to say a lot..will tell later.
"""}
lower_case=[]
for i in message:
    lower_case=[i.lower() for i in message]
    print(lower_case)


# Now we will use punctutation for sorting out the sentences

# In[ ]:


sans_punctuation = []
import string

for i in lower_case:
    sans_punctuation.append(i.translate(str.maketrans('', '', string.punctuation)))
print(sans_punctuation)


# **Tokenization**

# In[ ]:


preprocessed_documents = []
for i in sans_punctuation:
     preprocessed_documents=[[w for w in i.split()] for i in message]
print(preprocessed_documents)


# Now we begin with counting the numbers as how much is their frequency

# In[ ]:


import pprint
from collections import Counter
frequency_num=[]

for i in preprocessed_documents:
    frequency_count=Counter(i)
    frequency_num.append(frequency_count)
pprint.pprint(frequency_num)


# # Counting The Occurence of Words

# Let's try the above with CountVectorizer tool 

# In[ ]:


count_vector=CountVectorizer()
print(count_vector)


# Now, using count_vector i have converted the words to vocabulary as well

# In[ ]:


count_vector.fit(message)
voc=count_vector.get_feature_names()
voc


# 
# we convert the message words to array form

# In[ ]:


doc_to_array=count_vector.transform(voc).toarray()
doc_to_array


# Table created 

# In[ ]:


frequency_matrix = pd.DataFrame(doc_to_array, 
                                columns = count_vector.get_feature_names())
frequency_matrix


# # Training and Testing the model

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df['sms'],df['labels'],random_state=1)
print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# In[ ]:


training_data=count_vector.fit_transform(X_train)
testing_data=count_vector.transform(X_test)


# In[ ]:


mnb=MultinomialNB()
mnb.fit(training_data, y_train)

predictions=mnb.predict(testing_data)
mnb_accuracy = accuracy_score(y_test,predictions)
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))

print('precision score: ', format(precision_score(y_test,predictions)))
print('recall score: ', format(recall_score(y_test,predictions)))
print('f1 score: ', format(f1_score(y_test,predictions)))


# Using Decision Trees

# In[ ]:


dtc=DecisionTreeClassifier()
dtc.fit(training_data,y_train)

predictions=dtc.predict(testing_data)
dtc_accuracy = accuracy_score(y_test,predictions)
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('precision score: ', format(precision_score(y_test,predictions)))
print('recall score: ', format(recall_score(y_test,predictions)))
print('f1 score: ', format(f1_score(y_test,predictions)))


# RandomForest Classifier

# In[ ]:


rfc=RandomForestClassifier()
rfc.fit(training_data,y_train)

predictions=rfc.predict(testing_data)
rfc_accuracy = accuracy_score(y_test,predictions)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('precision score: ', format(precision_score(y_test,predictions)))
print('recall score: ', format(recall_score(y_test,predictions)))
print('f1 score: ', format(f1_score(y_test,predictions)))


# KNN

# In[ ]:


knn=KNeighborsClassifier()
knn.fit(training_data, y_train)

predictions=knn.predict(testing_data)
knn_accuracy = accuracy_score(y_test,predictions)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('precision score: ', format(precision_score(y_test,predictions)))
print('recall score: ', format(recall_score(y_test,predictions)))
print('f1 score: ', format(f1_score(y_test,predictions)))


# Bagging Classifer and AdaBoost

# In[ ]:


bgc=BaggingClassifier()
bgc.fit(training_data, y_train)

predictions=bgc.predict(testing_data)
bgc_accuracy = accuracy_score(y_test,predictions)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('precision score: ', format(precision_score(y_test,predictions)))
print('recall score: ', format(recall_score(y_test,predictions)))
print('f1 score: ', format(f1_score(y_test,predictions)))


# In[ ]:


#AdaBoost
adb=AdaBoostClassifier()
adb.fit(training_data, y_train)
predictions=adb.predict(testing_data)
adb_accuracy = accuracy_score(y_test,predictions)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('precision score: ', format(precision_score(y_test,predictions)))
print('recall score: ', format(recall_score(y_test,predictions)))
print('f1 score: ', format(f1_score(y_test,predictions)))


# # Data Visualization (Part-2)

# **Accuracy Plots estimations **

# In[ ]:


clf=(mnb_accuracy,dtc_accuracy,rfc_accuracy,knn_accuracy,bgc_accuracy,adb_accuracy)
plt.figure(figsize=(16,6))
sns.distplot(a=clf, hist=True)
plt.xlabel('Accuracy scores')
plt.title('Accuracy comparison')
plt.legend()


# Bar plot for all model accuracies

# In[ ]:


sns.barplot(data=clf)
plt.title('Accuracy estimates')
plt.legend()


# **Confusion Matrix**

# In[ ]:


cm=true_positive, false_negative, false_positive, true_negative = confusion_matrix(y_test,predictions).ravel()
print('True positive : ',true_positive)
print('False negative : ',false_negative)
print('False positive : ',false_positive)
print('True negative : ',true_negative)

accuracy = (true_positive + true_negative)/(true_positive + false_negative + false_positive + true_negative)
print('General accuracy : ',accuracy)


# So our model has a very good predictions on messages that are spam or ham and using certain algorithms like AdaBoost can enhance the accuracy of the model.

# Any suggestions feel free to comment

# Hope you enjoyed this!!
