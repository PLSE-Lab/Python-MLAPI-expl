#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as ml
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
ml.style.use('ggplot')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')
print(data.shape)
data.head()


# In[ ]:


data.info()


# ### RENAMING COLUMNS

# In[ ]:


cols = {'v1':'Label','v2':'Text'}
data = data.rename(columns=cols)
data


# ### FIND RATIO OF NULL VALUES IN LAST 3 COLUMNS TO DECIDE WHETHER TO KEEP OR DROP

# In[ ]:


col3 = data['Unnamed: 2'].isnull().sum()/data.shape[0]
col4 = data['Unnamed: 3'].isnull().sum()/data.shape[0]
col5 = data['Unnamed: 4'].isnull().sum()/data.shape[0]

print("Portion of NaN values in the 3rd column : ",(col3*100),"%")
print("Portion of NaN values in the 4th column : ",(col4*100),"%")
print("Portion of NaN values in the 5th column : ",(col5*100),"%")


# ##### We see a huge magnitude of the data is NULL in the last 3 columns. So we drop the last 3 columns

# In[ ]:


data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
data.head(10)


# ## FILTERING OUT THE COMMON WORDS AND FORMING A FREQUENCY DATASET.

# In[ ]:


# Getting all words in all 5572 SMS's
texts = list(data.Text.values)
words = []
for i in texts:
    words.extend(i.split(" "))
len(words)


# In[ ]:


# Replacing non-alphabetical instances
for i in range(len(words)):
    if not words[i].isalpha():
        words[i] = ""

# Create a Counter dictionary to maintain ('word':count) tuples.
word_dict = Counter(words)
word_dict.most_common(1)     # Get the 1st most common word.


# ##### This shouldn't be the case. The most common word can't be a ' '. So we remmove the occurrence of ' '(unwanted character) in the word_dict

# In[ ]:


del word_dict['']
# Check
word_dict.most_common(1)


# In[ ]:


# Getting the 3000 most common words in text messages (Why 3000 ? Hit and trial)
w_new = word_dict.most_common(3000)
w_new


# ##### Seems alright. Now we form our features(word counts for each message)
# 
# ## FEATURE ENGINEERING

# In[ ]:


features,columns = [],[]

# Creating the columns(features)
for word in w_new:
    columns.append(word[0])

# Creating data
for s in texts:
    d = []
    for wrd in w_new:
        d.append(s.count(wrd[0]))
    features.append(d)

# Create dataframe
df = pd.DataFrame(features,columns=columns)
df


# ## DROPPING 'TEXT' AND ASSIGNING LABELS MANUALLY
# 
# ##### 1. Since the label is of two types only ('spam' or 'ham'), we can manually encode them instead of using a LabelEncoder() object.
# ##### 2. We do not need the text messages feature anymore, because we have already got the count of 3000 most popular words in each one of them. So it is not merged

# In[ ]:


df['label'] = 0
df


# In[ ]:


labels = {'spam':1,'ham':0}             # Spam = 1, Not spam = 0
for i in range(df.shape[0]):
    df.iloc[i,3000] = int(labels[data.iloc[i,0]])
df


# ## TRAINING THE MODEL

# In[ ]:


# X = df.iloc[:,:3000].values
X = np.array(features)
# Y = df['label'].values
Y = np.array(df.label)

train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.7,random_state=10)
print("Train X size = ",train_x.shape,", Test X size = ",test_x.shape)
print("Train Y size = ",train_y.shape,", Test Y size = ",test_y.shape)


# In[ ]:


# Defining the Naive Bayes classifier
mnb = MultinomialNB(alpha = 5, fit_prior = True)
# Training the model
mnb.fit(train_x,train_y)


# ## PREDICTIONS AND CONFUSION MATRIX
# ##### Confusion matrix to get an idea of the accuracy of the model. X-axis = Predicted label. Y-axis = True label.

# In[ ]:


y_pred = mnb.predict(test_x)
print(confusion_matrix(test_y,y_pred,labels=[1,0],normalize=None))
tn,fp,fn,tp = confusion_matrix(test_y,y_pred).ravel()
print("\nTP = ",tp,", FP = ",fp,", FN = ",fn,", TN = ",tn)
print("Confirmation : TP + FP + FN + TN = ",tp+fp+fn+tn)
print("Equal to test_y size(3901)")
# Accuracy
print("\nAccuracy = ",(tp+tn)*100/(tp+tn+fp+fn),"%")


# In[ ]:




