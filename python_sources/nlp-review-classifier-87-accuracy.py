#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords


# In[ ]:


reviews = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv").dropna()
reviews.info()


# Label every text if it is positive, neutral or negative comment as 1, 0 and -1

# In[ ]:


reviews.loc[reviews['Sentiment_Polarity'] >0,'Positivity']=1
reviews.loc[reviews['Sentiment_Polarity'] ==0,'Positivity']=0
reviews.loc[reviews['Sentiment_Polarity'] <0,'Positivity']=-1

reviews.head(5)


# In[ ]:


data = pd.concat([reviews.Translated_Review,reviews.Positivity],axis=1)
data.head(3)


# **Clean, Lemmatize Data**

# In[ ]:


def prepare(data):
    CleanData=[]

    for descr in data.Translated_Review:
        descr=re.sub("[^a-zA-Z]"," ",descr) 
        descr=descr.lower()
        descr = nltk.word_tokenize(descr)
        descr = [word for word in descr if not word in set(stopwords.words("english"))] 
        lemma = nltk.WordNetLemmatizer()
        descr = [lemma.lemmatize(word) for word in descr]
        descr = " ".join(descr)
        CleanData.append(descr)
    return CleanData
    
CleanData=prepare(data)


# In[ ]:


max_features= 300

count_vectorizer = CountVectorizer(max_features=max_features,stop_words="english")
sparce_matrix=count_vectorizer.fit_transform(CleanData).toarray()
print("Most used {} word: {}".format(max_features,count_vectorizer.get_feature_names()))
#%%
y= data.iloc[:,1].values # Disaster or not
x=sparce_matrix #Texts for training
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.01,random_state=42)


# In[ ]:


#%%Naive bayes 
nb = GaussianNB()
nb.fit(x_train,y_train)
#%% Logistic Regression
from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state=42,max_iter=100, C=10)
logreg.fit(x_train,y_train.T)


# In[ ]:


print("Test accuracy of naive bayes: ",nb.score(x_test,y_test))
print("Test accuracy of Logistic Regression:  ",logreg.score(x_test,y_test.T))


# In[ ]:


from keras.utils import to_categorical 
train=to_categorical(y_train+1)
test =to_categorical(y_test+1)
print(test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.callbacks import ModelCheckpoint
model = Sequential()
model.add(Dense(input_dim=x_train.shape[1],
                output_dim = train.shape[1],
                init =   'uniform',
                activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(200, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dense(train.shape[1], kernel_initializer='uniform'))
model.add(Activation('softmax'))

model.compile(loss = "binary_crossentropy", optimizer = 'adamax',metrics=["accuracy"])
model.fit(x_train,
          train,
          epochs = 200,
          batch_size = 1000,
          validation_data = (x_test,test),
          verbose=1)


# In[ ]:


print("Test loss and accuracy of Neural Network: ",model.evaluate(x_test,test))


# PLOT TRAINING

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Test Loss','Train Accuracy','Test Accyracy'], loc='upper left')
plt.show()


# Comparing Naive Bayes, Logistic Regeression and Neural Network accuracies

# In[ ]:


y_axis=[nb.score(x_test,y_test),logreg.score(x_test,y_test.T),model.evaluate(x_test,test)[1]]
x_axis=["Naive Bayes","Logistic Regression","Neural Network"]

fig,ax=plt.subplots(figsize=(10,6))
ax.bar(x_axis,y_axis)
plt.show()


# **Neural Network reaches up to %87 test accuracy which is very nice. Adding layers don't change loss and without initializing first weights with lecun uniform model is limits itself by %80 accuracy. Model maybe improved using BERT model and other initializer-regularizer combinations.**
