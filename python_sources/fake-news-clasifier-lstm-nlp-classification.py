#!/usr/bin/env python
# coding: utf-8

# # FAKE NEWS CLASSIFIER USING LSTM
# 
# IN THIS NOTEBOOK WE WILL CLASSIFY WHETHER THE NEWS IS FAKE OR NOT USING LSTM MODEL.
# 
# 1--> REAL
# 
# 0--> FAKE
# 
# WE WILL PERFORM SOME TEXT PREPROCESSING( STEMMING, REMOVAL OF STOP WORDS, CONVERTING TEXT INTO VECTORS)
# 
# THEN WE WILL BUILD OUR LSTM MODEL TO CLASSIFY THE NEWS

# In[ ]:


# IMPORTING LIBRARY
import pandas as pd


# In[ ]:


# OUR DATA FRAME
df= pd.read_csv('../input/fake-news-dataset/fakenews_train.csv')
df.head()


# In[ ]:


# DROPPING NaN VALUES
df=df.dropna()


# In[ ]:


## GETTING INDEPENDENT FEATURES
x= df.drop('label',axis=1)


# In[ ]:


# DEPENDENT FEATURE OR TARGET FEATURE
y=df['label']


# In[ ]:


# LOOKING AT THE SHAPE OF OUR IINDEPENDENT FEATURES DATASET
x.shape


# In[ ]:


# LOOKING AT THE SHAPE OF OUR DEPENDENT FEATURE
y.shape


# In[ ]:


# IMPORTING MORE NECESSARY LIBRARIES
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


# In[ ]:


# vocabularry size
voc_size=5000


# # NATURAL LANGUAGE PROCESSING

# In[ ]:


# COPYING OUR INDEPENDENT FEATURE DATASET 'x' INTO NEW VARIABLE 'messages'
messages= x.copy()


# In[ ]:


# RESETTING THE INDEX VALUES
messages.reset_index(inplace=True)


# In[ ]:


# IMPORTING LIBRARIES FOR NATURAL LANGUAGE PROCESSING
import nltk
import re
from nltk.corpus import stopwords


# In[ ]:


# DOWNLOADING THE STOPWORDS
nltk.download('stopwords')


# In[ ]:


# TEXT PREPROCESSING--> STEMMING, REMOVAL OF STOP WORDS, CONVERTING INTO LOWER CASE
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[ ]:


# LOOKING AT OUR FINAL CORPUS
corpus


# # ONE HOT REPRESENTATION

# In[ ]:


from tensorflow.keras.preprocessing.text import one_hot


# In[ ]:


onehot_repr= [one_hot(words,voc_size)for words in corpus]
print(onehot_repr)


# # EMBEDDING REPRESENTATION

# In[ ]:


# SETTING OUR SENTENCE LENGTH = 20
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


# In[ ]:


len(embedded_docs)


# # CREATING OUR LSTM MODEL

# In[ ]:


## CREATING OUR LSTM MODEL
embedding_vector_features= 40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length)) # making embedding layer
model.add(LSTM(100))  # one LSTM Layer with 100 neurons
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[ ]:


# CONVERTING X AND Y INTO ARRAYS
import numpy as np
x_final= np.array(embedded_docs)
y_final= np.array(y)


# In[ ]:


# SPLITTING THE DATA INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,test_size=0.33,random_state=0)


# # MODEL TRAINING

# In[ ]:


# FITTING NTO THE MODEL
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)


# In[ ]:


# OUR PREDICTION VARIABLE
y_pred= model.predict_classes(x_test)


# In[ ]:


y_pred


# In[ ]:


# IMPORTING LIBRARIES TO SEE THE ACCURACY
from sklearn.metrics import confusion_matrix, accuracy_score

# PRINTING CONFUSION MATRIX
cm= confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


# ACCURACY SCORE
ac= accuracy_score(y_test,y_pred)
print(ac)


# ##### ACCURACY SCORE= 91.66 %

# # ADDING DROPOUT LAYER

# In[ ]:


from tensorflow.keras.layers import Dropout


# In[ ]:


## CREATING MODEL WITH DROPOUT LAYER
embedding_vector_features= 40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length)) # making embedding layer
model.add(Dropout(0.3))
model.add(LSTM(100))  # one LSTM Layer with 100 neurons
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[ ]:


# CONVERTING X AND Y VARIABLES INTO ARRAYS
import numpy as np
x_final= np.array(embedded_docs)
y_final= np.array(y)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,test_size=0.33,random_state=0)


# In[ ]:


# FITTING INTO THE MODEL
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)


# In[ ]:


# PREDICTION MODEL
y_pred= model.predict_classes(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score

# PRINTING CONFUSION MATRIX
cmm= confusion_matrix(y_test,y_pred)
print(cmm)


# In[ ]:


# ACCURACY SCORE
ac= accuracy_score(y_test,y_pred)
print(ac)


#  ##### ACCURACY SCORE AFTER ADDING DROPOUT LAYER= 91.48%
# 
# 
#  ##### OUR ACCURACY DECREASED A LITTLE IN THIS CASE AFTER ADDING DROPOUT LAYER
