#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import packages
import pandas as pd
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.layers import LSTM
from keras.layers import Dense


# In[ ]:


#read train data
data = pd.read_csv("../input/fake-news/train.csv")
data.shape


# In[ ]:


#print top 3 rows
data.head(3)


# In[ ]:


#information about train data
data.info()


# In[ ]:


#droping na data
data = data.dropna()
data.shape


# In[ ]:


#features
X = data.drop("label",axis=1)
X.shape


# In[ ]:


#label/target
y = data["label"]
y.shape


# In[ ]:


import re #for regular espression
from nltk.corpus import stopwords #for stopword remove
from nltk.stem.porter import PorterStemmer #for stemming


# In[ ]:


message = X.copy()
message.head(3)


# In[ ]:


#reset index because we removed na rows
message.reset_index(inplace=True)
message.head(3)


# In[ ]:


#creating corpus
ps = PorterStemmer()
corpus = []
for i in range(0, len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[ ]:


corpus[0:5]


# In[ ]:


#onehot Representation
voc_size = 5000
onehot_repr=[one_hot(words,voc_size)for words in corpus] 


# In[ ]:


onehot_repr[0:5]


# In[ ]:


#Embedding Representation
sent_len = 20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_len)
print(embedded_docs)


# In[ ]:


embedded_docs[0]


# In[ ]:


#creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_len))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[ ]:


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)
X_final.shape,y_final.shape


# In[ ]:


#split data into train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


# In[ ]:


#training/ fit the model
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


# In[ ]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


from keras.layers import Dropout
## Creating model with dropout
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_len))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


#training/ fit the model
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


# In[ ]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:




