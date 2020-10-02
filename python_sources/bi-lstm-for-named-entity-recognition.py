#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)
plt.style.use("ggplot")

import tensorflow as tf
print('Tensorflow version:', tf.__version__)
print('GPU detected:', tf.config.list_physical_devices('GPU'))


# # ###  Load and Explore the NER Dataset

# *Essential info about tagged entities*:
# - geo = Geographical Entity
# - org = Organization
# - per = Person
# - gpe = Geopolitical Entity
# - tim = Time indicator
# - art = Artifact
# - eve = Event
# - nat = Natural Phenomenon

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Loading data
data = pd.read_csv('/kaggle/input/ner-datasetcsv/ner_dataset.csv',encoding='latin1')
data = data.fillna(method = 'ffill')
data.head(20)


# In[ ]:


print("Unique words in corpus:",data['Word'].unique())
print("Unique Tags in corpus:",data['Tag'].unique())


# In[ ]:


words = list(set(data["Word"].values))
words.append("ENDPAD")
num_words = len(words)


# In[ ]:


tags = list(set(data["Tag"].values))
num_tags = len(tags)


# In[ ]:


num_words, num_tags


# # ******Retrieve Sentences and Corresponsing Tags

# In[ ]:


class SentanceGetter(object):
    def __init__(self,data):
        self.n_sent = 1
        self.data = data
        agg_func = lambda s: [(w,p,t) for w,p,t in zip(s["Word"].values.tolist(),
                                                       s["POS"].values.tolist(),
                                                       s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentances = [s for s in self.grouped]


# In[ ]:


getter = SentanceGetter(data)
sentances = getter.sentances


# In[ ]:


# checking the structure of sentances 
sentances[1]


# # ###  Define Mappings between Sentences and Tags

# In[ ]:


word2idx = {w: i+1 for i, w in enumerate(words)}
tag2idx = { t: i for i,t in enumerate(tags)}


# In[ ]:


word2idx


# # ### Padding Input Sentences and Creating Train/Test Splits

# In[ ]:


plt.hist([len(s) for s in sentances], bins=60)
plt.show()


# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
max_len = 60
X = [[word2idx[w[0]] for w in s] for s in sentances]
X = pad_sequences(maxlen = max_len, sequences =X , padding ='post',value =num_words-1)
y = [[tag2idx[w[2]] for w in s]for s in sentances]
y = pad_sequences(maxlen=max_len,sequences =y,padding ='post',value=tag2idx["O"])
y = [to_categorical(i, num_classes=num_tags) for i in y]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state =1)


# # ### Build and Compile a Bidirectional LSTM Model

# In[ ]:


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional


# In[ ]:


input_word = Input(shape = (max_len))
model = Embedding(input_dim = num_words,output_dim = max_len,input_length  = max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units = 100,return_sequences=True,recurrent_dropout = 0.1))(model)
out = TimeDistributed(Dense(num_tags, activation ='softmax'))(model)
model = Model(input_word,out)
model.summary()


# In[ ]:


model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# # ###  Train the Model

# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[ ]:


early_stopping = EarlyStopping(monitor='val_accuracy',patience=2,verbose = 1, mode ='max',restore_best_weights = True)
callbacks = early_stopping
history = model.fit( x_train,np.array(y_train), validation_split = 0.1, batch_size = 32,epochs = 20, verbose =1,callbacks=callbacks)


# # ### Evaluate Named Entity Recognition Model

# In[ ]:


model.evaluate(x_test,np.array(y_test))


# In[ ]:


#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'g',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


# In[ ]:


#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'g',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)


# In[ ]:


i = np.random.randint(0,x_test.shape[0])
p = model.predict(np.array([x_test[i]]))
p = np.argmax(p, axis =-1)
y_true = np.argmax(np.array(y_test), axis =-1)[i]
print("{:15}{:5}\t {} \n".format("Word","True","Pred"))
print("-"*30)
for w,true,pred in zip(x_test[i],y_true,p[0]):
    print("{:15}{}\t{}".format(words[w-1],tags[true],tags[pred]))


# In[ ]:




