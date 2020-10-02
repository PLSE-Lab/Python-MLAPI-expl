#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import json
import numpy as np


# <h1>Load and read dataset</h1>

# In[ ]:


def read_dataset(path):
    return json.load(open(path)) 

train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')


# <h1>Converting lists of ingredients to strings</h1>

# In[ ]:


def generate_text(data):
    text_data = [" ".join(doc['ingredients']).lower() for doc in data]
    return text_data 


# In[ ]:


train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]


# <h1>TF-IDF on text data</h1>

# In[ ]:


tfidf = TfidfVectorizer(binary=True)
def tfidf_features(txt, flag):
    if flag == "train":
        x = tfidf.fit_transform(txt)
    else:
        x = tfidf.transform(txt)
    x = x.astype('float16')
    return x 
X = tfidf_features(train_text, flag="train")
X_test = tfidf_features(test_text, flag="test")


# <h1>Converting the list of strings to the matrix of vectors</h1>
# (to be fed our nn)

# In[ ]:


lb = LabelEncoder()
y = lb.fit_transform(target)
y = keras.utils.to_categorical(y)


# <h1>Build a model</h1>
# <p>You can monkey with value of dropout to see how looks underfit/overfit (later). Also you can add other regularizers such as l1, l2.</p>
# <p>He initializer works better with relu.</p>

# In[ ]:


model = keras.Sequential()
model.add(keras.layers.Dense(1000, kernel_initializer=keras.initializers.he_normal(seed=1), activation='relu', input_dim=3010))
model.add(keras.layers.Dropout(0.81))
model.add(keras.layers.Dense(1000, kernel_initializer=keras.initializers.he_normal(seed=2), activation='relu'))
model.add(keras.layers.Dropout(0.81))
model.add(keras.layers.Dense(20, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=4), activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# <h1>Training the model</h1>
# <p>Don't forget to turn on GPU</p>

# In[ ]:


history = model.fit(X, y, epochs=20, batch_size=512, validation_split=0.1)
model.save_weights("model.h5")
print("Saved model to disk")


# <h1>Plotting learning curves</h1>
# Learning curves show us overting/underfiting

# In[ ]:


print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


predictions_encoded = model.predict(X_test)
predictions_encoded.shape


# <h1>Converting predicted vectors to names of cuisines</h1>

# In[ ]:


predictions = lb.inverse_transform([np.argmax(pred) for pred in predictions_encoded])
predictions


# <h1>Submission</h1>

# In[ ]:


test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': predictions}, columns=['id', 'cuisine'])
sub.to_csv('output.csv', index=False)


# In[ ]:




