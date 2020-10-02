#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.datasets import reuters

(train_data , train_labels) , (test_data , test_labels) = reuters.load_data(num_words = 10000)


# In[ ]:


word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in
train_data[0]])


# In[ ]:


import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)


# In[ ]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64 , activation = 'relu' , input_shape = (10000,)))
model.add(layers.Dense(64 , activation = 'relu' ))
model.add(layers.Dense(46 , activation = 'softmax'))


# In[ ]:


model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


# In[ ]:


x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


# In[ ]:


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 9,
                    batch_size = 512,
                    validation_data=(x_val, y_val))


# In[ ]:


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1 , len(loss) + 1)

plt.plot(epochs , loss , 'bo' , label = 'Training loss' )
plt.plot(epochs , val_loss, 'b' , label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()
plt.rc('font', size = 15)
plt.rc('figure', figsize=[10,10])
plt.show()


# In[ ]:



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1 , len(acc) + 1)

plt.plot(epochs , acc , 'bo' , label = 'Training accuracy' )
plt.plot(epochs , val_acc, 'b' , label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')

plt.legend()
plt.rc('font', size = 15)
plt.rc('figure', figsize=[10,10])
plt.show()


# In[ ]:


results = model.evaluate(x_test, one_hot_test_labels)
results


# In[ ]:


predictions = model.predict(x_test)
print(predictions[0].shape,'***',
np.sum(predictions[0]),'***',
np.argmax(predictions[0]))
