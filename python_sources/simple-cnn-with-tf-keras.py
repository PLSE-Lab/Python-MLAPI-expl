#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# # Import necessary modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers


# # Load and prepare the data

# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


train.head()


# As we can see, apart from **label**, dataset has $784$ more columns, which represnets pixle values of an $28$ x $28$ image.
# 
# Let's create `X_train` and `y_train` from `train`.

# In[ ]:


y_train = train['label']
X_train = train.drop(['label'], axis=1)


# It's time for some normalization and reshaping.

# In[ ]:


X_train /= 255.0
test /= 255.0


# In[ ]:


X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)


# Now, as we are ready with our data, it's time for defining and fitting a model. But before that, let's visualize what we have.

# # Visualization

# Following function plots $12$ images from the dataset in a $3$ x $4$ grid. Title of each image represents its corresponding label.

# In[ ]:


def plot(data, labels, title='Label'):
    plt.figure(figsize=(10, 9))
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.imshow(data[i][:,:,0])
        plt.title('{}: {}'.format(title, labels[i]))
        plt.axis('off');


# In[ ]:


plot(X_train, y_train)


# # Define the model

# In[ ]:


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding = 'Same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu', padding = 'Same'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[ ]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['acc'])


# In[ ]:


model.summary()


# # Fit the model

# In[ ]:


EPOCHS = 5


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(X_train, y_train.values,\n                   validation_split=.1,\n                   epochs=EPOCHS, batch_size=64,\n                   verbose=2)')


# Plot Accuracy vs. Loss!

# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[ ]:


epochs = range(EPOCHS)


# In[ ]:


plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.title('Training and Validation Loss')
plt.plot(epochs, loss, label='Training')
plt.plot(epochs, val_loss, label='Validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.title('Training and Validation Accuracy')
plt.plot(epochs, acc, label='Training')
plt.plot(epochs, val_acc, label='Validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()


# # Make predictions

# In[ ]:


get_ipython().run_cell_magic('time', '', 'results = model.predict(test)')


# In[ ]:


results = np.argmax(results, axis=1)


# Let's see some of our predictions.

# In[ ]:


plot(test, results, 'Predicted label')


# # Export predictions

# Convert `results` to **Pandas Series**.

# In[ ]:


results = pd.Series(results, name='Label')


# In[ ]:


submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)


# In[ ]:


submission.to_csv('digit_recognizer.csv', index=False)

