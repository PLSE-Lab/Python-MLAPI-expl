#!/usr/bin/env python
# coding: utf-8

# # Example for my blog post [How To Avoid Overfitting](https://medium.com/@gokhang1327/how-to-avoid-overfitting-for-beginners-deep-learning-ed2817a7e65)

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


# ## Reading data into a pandas dataframe

# In[ ]:


df = pd.read_csv("/kaggle/input/dont-overfit-ii/train.csv", index_col="id")


# ## Checking missing values

# In[ ]:


df.info()


# There is not any missing value situation in this data.

# ## Train-test split

# In[ ]:


X = df.drop(columns=["target"])
y = df.filter(["target"])

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42
)


# 80% train, 20% test

# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## Model

# In[ ]:


input_layer = tf.keras.Input(shape=(300,))

hidden_layer_1 = tf.keras.layers.Dense(1024, activation='relu')(input_layer)

hidden_layer_2 = tf.keras.layers.Dense(512, activation='relu')(hidden_layer_1)

hidden_layer_3 = tf.keras.layers.Dense(256, activation='relu')(hidden_layer_2)

output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer_3)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)


# In[ ]:


model.summary()


# ## Train

# In[ ]:


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=100,
    validation_split=0.25
)


# ## Evaluation

# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# This is the example of overfitting.

# ## How To Avoid Overfitting

# In[ ]:


input_layer = tf.keras.Input(shape=(300,))

hidden_layer_1 = tf.keras.layers.Dense(32, activation='relu')(input_layer)  # Simplifying model: less layers, less neurons
hidden_layer_1 = tf.keras.layers.Dropout(rate=0.2)(hidden_layer_1)  # Adding dropout layers

hidden_layer_2 = tf.keras.layers.Dense(16, activation='relu')(hidden_layer_1)  # Simplifying model: less layers, less neurons
hidden_layer_2 = tf.keras.layers.Dropout(rate=0.2)(hidden_layer_2)  # Adding dropout layers

output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer_2)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3
)

history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=100,
    callbacks=[es_callback],  # Stop training process earlier
    validation_split=0.25
)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# You can see that these methods decrease overfitting. But ofcourse not enough because this train set is too small to make this competition harder.
