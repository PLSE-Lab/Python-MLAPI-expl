#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from mlxtend.plotting import plot_confusion_matrix


# In[ ]:


train = pd.read_csv("../input/Kannada-MNIST/train.csv")
test = pd.read_csv("../input/Kannada-MNIST/test.csv")
sample_sub = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
dig = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")


# In[ ]:


X_train = train.iloc[:, 1:].values.astype(np.float32)
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values.astype(np.float32)
id_test = test.iloc[:, 0].values

X_dig = dig.iloc[:, 1:].values.astype(np.float32)
y_dig = dig.iloc[:, 0].values


# Let's scale the values to 0 mean and 1 std, and reshape them to proper tensors. 

# In[ ]:


X_train = X_train / 255.0
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_dig = X_dig / 255.0
X_dig = X_dig.reshape(X_dig.shape[0], 28, 28, 1)


# Now we convert the target labels to one-hot encoding (for better NN processing). 

# In[ ]:


y_train_oh = to_categorical(y_train)
y_dig_oh = to_categorical(y_dig)


# 

# Let's create a rather simple CNN model. 

# In[ ]:


np.random.seed(420)

model = Sequential()

model.add(Conv2D(32, input_shape=(28, 28, 1), kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))


# In[ ]:


model.summary()


# We can now train the model, using the *Dig* dataset as validation data. This dataset is a bit different from the training data, so prediction performance will be far from optimal, but it can be useful to tweak the model and tune it for better results. 

# In[ ]:


early_stop = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train_oh, epochs=20, batch_size=32, validation_data=(X_dig, y_dig_oh),
                    callbacks=[early_stop], verbose=0)


# Let's check the performance of the model on training and evaluation data. 

# In[ ]:


plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["figure.facecolor"] = "white"

plt.subplot(1, 2, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', "dig"], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', "dig"], loc='upper left')

plt.show()


# It seems like the model might need some more adjustments to avoid overfitting, but the test set should be more similar to the training data than the Dig set, so for the time being I will stick with it and try to predict labels. 

# In[ ]:


y_pred_oh = model.predict(X_test)
y_pred = np.argmax(y_pred_oh, axis=1)


# In[ ]:


df_pred = pd.DataFrame({"id": id_test, "label": y_pred})
df_pred.head()


# In[ ]:


df_pred.to_csv("subm_v1.csv", index=False, header=True)

