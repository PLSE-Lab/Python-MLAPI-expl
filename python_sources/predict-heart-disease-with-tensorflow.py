#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow-gpu==2.0.0-alpha')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np 
import warnings
import os

print(os.listdir("../input"))


# In[ ]:


warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv("../input/framingham.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


columns = data.columns


# In[ ]:


for col in columns:
    print(col, sum(pd.isnull(data[col])))


# In[ ]:


imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imp.fit(data)

new_data = imp.transform(data)


# In[ ]:


data = pd.DataFrame(new_data, columns=columns)


# In[ ]:


data.head()


# In[ ]:


for col in columns:
    print(col, sum(pd.isnull(data[col])))


# In[ ]:


data["TenYearCHD"] = data["TenYearCHD"].map(lambda x: int(x))


# In[ ]:


sns.countplot(data["TenYearCHD"])
plt.title("Labels")
plt.show()


# In[ ]:


data.hist(figsize=(20, 20))
plt.show()


# In[ ]:


labels = data.pop("TenYearCHD").values
data = data.values


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data ,labels,
                                                    test_size=.1,
                                                    random_state=5)


# In[ ]:


print("shape x_train:", x_train.shape)
print("shape y_train:", y_train.shape)
print("shape x_test:", x_test.shape)
print("shape y_test:", y_test.shape)


# In[ ]:


def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)


# In[ ]:


x_train = min_max_normalized(x_train)
x_test = min_max_normalized(x_test)


# In[ ]:


x_train = tf.cast(x_train, dtype=tf.float32)
y_train = tf.cast(y_train, dtype=tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)


x_test = tf.cast(x_test, dtype=tf.float32)
y_test = tf.cast(y_test, dtype=tf.float32)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)


# In[ ]:


optimizer = tf.keras.optimizers.Adam(lr=0.001)


# In[ ]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(16, activation='relu', input_shape=(15,)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation='softmax')
])


# In[ ]:


model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


H = model.fit(train_dataset, epochs=10, validation_data=test_dataset)


# In[ ]:


plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()


# In[ ]:




