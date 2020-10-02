#!/usr/bin/env python
# coding: utf-8

# # Whale Identification using CNN

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
from IPython.display import Image

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


# In[ ]:


os.listdir("../input")


# In[ ]:


# load image data
df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


df.tail()


# Now let's open one of the images in the training set to see how they look like.

# In[ ]:


#show sample image
Image(filename="../input/train/"+random.choice(df['Image'])) 


# In[ ]:


# lets find total number of different whales present
print(f'Training examples: {len(df)}')
print("Unique whales: ",df['Id'].nunique()) # it includes new_whale as a separate type.


# In[ ]:


training_pts_per_class = df.groupby('Id').size()
training_pts_per_class


# In[ ]:


print("Min example a class can have: ",training_pts_per_class.min())
print("0.99 quantile: ",training_pts_per_class.quantile(0.99))
print("Max example a class can have: \n",training_pts_per_class.nlargest(5))    


# In[ ]:


data = training_pts_per_class.copy()
data.loc[data > data.quantile(0.99)] = '22+'
plt.figure(figsize=(15,10))
sns.countplot(data.astype('str'))
plt.title("#classes with different number of images",fontsize=15)
plt.show()


# In[ ]:


# new_whales is addes as a new class.
# above graph shows that there are more than 2000 classes with just one training example.
# and around 1300 classes with 2 training examples.
# it also shows that around 50 classes have more than 22 training examples.

data


# The next set of code is meant to prepare the images to be used for the training. It changes their shape and converts it into an array.

# In[ ]:


def prepare_images(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train


# Preparing the labels, by converting them into one-hot vectors.

# In[ ]:


def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder


# In[ ]:


X = prepare_images(df, df.shape[0], "train")
X /= 255


# In[ ]:


y, label_encoder = prepare_labels(df['Id'])


# In[ ]:


model = Sequential()

model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3), name='avg_pool'))

model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(y.shape[1], activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(X, y, epochs=100, batch_size=100, verbose=1)
gc.collect()


# In[ ]:


plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()


# In[ ]:


test = os.listdir("../input/test/")
print(len(test))


# In[ ]:


col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''


# In[ ]:


X_test = prepare_images(test_df, test_df.shape[0], "test")
X_test /= 255


# In[ ]:


predictions = model.predict(np.array(X_test), verbose=1)


# In[ ]:


for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))


# In[ ]:




test_df.head(10)
test_df.to_csv('submission.csv', index=False)

