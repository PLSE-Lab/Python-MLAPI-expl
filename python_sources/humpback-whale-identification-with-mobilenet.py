#!/usr/bin/env python
# coding: utf-8

# # Humpback Whale Identification with MobileNet
# * This kernel is based on 
# <br>@Ankit : https://www.kaggle.com/satian/keras-mobilenet-starter, which is a combination of 
# <br>@peter : https://www.kaggle.com/pestipeti/keras-cnn-starter and 
# <br>@beluga: https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892 from google doodle quickdraw competition

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


# In[ ]:


os.listdir("../input/")


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
train_df.head()


# In[ ]:


def prepareImages(data, m, dataset):
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


X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255


# In[ ]:


y, label_encoder = prepare_labels(train_df['Id'])


# In[ ]:


y.shape


# In[ ]:


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# In[ ]:


model = MobileNet(input_shape=(100, 100, 3), alpha=1., weights=None, classes=5005)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
print(model.summary())


# In[ ]:


# Ankit's kernel ran for 100 epochs
history = model.fit(X, y, epochs=50, batch_size=100, verbose=1)


# In[ ]:


plt.plot(history.history['categorical_accuracy'])
plt.title('Model categorical accuracy')
plt.ylabel('categorical accuracy')
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


X = prepareImages(test_df, test_df.shape[0], "test")
X /= 255


# In[ ]:


predictions = model.predict(np.array(X), verbose=1)


# In[ ]:


len(predictions)


# In[ ]:


for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))


# In[ ]:


test_df.head(10)
test_df.to_csv('submission.csv', index=False)


# Here's a look at the success of a few variations of this kernel on the public test data.
# 
# @Ankit's kernel yields a score of:<br>
# * 0.313 unaltered<br>
# * 0.301 with fitting divided into two 50-epoch calls<br>
# * 0.321 with only 50 epochs total<br>
# 
# The differences between the first two versions can be attributed to [this issue](https://github.com/rstudio/keras/issues/415) with multiple fit calls.<br><br>
# With fewer epochs leading to a higher score, it seems that overfitting hurt the performance with 100 epochs. However, with only a minor portion of the test data used to obtain these scores, the 3rd option could turn out to be less robust overall.  
