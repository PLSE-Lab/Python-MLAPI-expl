#!/usr/bin/env python
# coding: utf-8

# # Humpback Whale Identification
# 
# ## CNN (Keras) with `new_whale` threeshold
# 
# Notebook adapted from https://www.kaggle.com/pestipeti/keras-cnn-starter

# In[ ]:


import gc
import os

import numpy as np
import pandas as pd
import progressbar

import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


# In[ ]:


def prepareImages(data, m, dataset):
    #print("Preparing images")
    #print(m)
    X_train = np.zeros((m, 128, 128, 3))
    count = 0
    
    for fig in progressbar.progressbar(data['Image']):
        # load images into images of size 128x128x3
        img = image.load_img("../input/humpback-whale-identification/"+dataset+"/"+fig, target_size=(128, 128, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train[count] = x
        count += 1
    
    return X_train


# In[ ]:


def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)

    y = onehot_encoded
    #print(y.shape)
    return y, label_encoder


# In[ ]:


train = os.listdir("../input/humpback-whale-identification/train/")
print(len(train))


# In[ ]:


train = pd.read_csv("../input/humpback-whale-identification/train.csv")
train.Id.value_counts().head()


# ---

# In[ ]:


# From https://www.kaggle.com/suicaokhoailang/removing-class-new-whale-is-a-good-idea
if not os.path.isfile("../input/cnn-cnn-without-new-whale-class/keras-cnn-starter-without-new-whales.json"):
    train_df = train[train['Id'] != 'new_whale']
    train_df.Id.value_counts().head()


# Maybe a better idea would be to still integrate `new_whale` data, but rather as a vector `np.zeros(...)` (with `softmax` activation) than a separate class.

# In[ ]:


if not os.path.isfile("../input/cnn-cnn-without-new-whale-class/keras-cnn-starter-without-new-whales.json"):
    X = prepareImages(train_df, train_df.shape[0], "train")
    X /= 255


# In[ ]:


if not os.path.isfile("../input/cnn-cnn-without-new-whale-class/keras-cnn-starter-without-new-whales.json"):
    y, label_encoder = prepare_labels(train_df['Id'])
    y.shape


# In[ ]:


# Free Memory! Free Memory!
if not os.path.isfile("../input/cnn-cnn-without-new-whale-class/keras-cnn-starter-without-new-whales.json"):
    del train_df
    gc.collect()


# ## Train our model

# In[ ]:


if not os.path.isfile("../input/cnn-cnn-without-new-whale-class/keras-cnn-starter-without-new-whales.json"):
    model = Sequential()

    model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (128, 128, 3)))

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
    model.summary()


# In[ ]:


if not os.path.isfile("../input/cnn-cnn-without-new-whale-class/keras-cnn-starter-without-new-whales.json"):
    history = model.fit(X, y, epochs=100, batch_size=100, verbose=1)


# In[ ]:


# Free Memory! Free Memory!
if not os.path.isfile("../input/cnn-cnn-without-new-whale-class/keras-cnn-starter-without-new-whales.json"):
    del X, y
    gc.collect()


# In[ ]:


if not os.path.isfile("../input/cnn-cnn-without-new-whale-class/keras-cnn-starter-without-new-whales.json"):
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()


# ## Serialize Keras model to `.json`
# 
# From https://keras.io/models/about-keras-models/

# In[ ]:


if not os.path.isfile("../input/cnn-cnn-without-new-whale-class/keras-cnn-starter-without-new-whales.json"):
    with open("../input/cnn-cnn-without-new-whale-class/keras-cnn-starter-without-new-whales.json", "w") as f:
        f.write(model.to_json())


# ---

# In[ ]:


from keras.models import model_from_json

with open('../input/cnn-cnn-without-new-whale-class/keras-cnn-starter-without-new-whales.json', 'r') as f:
    model = model_from_json(f.read())


# ## `MAP@5` Score computation function
# 
# From https://www.kaggle.com/pestipeti/explanation-of-map5-scoring-metric

# In[ ]:


def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """    
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


# In[ ]:


def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l,p in zip(labels, predictions)])


# ## Fit `new_whale` threeshold

# In[ ]:


X = prepareImages(train, train.shape[0], "train")
X /= 255


# In[ ]:


y, label_encoder = prepare_labels(train['Id'])


# In[ ]:


predictions_encoded = model.predict(np.array(X), verbose=1)


# In[ ]:


# Free Memory! Free Memory!
del X, y
gc.collect()


# ---

# In[ ]:


best_pred = max(predictions_encoded.flatten())
best_pred # TODO ???!!! should not be < THREESHOLD


# In[ ]:


worst_pred = min(predictions_encoded.flatten())
worst_pred


# ---

# In[ ]:


# Function that's assign class "new_whale" to encoders with low threeshold
def get_top5(treeshold, pred):
    args5 = pred.argsort()[-5:][::-1]
    classes5 = [i for i in label_encoder.inverse_transform(args5)]
    for i, t in enumerate(args5):
        if pred[t] < treeshold:
            for j in range(i + 1, 5):
                classes5[j] = classes5[j - 1]
            classes5[i] = "new_whale"
            break
    return classes5


# In[ ]:


X_ = []
y_ = []

def get_score(treeshold):
    print("get_score(%s) = " % treeshold, end="")
    predictions = []
    for i, pred in enumerate(predictions_encoded):
        predictions.append(get_top5(treeshold, pred))
    result = map_per_set(train['Id'].values, predictions)
    print(result)
    X_.append(treeshold)
    y_.append(result)
    return result


# ---

# In[ ]:


# From https://www.scipy-lectures.org/advanced/mathematical_optimization/#getting-started-1d-optimization
from scipy import optimize


# In[ ]:


# "new_whale" threeshold -> will converge to optimal split threeshold
result = optimize.minimize_scalar(lambda x: - get_score(x), bounds=(worst_pred, best_pred), method='bounded')  # -SCORE (we try to minimize the function)


# In[ ]:


plt.plot(X_, y_, '-')
plt.show()


# In[ ]:


new_whale_treeshold = result.x
new_whale_treeshold # == best_pred :(


# ---

# In[ ]:


train_new_whale = train[train['Id'] == 'new_whale']
train_new_whale.Id.value_counts().head()


# In[ ]:


X = prepareImages(train_new_whale, train_new_whale.shape[0], "train")
X /= 255


# In[ ]:


predictions_encoded_new_whale = model.predict(np.array(X), verbose=1)


# In[ ]:


np.mean(predictions_encoded_new_whale.flatten())


# In[ ]:


# Free Memory! Free Memory!
del train, train_new_whale
gc.collect()


# ## Predict labels of test dataset

# In[ ]:


test = os.listdir("../input/humpback-whale-identification/test/")
print(len(test))


# In[ ]:


col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''


# In[ ]:


X = prepareImages(test_df, test_df.shape[0], "test")
X /= 255


# In[ ]:


predictions_encoded = model.predict(np.array(X), verbose=1)


# In[ ]:


for i, pred in enumerate(predictions_encoded):
    test_df.loc[i, 'Id'] = ' '.join(get_top5(new_whale_treeshold, pred))


# In[ ]:


test_df.head(10)
test_df.to_csv('keras-cnn-with-new-whale-threeshold.csv', index=False) #> Score = 0.286


# In[ ]:


# Free Memory! Free Memory!
del test_df, X
gc.collect()


# In[ ]:


#!kaggle competitions submit -c humpback-whale-identification -f "keras-cnn-with-new-whale-threeshold.csv" -m "CNN with Keras with new_whale (threeshold = 0.00022381447120760044)"

