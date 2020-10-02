#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd

import scikitplot
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import Callback, EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train_label = train_data['label']
train_data.drop('label', axis=1, inplace=True)

train_data.shape, test_data.shape


# So this is not the standard MNIST data with 60k train and 10k test images.

# In[ ]:


print(f'frequency of labels :\n{train_label.value_counts() / len(train_label)}')
train_label.value_counts().sort_index().plot.bar()
plt.show()


# In[ ]:


X_train_df, X_valid_df, y_train_s, y_valid_s = train_test_split(train_data, train_label,
                                                    shuffle=True, stratify=train_label,
                                                    test_size=0.2, random_state=42
                                            )
X_train_df.shape, X_valid_df.shape, y_train_s.shape, y_valid_s.shape


# In[ ]:


sample_train_img = X_train_df.iloc[0]
sample_valid_img = X_valid_df.iloc[0]

fig = plt.figure(0, (12, 4))

ax1 = plt.subplot(1,2,1)
ax1.imshow(np.array(sample_train_img).reshape(28, 28), cmap='gray')
ax1.set_title('train sample image')

ax2 = plt.subplot(1,2,2)
ax2.imshow(np.array(sample_valid_img).reshape(28, 28), cmap='gray')
ax2.set_title('validation sample image')

plt.show()


# In[ ]:


def data_preprocessing(df, to_normalize=True, **kwargs):
    X = df.astype('float32').values
    
    if to_normalize:
        X /= 255.
        
    if 'noise' in kwargs:
        rng = np.random.RandomState(42)
        X += kwargs['noise'] * rng.normal(loc=0.0, scale=1.0, size=X.shape)
        X = np.clip(X, 0., 1.)
        
    return X.reshape(-1, 28, 28, 1)


# In[ ]:


X_train = data_preprocessing(X_train_df)# noise=0.2) # Training on noisy data didn't helped
X_valid = data_preprocessing(X_valid_df)

X_train.shape, X_valid.shape


# In[ ]:


y_train = np_utils.to_categorical(y_train_s)
y_valid = np_utils.to_categorical(y_valid_s)

y_valid.shape, y_train.shape


# In[ ]:


img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]


# In[ ]:


def build_cnn():        
    net = Sequential()

    net.add(
        Conv2D(
            filters=32,
            kernel_size=(3,3),
            input_shape=(img_width, img_height, img_depth),
            activation='relu'
        )
    )
    net.add(
        Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation='relu'
        )
    )
    net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2,2)))
    net.add(Dropout(0.2))

    net.add(
        Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='relu'
        )
    )
    net.add(
        Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='relu'
        )
    )
    net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2,2)))
    net.add(Dropout(0.2))

    net.add(Flatten())
    
    net.add(Dense(128, activation='relu'))
    net.add(BatchNormalization())
    net.add(Dropout(0.4))
    
    net.add(Dense(num_classes, activation='softmax'))
    
    net.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    net.summary()
    
    return net


# In[ ]:


early_stopping = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    min_delta=0.0001,
    baseline=0.98,
    patience=6,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    factor=0.25,
    min_lr=1e-5
)

callbacks = [
    early_stopping,
    lr_scheduler,
    TerminateOnNaN(),
]


# In[ ]:


model = build_cnn()
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64,
    epochs=30,
    callbacks=callbacks,
    use_multiprocessing=True
)


# `The below plots clearly shows that we overfit the train data as the validation accuracy is considerably low.`

# In[ ]:


sns.set()
fig = plt.figure(0, (12, 4))

ax = plt.subplot(1, 2, 1)
sns.lineplot(history.epoch, history.history['accuracy'], label='train')
sns.lineplot(history.epoch, history.history['val_accuracy'], label='valid')
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.lineplot(history.epoch, history.history['loss'], label='train')
sns.lineplot(history.epoch, history.history['val_loss'], label='valid')
plt.title('Loss')
plt.tight_layout()

plt.show()


# In[ ]:


df_accu = pd.DataFrame({'train': history.history['accuracy'], 'valid': history.history['val_accuracy']})
df_loss = pd.DataFrame({'train': history.history['loss'], 'valid': history.history['val_loss']})

fig = plt.figure(0, (14, 4))
ax = plt.subplot(1, 2, 1)
sns.violinplot(x="variable", y="value", data=pd.melt(df_accu), showfliers=False)
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.violinplot(x="variable", y="value", data=pd.melt(df_loss), showfliers=False)
plt.title('Loss')
plt.tight_layout()

plt.show()


# In[ ]:


yhat_valid = model.predict_classes(X_valid)
scikitplot.metrics.plot_confusion_matrix(np.argmax(y_valid, axis=1), yhat_valid, figsize=(7,7))
print(f'total wrong validation predictions: {np.sum(np.argmax(y_valid, axis=1) != yhat_valid)}\n\n')
print(classification_report(np.argmax(y_valid, axis=1), yhat_valid))


# `Now we will use ImageDataGenerator to train on more images for better performance.`

# In[ ]:


train_datagen = ImageDataGenerator(
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
train_datagen.fit(X_train)

model = build_cnn()
history = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_valid, y_valid),
    steps_per_epoch=len(X_train) / 64,
    epochs=25,
    callbacks=callbacks,
    use_multiprocessing=True
)


# `The below plots shows that our model generalizes really well while having slightly better performance as earlier.`

# In[ ]:


fig = plt.figure(0, (12, 4))

ax = plt.subplot(1, 2, 1)
sns.lineplot(history.epoch, history.history['accuracy'], label='train')
sns.lineplot(history.epoch, history.history['val_accuracy'], label='valid')
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.lineplot(history.epoch, history.history['loss'], label='train')
sns.lineplot(history.epoch, history.history['val_loss'], label='valid')
plt.title('Loss')
plt.tight_layout()

plt.show()


# In[ ]:


df_accu = pd.DataFrame({'train': history.history['accuracy'], 'valid': history.history['val_accuracy']})
df_loss = pd.DataFrame({'train': history.history['loss'], 'valid': history.history['val_loss']})

fig = plt.figure(0, (14, 4))
ax = plt.subplot(1, 2, 1)
sns.violinplot(x="variable", y="value", data=pd.melt(df_accu), showfliers=False)
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.violinplot(x="variable", y="value", data=pd.melt(df_loss), showfliers=False)
plt.title('Loss')
plt.tight_layout()

plt.show()


# In[ ]:


yhat_valid = model.predict_classes(X_valid)
scikitplot.metrics.plot_confusion_matrix(np.argmax(y_valid, axis=1), yhat_valid, figsize=(7,7))
print(f'total wrong validation predictions: {np.sum(np.argmax(y_valid, axis=1) != yhat_valid)}\n\n')
print(classification_report(np.argmax(y_valid, axis=1), yhat_valid))


# `Now we will train on the entire train data.`

# In[ ]:


early_stopping = EarlyStopping(
    monitor='accuracy',
    mode='max',
    min_delta=0.0001,
    baseline=0.98,
    patience=6,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='accuracy',
    patience=3,
    factor=0.25,
    min_lr=1e-5
)

callbacks = [
    early_stopping,
    lr_scheduler,
    TerminateOnNaN(),
]

train_datagen = ImageDataGenerator(
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
train_datagen.fit(X_train)

train_data_pp = data_preprocessing(train_data)
train_label_pp = np_utils.to_categorical(train_label)

model = build_cnn()

# changing the steps as data is increased now.
history = model.fit_generator(
    train_datagen.flow(train_data_pp, train_label_pp, batch_size=32),
    steps_per_epoch=len(train_data_pp) / 32,
    epochs=25,
    callbacks=callbacks,
    use_multiprocessing=True
)


# In[ ]:


submission_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
X_test = data_preprocessing(test_data)
yhat_test = model.predict_classes(X_test)
submission_df['Label'] = yhat_test
submission_df.to_csv('my_submission.csv', index=False)


# Digit-MNIST is the most fundamental and easiest dataset possible for deep learning purposes. And we have seen that without much efforts we achieved almost 99.5% accuracy on this dataset (which is not even the standard 60k MNIST). Also with this same model and configuration we may have achieved even higher accuracy with the 60k MNIST.
# 
# Although we have achieved pretty good accuracy in such an easy dataset but for such we an easy dataset our model has parameters in the range of 100k-200k, which is way more for such an easy task. So can we do better i.e., reducing the model size significantly and simultaneously retaining the accuracy in the same range.
# 
# Recently I was given a task by some organization to achieve an accuracy of atleast 99.4% on the MNIST. You might think what's good about that, even handicapped models can achieve an accuracy of around 98-99% in MNIST without doing anything. But the constraint was to achieve such an high accuracy using a model having atmost 8k parameters. Although I wasn't able to touch the 99.4% bar but I got an best accuracy of around 99.2% having less than 8k parameters in the given time limit. And I was sure if the model have given enough time then 99.4% accuracy is achievable afterall no one design a task which is un-achievable. The organization may already achieved that and that's why given us such a task.
# 
# But the idea here is that before jumping to deep networks we first try to achieve the goal in minimal model possible. So using such low number of parameters we achieved almost the same accuracy. The benefits are many like easy to inspect, debug, load, train etc..
# 
# I thought it's worth sharing the notebook, it's very simple and I didn't do much. If you are interested [here](https://www.kaggle.com/gauravsharma99/mnist-8k-params/) is the link to the notebook.

# In[ ]:




