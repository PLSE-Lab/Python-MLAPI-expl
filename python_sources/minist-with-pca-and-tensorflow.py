#!/usr/bin/env python
# coding: utf-8

# ## Image Classification with dimentionality reduction
#     * Load data and visualize it 
#     * Create Scikit learn data pipline
#     * Create tensorflow dataset
#     * Create tensorflow model
#     * Create Callbacks for the model
#     * Train model
#     * Check how model is learning during training
#     * Visualize model performance with tensorboard
#     * Do inferance on created model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import functools
import tensorflow as tf
import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# tf.enable_eager_execution()

# Any results you write to the current directory are saved as output.


# In[ ]:


tf.__version__


# In[ ]:


TRAIN_PATH = '../input/digit-recognizer/train.csv'
TEST_PATH = '../input/digit-recognizer/test.csv'
BATCH_SIZE = 1000


# ## Load data
# Load the input data with pandas and seperate labels from it. It good to have training set and validation dataset, so with scikit learn ```train_test_split``` we are creating training and validation dataset with 80, 20 percent for input training data.

# In[ ]:


train_df = pd.read_csv(TRAIN_PATH)
train_df.shape


# In[ ]:


train_df.info()


# In[ ]:


labels = train_df['label']
train_df.drop('label', axis=1, inplace=True)


# In[ ]:


train_df.columns


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_df.values, labels.values, test_size=0.2, random_state=42)
# X_train = train_df
# y_train = labels


# ## Visualize data
# Since we are dealing with images let's plot some images from dataset

# In[ ]:


X_img = X_train
X_img.shape


# In[ ]:


X_img = X_img.reshape(X_img.shape[0], 28, 28)


# In[ ]:


pyplot.imshow(X_img[0])
pyplot.title(y_train[0])


# ## Creating data pipline
# In this dataset we have pixel densities as training data ranging from 0-255. For better performance of our model let's do scaling first and then we will apply dimentionality reduction.Since we need to apply dimentionality reduction to all of the data, lets create a pipeline todo the same. Here we are using scikit learn's PCA to do the same. 95% of variance will give upto 153 dimentions, so let's take 169 dimentions which will be helpful for visualization as well. Using transform method of estimators, apply inverse trasform of PCA data, to get an approximation of training dataset.

# In[ ]:


pca = PCA(169)
class SqueezeDataset(BaseEstimator, TransformerMixin):
    def __init__(self, expected_variance_ratio = 0.95):
        self.variance_ratio = expected_variance_ratio
        self.pca = pca
    
    def fit(self, X, y=None):
        self.pca.fit(X)
        return self
    
    def transform(self, X, y=None):
        X_transformed = self.pca.transform(X)
        return X_transformed


# Combining both estimator to create a full pipeline and transforming data

# In[ ]:


train_data_pipeline = Pipeline([
    ('scale_dataset', StandardScaler()),
    ('squeeze_dataset', SqueezeDataset())
])


# In[ ]:


X_processed = train_data_pipeline.fit_transform(X_train)


# In[ ]:


X_processed.shape


# Appying inverse transform to get an approximation of given training data after PCA

# In[ ]:


x_inverse_trans = pca.inverse_transform(X_processed)


# In[ ]:


x_inverse_trans = x_inverse_trans.reshape(x_inverse_trans.shape[0], 28, 28)


# In[ ]:


x_inverse_trans.shape


# Visialize approximation of training data

# In[ ]:


fig, ax = pyplot.subplots(1, 3)
ax[0].imshow(x_inverse_trans[0])
ax[0].set_title('inverse transform')
ax[1].imshow(X_processed[0].reshape(13, 13))
ax[1].set_title('processed image')
ax[2].imshow(X_img[0])
ax[2].set_title('original')


# ## Create tensorflow dataset
# With the approximation we have, let's create a dataset and an iterator, which can do shuffling, batching on out dataset.
# Create a one shot iterator to iterate over data.

# In[ ]:


def prepare_dataset(features, labels):
    print(features.shape)
    dataset = (
        tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(len(labels))
            .repeat()
            .batch(BATCH_SIZE)
            .prefetch(1)
    )
    iterator = dataset.make_one_shot_iterator()
    return iterator


# Declare some layers with common variables to avoid repeating of code.

# In[ ]:


Input = tf.keras.Input
Conv2D = functools.partial(
        tf.keras.layers.Conv2D,
        activation='elu',
        padding='same'
    )
BatchNormalization = tf.keras.layers.BatchNormalization
Dense = tf.keras.layers.Dense
AveragePooling2D = tf.keras.layers.AveragePooling2D
Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten
L2 = functools.partial(
        tf.keras.regularizers.l2,
        l=0.2
    )


# ## Creating CNN with tensorflow

# In[ ]:


def prepare_model():
    input = Input(shape=(28,28,1,))
    conv1 = Conv2D(8, (3, 3))(input)
    conv2 = Conv2D(16, (3, 3))(conv1)
    batch_norm1 = BatchNormalization(axis=3)(conv2)
    conv3 = Conv2D(32, (5, 5))(batch_norm1)
    dropout = Dropout(0.3)(conv3)
#     avg_pool_1 = AveragePooling2D((2, 2))(dropout)
    conv4 = Conv2D(32, (5, 5))(dropout)
    batch_norm2 = BatchNormalization(axis=3)(conv4)
#     avg_pool_2 = AveragePooling2D(2, 2)(conv4)
    conv5 = Conv2D(16, (7, 7))(batch_norm2)
    conv6 = Conv2D(8, (7, 7))(conv5)
    flt1 = Flatten()(conv6)
    output = Dense(10, activation='softmax')(flt1)
    model = tf.keras.Model(input, output)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) # Since labels are not one hot encoded using just categorial_crossentropy dosen't make sence here.
    model.summary()
    return model


# Get iterator from prepare dataset function defined above for both training and validation datasets.

# In[ ]:


iterator = prepare_dataset(X_train.reshape(X_train.shape[0], 28, 28, 1), y_train)


# In[ ]:


val_iterator = prepare_dataset(X_test.reshape(X_test.shape[0], 28, 28, 1), y_test)


# In[ ]:


model = prepare_model()


# ## Callbacks to model
# Create callbacks to save model, visualize its performance with tensorboard, and for early stoping to avoid vanishing gradient problem

# In[ ]:


logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='../input/weights.hdf5', verbose=1, save_best_only=True),
    tensorboard_callback,
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
]


# ## Training model
# Train model using both training and valiadion iterator so that it will train and validate itself in every epoch. Set epochs to 50, if there is no improvement in learning for 3 epoch model will stop training as mentioned in callbacks.

# In[ ]:


history = model.fit(iterator, steps_per_epoch=34, epochs=50,  verbose=1, callbacks=callbacks,
                   validation_data=val_iterator, validation_steps=9)


# In[ ]:





# Visualize some testing data

# In[ ]:


pyplot.imshow(X_test[0].reshape(28, 28))
pyplot.title(y_test[0])


# ## Check how model is learning during training
# Let's check how model is learning during training. get outputs of 3rd layer from the model and try to visulaze it.

# In[ ]:


# Check how model is learning internally
inputs = model.input
outputs = [layer.output for layer in model.layers][3:]
functor = tf.keras.backend.function([inputs, tf.keras.backend.learning_phase()], outputs)


# In[ ]:


# Testing
test = X_test[0].reshape(1, 28, 28, 1)
layer_outs = functor([test, 1.])
print(layer_outs)


# In[ ]:


layer_outs[0].shape


# In[ ]:


sample = np.squeeze(layer_outs[0])
sample = np.transpose(sample)
sample[0].shape


# In[ ]:


pyplot.imshow(sample[10])


# In[ ]:





# In[ ]:





# ## Visualize model performance with tensorboard

# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard.notebook')
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[ ]:





# ## Inferance
# After all training and understanding how model learns and performs, let's do inferance on our model with test dataset. Load test dataset and send it to model to predict labels and write it to CSV file.

# In[ ]:


test_df = pd.read_csv(TEST_PATH)


# In[ ]:


test_df.shape


# In[ ]:


test_df = test_df.values.reshape(test_df.shape[0], 28, 28, 1)


# In[ ]:


test_df.shape


# In[ ]:


predictions = model.predict(test_df)


# In[ ]:


predictions = np.argmax(predictions, axis = 1)


# In[ ]:


submission_df = pd.concat([pd.Series(range(1, 28001), name="ImageId"), pd.Series(predictions, name="Label")], axis = 1)


# In[ ]:


submission_df.head(5)


# In[ ]:


pyplot.imshow(test_df[3].reshape(28, 28))


# In[ ]:


submission_df.to_csv("cnn_mnist_datagen.csv", index=False)

