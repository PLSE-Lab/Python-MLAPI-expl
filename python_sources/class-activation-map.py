#!/usr/bin/env python
# coding: utf-8

# In[ ]:


rm ./celeba_clf.keras


# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile, sys, os
sys.path.insert(0, os.path.abspath('..'))

# CNN
import keras
from keras.models import Sequential, Model, load_model 
from keras.layers import Dense, Dropout, Flatten, Activation

from keras.constraints import maxnorm
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
np.random.seed(3)

# plot function
from skimage import feature, transform
import matplotlib as mp
import matplotlib.pyplot as plt

#   hyper params
CLASS_NUM=2
EPOCHS = 3
BATCH_SIZE = 32
LR = 0.01

IMG_ROW = 218
IMG_COL = 178
CH = 3

DATA_PATH = '../input'
IMG_PATH = os.path.join(DATA_PATH,'img_align_celeba', 'img_align_celeba')
CSV_PATH = os.path.join(DATA_PATH,'list_attr_celeba.csv')
TRAIN_CNT = 10000
TEST_CNT  = 1000

ATTR_SHOW = 8

def data_preprocess():
    data = pd.read_csv(CSV_PATH)[['image_id', 'Male']]
    sample_data = data.sample(frac=1).reset_index(drop=True)

    test_dataframe = sample_data[:TEST_CNT]
    train_dataframe = sample_data[TEST_CNT:TRAIN_CNT]

    train_dataframe = train_dataframe.reset_index(drop=True)

    test_dataframe.to_csv('./test_data.csv', encoding='utf-8')
    train_dataframe.to_csv('./train_data.csv', encoding='utf-8')

def load_gen(train_df, test_df):    
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_dataframe(dataframe=train_df, directory=IMG_PATH,
                                            x_col='image_id', y_col='Male',
                                            has_ext=True, class_mode="categorical",
                                            target_size=(IMG_ROW, IMG_COL), batch_size=BATCH_SIZE)

    test_gen = datagen.flow_from_dataframe(dataframe=test_df, directory=IMG_PATH,
                                           x_col='image_id', y_col='Male',
                                           has_ext=True, class_mode="categorical",
                                           target_size=(IMG_ROW, IMG_COL), batch_size=BATCH_SIZE)

    return train_gen, test_gen

def createLayers():
    model = Sequential()
    model.add(Conv2D(filters=32,
                     kernel_size=5,
                     input_shape=(IMG_ROW, IMG_COL, CH),
                     padding='same',
                     activation='relu',
                     kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(CLASS_NUM, activation='softmax'))

    decay = LR / EPOCHS
    sgd = SGD(lr=LR, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model

def train(train_gen, test_gen, step_size_train, step_size_valid):
    model = createLayers()
    model.fit_generator(generator=train_gen,
                    steps_per_epoch=step_size_train,
                    validation_data=test_gen,
                    validation_steps=step_size_valid,
                    epochs=EPOCHS)

    model.save('celeba_clf.keras')

data_preprocess()

test_df = pd.read_csv('./test_data.csv')
train_df = pd.read_csv('./train_data.csv')

# the data, shuffled and split between train and test sets
train_gen, test_gen = load_gen(train_df, test_df)

step_size_train = train_gen.n // train_gen.batch_size
step_size_valid = test_gen.n // test_gen.batch_size

if not os.path.isfile("./celeba_clf.keras"):
    train(train_gen, test_gen, step_size_train, step_size_valid)

model = load_model("./celeba_clf.keras")

r_batch = np.random.randint(low=0, high=len(test_gen))
x_test_batch, y_test_batch = test_gen[r_batch]

show_cnt = ATTR_SHOW if ATTR_SHOW < BATCH_SIZE else BATCH_SIZE

x_m_input = x_test_batch[:show_cnt]
last_conv_layer = model.layers[-5] # last layer until FC (Dense).

# Plot
n_cols = 4
n_rows = int(show_cnt / 2)
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3*n_cols, 3*n_rows))
fig_loop = 0
for x in x_m_input:
    row, col= divmod(fig_loop,2)
    x_extend = np.expand_dims(x, axis=0)
    preds = model.predict(x_extend)

    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    
    
    grads = K.gradients(class_output, last_conv_layer.output)[0] # get gradients
    pooled_grads = K.mean(grads, axis=(0, 1, 2)) # get average value
    # NOTE: 1 batch = 1 data in decoding time
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x_extend])

    # loop # of filters.(See conv.func)
    for i in range(32):
        # NOTE: This code prevents 'divided by zero' problem
        # If you want, you can comment out this codes
        filter_mat = np.zeros(dtype=np.float32, shape=conv_layer_output_value.shape[0:2])
        filter_mat = conv_layer_output_value[:, :, i] * pooled_grads_value[i]
        if np.maximum(np.sum(filter_mat),0).all() == 0: 
            conv_layer_output_value[:, :, i] = 0 # deactivate filter.
            continue 
        # comment END
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i] # a^c_k * A^k
        
    heatmap = np.sum(conv_layer_output_value, axis=-1) # sigma_{k}
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)        # normalize
    
    # resize heatmap
    heatmap = cv2.resize(heatmap, (IMG_COL, IMG_ROW))
    x = (x*255).astype(np.uint8)
    axes[row, col*2].imshow(x)
    axes[row, col*2].axis('off')
    heatmap = np.uint8(255 * heatmap)
    axes[row, col*2+1].imshow(heatmap, cmap='gist_heat_r')
    axes[row, col*2+1].axis('off')
    fig_loop += 1
    
fig.savefig('full_figure.png')

