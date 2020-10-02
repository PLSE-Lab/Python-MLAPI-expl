#!/usr/bin/env python
# coding: utf-8

# **WARNING: notebook under construction :)**  
# # GAPCV - Gray Images - Expand dim for Keras

# ## Install GAPCV and import libraries

# In[ ]:


get_ipython().run_cell_magic('capture', '', '#install GapCV\n!pip install -q gapcv==1.0rc4')


# In[ ]:


get_ipython().system('pip show gapcv')


# In[ ]:


import os
import time
import cv2
import gc
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks

import gapcv
from gapcv.vision import Images

from sklearn.utils import class_weight

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('tensorflow version: ', tf.__version__)
print('keras version: ', tf.keras.__version__)
print('gapcv version: ', gapcv.__version__)

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

os.makedirs('model', exist_ok=True)
print(os.listdir('../input'))
print(os.listdir('./'))


# ## utils function

# In[ ]:


def plot_sample(imgs_set, labels_set, img_size=(12,12), columns=4, rows=4, random=False):
    """
    Plot a sample of images
    """
    
    fig=plt.figure(figsize=img_size)
    
    for i in range(1, columns*rows + 1):
        if random:
            img_x = np.random.randint(0, len(imgs_set))
        else:
            img_x = i-1
        img = imgs_set[img_x]
        ax = fig.add_subplot(rows, columns, i)
        ax.set_title(str(labels_set[img_x]))
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE))
    plt.show()


# In[ ]:


def plot_history(history, val_1, val_2, title):
    plt.plot(history.history[val_1])
    plt.plot(history.history[val_2])

    plt.title(title)
    plt.ylabel(val_1)
    plt.xlabel('epoch')
    plt.legend([val_1, val_2], loc='upper left')
    plt.show()


# ### Get a sample of the dataset

# In[ ]:


dataset_name = 'wildlife128'
wildlife_filter = ('black_bear', 'bald_eagle', 'cougar', 'elk', 'gray_wolf')

## create two list to use as paramaters in GapCV
images_list = []
classes_list = []
for folder in os.scandir('../input/oregon-wildlife/oregon_wildlife/oregon_wildlife'):
    if folder.name in wildlife_filter:
        for image in os.scandir(folder.path):
            images_list.append(image.path)
            classes_list.append(image.path.split('/')[-2])


# ## Image preprocessing with GAPCV

# In[ ]:


## GapCV
images = Images(
    dataset_name,
    images_list,
    classes_list,
    config=[
        'resize=(128,128)',
        'gray-expand_dim',
        'stream',
        'verbose'
    ]
)


# In[ ]:


print(f'images preprocessed: {images.count}. Time elapsed: {images.elapsed}')


# ### Load images in stream mode to use data augmentation (optional)

# In[ ]:


del images
images = Images(
    config=['stream', 'gray-expand_dim'],
    augment=[
        'flip=horizontal',
        'zoom=0.4',
        'denoise'
    ]
)
images.load(dataset_name)
print(f'{dataset_name} dataset ready for streaming')


# In[ ]:


print('content:', os.listdir("./"))
print('number of images in data set:', )
print('classes:', images.classes)
print('data type:', images.dtype)


# Setup generator and get a batch of gray scale images with a expanded dimension to use in `keras`: `(32, 128, 128)` -> `(32, 128, 128, 1)`

# In[ ]:


minibatch_size = 32
images.split = 0.2
X_test, Y_test = images.test
images.minibatch = minibatch_size
gap_generator = images.minibatch
X_train, Y_train = next(gap_generator)

print('(minibatch_size, width, height, channel)')
print(X_train.shape)


# ## Show images sample with labels

# In[ ]:


plot_sample(X_train, Y_train, random=True)


# ## Handling un-balance classes

# In[ ]:


Y_int = [y.argmax() for y in Y_test]
class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(Y_int),
    Y_int
)

total_train_images = images.count - len(X_test)
n_classes = len(images.classes)


# ## Starting a Keras model

# In[ ]:


model = Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Dropout(0.2),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Dropout(0.2),
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Dropout(0.2),
    layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])


# In[ ]:


model.summary()


# ### Compile model

# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ### Callbacks

# In[ ]:


model_file = './model/model.h5'
earlystopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5
)

model_checkpoint = callbacks.ModelCheckpoint(
    model_file,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max'
)


# ### Train the model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit_generator(\n    generator=gap_generator,\n    validation_data=(X_test, Y_test),\n    epochs=50,\n    steps_per_epoch=int(total_train_images / minibatch_size),\n    initial_epoch=0,\n    verbose=1,\n    class_weight=class_weights,\n    callbacks=[\n        # earlystopping,\n        model_checkpoint\n    ]\n)')


# ### Performance charts ( overfitting... :( )

# In[ ]:


plot_history(history, 'accuracy', 'val_accuracy', 'Accuracy')
plot_history(history, 'loss', 'val_loss', 'Loss')


# ### Load the best model saved

# In[ ]:


del model
model = load_model(model_file)


# ### Evaluate the model againts test data set

# In[ ]:


get_ipython().run_cell_magic('capture', '', '# captured to hide kaggle bug\nscores = model.evaluate(X_test, Y_test, batch_size=32)')


# In[ ]:


for score, metric_name in zip(scores, model.metrics_names):
    print(f'{metric_name} : {score}')


# ## Inference

# ### Get a wildlife image randomly from Google

# In[ ]:


get_ipython().system('curl https://d36tnp772eyphs.cloudfront.net/blogs/1/2016/11/17268317326_2c1525b418_k.jpg > test_image.jpg')


# ### Preprocess the image with GAPCV for inference

# In[ ]:


labels = {val:key for key, val in images.classes.items()}
labels


# In[ ]:


image2 = Images('foo', ['test_image.jpg'], [0], config=['resize=(128,128)', 'gray-expand_dim'])
img = image2._data[0]


# ### Predicted image

# In[ ]:


prediction = model.predict_classes(img)
prediction = labels[prediction[0]]

plot_sample(img, [f'predicted image: {prediction}'], img_size=(8, 8), columns=1, rows=1)

