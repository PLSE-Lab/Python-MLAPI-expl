#!/usr/bin/env python
# coding: utf-8

# <h1>Transfer learning for dog breed classification - Part I</h1>
# 
# In every single computer vision competition, top rank competitors almost never train their networks from scratch. They transfer knowledge from pretrained models to theirs. This is something I painfully learned in the first official competition I took place in (TGS Salt Identification), and I wish someone would have told me then. For that reason, I'm planning to create a series of kernels for beginners in which I'll cover the basic techniques in transfer learning. I will only touch the problem of image classification, but these techniques could be applied to any other computer vision problem (image segmentation, object detection, ...). I will be using a pretrained model for the dog breed classification problem. At the same time, I will be covering some useful techniques which I also painfully learned about such as using <code>ImageDataGenerator</code> for reading images into memory from disk and data augmentation.
# 
# The series will have three parts, in each of which I will cover three basic techniques and some of its variants:
# 
# <olist>
#     <li>Part I - Extract features from the last convolutional block of VGG16 and use them to train a NN classifier, without data augmentation</li>
#     <li>Part II - Freeze the VGG16 base model, put a classifier on top of it and train the model with data augmentation.</li>
#     <li>Part III - Starting from the model trained in Part 2, unfreeze the last convolutional block and finetune the network with a small learning rate.</li>
# </olist>

# In[ ]:


get_ipython().system(' cp ../input/my-python/* ../working/')
get_ipython().system(' ls ../working ')


# In[ ]:


# Generic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Keras imports
from keras.applications import VGG16, InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.optimizers import RMSprop, Adam, SGD
from keras.preprocessing.image import load_img,img_to_array

# cyclical learning rates
from clr import LRFinder
from clr_callback import CyclicLR


# In[ ]:


# Constants
DATA_DIR = '../input/dog-breed-identification/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
BATCH_SIZE = 32
INPUT_SIZE = 224
NUM_CLASSES = 120
SEED = 42


# In[ ]:


# Let's check what's in the data directory
get_ipython().system(' ls $DATA_DIR')


# In[ ]:


# Read the train data set, which has the ids of the images and their labels (breeds)
# (adding the extension .jpg to the id becomes the file name of the image) 
train = pd.read_csv(DATA_DIR + 'labels.csv')
train.head()


# In[ ]:


# The submission file contains one column for the image id, and then one column 
# each breed in alphabetical order, with the probability of the dog in the image beeing of that breed
submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')
submission.head()


# In[ ]:


# Create a map of breeds to labels in the same order as the columns of the submission file
# and create a new column 'label' in the train data frame with the breeds mapped to this labels.
# This will make easier build the submission file from the predicted probabilities of the trained model
breed_labels = {breed:label for label,breed in enumerate(submission.columns[1:].values)}
train['label'] = train['breed'].map(breed_labels)
train.head()


# <h2>A bit of EDA</h2>

# In[ ]:


# Frequency of each breed in the train set. We can see that the most frequent breed
# has just above 120 images and the less frequent just above 60 images.
counts = train.breed.value_counts()
plt.figure(figsize=(10,40))
plt.xticks(np.arange(0, 130, 5))
sns.barplot(x=counts.values, y=counts.index);


# In[ ]:


# Let's plot some random images
fig, axs = plt.subplots(5,5, figsize=(20,20), squeeze=True)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
axs = axs.reshape(-1)
indices = np.random.choice(train.shape[0], 25, replace=False)
for ax, i in zip(axs, indices):
    img = cv2.imread(os.path.join(DATA_DIR, 'train', train.iloc[i].id + '.jpg'))
    h, w, c = img.shape
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(train.iloc[i].breed, fontsize=12)
    ax.imshow(img)
    


# *<h2>Variant 1 - Feature extraction from the last convolutional layer without pooling</h2>
# 
# We'll extract features directly from the last convolutional layer in VGG16, wich has shape (None, 7, 7, 512). This means that for every image
# we'll extract 7*7*512 = 25088 features, and connect them to our classifier.

# In[ ]:


base = VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
base.summary()


# <h3>flow_from_dataframe</h3>
# We don't have train and validation folders, so we cannot use directly the method <code>frow_from_directory</code> from <code>ImageDataGenerator</code>. Instead we'll be using the method <code>flow_from_dataframe</code>. Here's an <a href="https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c">article</a> on how to use it by the guy who wrote it, Vijayabhaskar J.

# In[ ]:


# Make the batchsize of the data generator a divisor of the number of images, as we have
# to make just one pass for feature extraction
batch_size = 269           # 10222 = 2 * 19 * 269

# No data augmentation, just rescaling the image
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(dataframe=train, 
                                              directory= TRAIN_DIR,
                                              x_col='id',
                                              y_col='label',
                                              class_mode='categorical',
                                              has_ext=False,
                                              batch_size=batch_size,   
                                              shuffle=False,
                                              seed=42,
                                              target_size=(224,224)                                              
                                             )


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Let's read all the images and labels into arrays in memory\n\ntrain_generator.reset()\ntrain_size = train.shape[0]\nfeatures = np.zeros(shape=(train_size, 7,7,512))\nlabels = np.zeros(shape=(train_size, NUM_CLASSES))\ni = 0\nfor inputs_batch, labels_batch in tqdm_notebook(train_generator):\n    features_batch = base.predict(inputs_batch)\n    features[i * batch_size:(i+1) * batch_size] = features_batch\n    labels[i * batch_size:(i+1) * batch_size] = labels_batch\n    i += 1\n    if i * batch_size >= train_size:\n        break;\n   \n# Flatten the output of the VGG16 base model\nfeatures = features.reshape(train_size, -1)")


# <h3>Fitting a CNN with the generated features</h3>

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.20, random_state=SEED)


# In[ ]:


# Build the classifier

def create_model(dropout=None):
    model = Sequential()
    model.add(Dense(1024, activation='relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    adam = Adam(lr=0.001)
    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'],  )
    return model


# In[ ]:


# Run a lr range test to find good learning rate margins

model = create_model()

BATCH_SIZE = 32
STEP_SIZE_TRAIN = X_train.shape[0] // BATCH_SIZE
STEP_SIZE_VALID = X_val.shape[0] // BATCH_SIZE

EPOCHS = 1
base_lr=0.0001
max_lr=100
step_size = EPOCHS * STEP_SIZE_TRAIN 
lrf = LRFinder(X_train.shape[0], BATCH_SIZE,
                       base_lr, max_lr,
                       # validation_data=(X_val, Yb_val),
                       lr_scale='exp', save_dir='./lr_find/', verbose=False)

history = model.fit(X_train, y_train, epochs=EPOCHS, steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data=[X_val, y_val], validation_steps = STEP_SIZE_VALID,
                   callbacks=[lrf])


# In[ ]:


# Training
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=[X_val, y_val])


# In[ ]:


fig = plt.figure(figsize=(15,7))
lrf.plot_schedule(clip_beginning=95, clip_endding=60)


# In[ ]:


10**(-1.7)


# In[ ]:


model = create_model()
EPOCHS=10
BATCH_SIZE=32
clr = CyclicLR(base_lr=0.005, max_lr=0.02, step_size=2*STEP_SIZE_TRAIN)

history = model.fit(X_train, y_train, epochs=EPOCHS, steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=[X_val, y_val], validation_steps=STEP_SIZE_VALID,
                    callbacks=[clr])


# As we can see in the history plot, there's a lot of overfitting. Varying the size of the FC layer in the network between 256 and 1024, and dropout between 0.1 and 0.5 doesn't change anything: train accuracy reaches around 0.99 and validation accuracy just below 0.3.

# In[ ]:


def plt_history(history, metric, title, ax, val=True):
    ax.plot(history[metric])
    if val:
        ax.plot(history['val_' + metric])
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    
hist = history.history
fig, ax = plt.subplots(1,2, figsize=(15,6))
plt_history(hist, 'loss', 'LOSS', ax[0])
plt_history(hist, 'acc', 'ACCURACY', ax[1])


# <h3>Create custom estimator</h3>
# Based on this excellent <a href="http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/">article</a> by Daniel Hnyk, and the Scikit-learn <a href="https://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator">documentation</a>.

# <h2>Variant 2 - Feature extraction from the last convolutional layer with average pooling</h2>
# 
# We'll extract features from the last convolutional layer in VGG16, wich has shape (None, 7, 7, 512), but average pooling this layer, so the output of the VGG16 model will be (None, 512).

# In[ ]:


base = InceptionV3(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3), pooling='avg')
base.summary()


# In[ ]:


# Make the batchsize of the data generator a divisor of the number of images, as we have
# to make just one pass for feature extraction
batch_size = 269           # 10222 = 2 * 19 * 269

# No data augmentation, just rescaling the image
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(dataframe=train, 
                                              directory= TRAIN_DIR,
                                              x_col='id',
                                              y_col='label',
                                              class_mode='categorical',
                                              has_ext=False,
                                              batch_size=batch_size,   
                                              shuffle=False,
                                              seed=42,
                                              target_size=(224,224)                                              
                                             )


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Let's read all the images and labels into arrays in memory. We can use the same generator,\n# but this time the features array will have shape (train_size, 512) instead of (train_size, 7*7*512)\n\n\ntrain_generator.reset()\ntrain_size = train.shape[0]\nfeatures = np.zeros(shape=(train_size, 2048))\nlabels = np.zeros(shape=(train_size, NUM_CLASSES))\ni = 0\nfor inputs_batch, labels_batch in tqdm_notebook(train_generator):\n    features_batch = base.predict(inputs_batch)\n    features[i * batch_size:(i+1) * batch_size] = features_batch\n    labels[i * batch_size:(i+1) * batch_size] = labels_batch\n    i += 1\n    if i * batch_size >= train_size:\n        break;\n   \n# This time the features array doesn't need flattening")


# In[ ]:


# Build the classifier. This time, the classifier has a lot fewer parameters
def create_model_pool(lr=0.001):
    model = Sequential()
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    sgd = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'],  )
    return model


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.25, random_state=SEED)


# In[ ]:


# Run a lr range test to find good learning rate margins

model = create_model_pool()

BATCH_SIZE = 32
STEP_SIZE_TRAIN = X_train.shape[0] // BATCH_SIZE
STEP_SIZE_VALID = X_val.shape[0] // BATCH_SIZE

EPOCHS = 1
base_lr=0.001
max_lr=1
step_size = EPOCHS * STEP_SIZE_TRAIN 
lrf = LRFinder(X_train.shape[0], BATCH_SIZE,
                       base_lr, max_lr,
                       validation_data=(X_val, y_val),
                       lr_scale='exp', save_dir='./lr_find/', verbose=False)

history = model.fit(X_train, y_train, epochs=EPOCHS, steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data=[X_val, y_val], validation_steps = STEP_SIZE_VALID,
                   callbacks=[lrf])


# In[ ]:


fig = plt.figure(figsize=(15,7))
lrf.plot_schedule(clip_beginning=50)


# In[ ]:


10**(-1.5)


# In[ ]:


# Training
model = create_model_pool()
EPOCHS=20
clr = CyclicLR(base_lr=0.01, max_lr=0.03, step_size=2*STEP_SIZE_TRAIN)

history = model.fit(X_train, y_train, epochs=EPOCHS, steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=[X_val, y_val], validation_steps=STEP_SIZE_VALID,
                    callbacks=[clr])


# In[ ]:


hist = history.history
fig, ax = plt.subplots(1,2, figsize=(15,6))
plt_history(hist, 'loss', 'LOSS', ax[0])
plt_history(hist, 'acc', 'ACCURACY', ax[1])


# As we can see, with this approach we still have a lot of overfitting and the results are even worse for the train set. It seems that we will need a wiser strategy (data augmentation) for this problem in part II of this series.
