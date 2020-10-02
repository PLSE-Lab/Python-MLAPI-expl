#!/usr/bin/env python
# coding: utf-8

# # MobileNet 126x126x3 100k per class
# 
# This kernel is description of my final solution for Quick, Draw! Doodle Recognition Challenge.<br>
# 
# I used MobileNet from keras.application package. Earlier I tried Resnet18 with Stochastic Depth, but it converged slower and had slightly lower performance. Changing model was first key decission to improve my score.<br>
# After that I experimented with image size and stroke encoding. It resulted with another rise of accuracy. Everything is described later in kernel.<br>
# I made some mistakes with reducing learning rate on plateou. LR was reduced to early and network hadn't possibility to converge because of too small learning rate. After increasing patient parameter to 10 was noticed significant advance.<br>
# 
# I also experimented with some RNN-CNN solutions, but there were worse than normal CNN.
# 
# 

# In[ ]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw import draw
from glob import glob
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from multiprocessing.dummy import Pool
from keras.models import load_model
import time
import ast
import keras
import random
import glob
import math


# In[ ]:


ALL_FILES = glob.glob('../input/shuffle-csvs*/*.csv.gz')
VALIDATION_FILE = '../input/shuffle-csvs-75000-100000/train_k0.csv.gz'
ALL_FILES.remove(VALIDATION_FILE)
INPUT_DIR = '../input/quickdraw-doodle-recognition/'
BASE_SIZE = 256
NCATS = 340
np.random.seed(seed=1987)


def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)


def plot_batch(x):    
    cols = 4
    rows = 6
    fig, axs = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(18, 18))
    for i in range(rows):
        for k in range(0,3):
            ax = axs[i, k]
            ax.imshow(x[i, :, :, k], cmap=plt.cm.gray)
            ax.axis('off')
        ax = axs[i, 3]
        ax.imshow(x[i, :, :], )
        ax.axis('off')
    fig.tight_layout()
    plt.show();


# # Learning and data hyper parameters<br> 
# If **AUGMENTATION** set True, images were flipped horizontaly with probability equals to 0.5.<br>
# **BATCH_SIZE** were reduced from 512 to 448, because there were a problem with memory when used loaded keras model. There is a issue ticket on keras github where people have the same problems.<br>
# In keras MobileNet documentation is stated that models trained on less downsized images have better accuracy. In this case, accuracy improvment was observed after changing **IMAGE_SIZE** from 64 to 128.<br>Strange image size which is not power of 2 is selected because of my typo in implementation, 126 would be much better choice. 
# 

# In[ ]:


AUGMENTATION = True
STEPS = 500
BATCH_SIZE = 448
EPOCHS = 0
LEARNING_RATE = 0.002


IMG_SHAPE = (128,128,3)
IMG_SIZE = IMG_SHAPE[0]


# # Image encoding
# 
# From raw stokes I created 3 images with different encoding. Encoded images were concatenated to one image of size 128x128 and 3 channels.<br>
# * First channel represents presence of line. Single point had 255 value if there were stroke or 0 otherwise.
# * Second channel encoded strokes in time. Usually people firstly draw outline of object and details later. This assumption were used to set weights of each stroke. First stroke was encoded with 255 value which was deacreased with every next stroke by 13,  down to 125.
# * Third channel encoded stroke points in time. There are some patterns in stroke directions. For example when I draw a spider, I start draw legs from body, not the other way. First point of a stroke have 255 value which was deacreased gradually down to 20.<br>
# 
# Later you can find some visualisation of presented encodings.
# 

# In[ ]:


def draw_cv2(raw_strokes, size=256, lw=6, augmentation = False):
    img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        points_count = len(stroke[0]) - 1
        grad = 255//points_count
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), (255, 255 - min(t,10)*13, max(255 - grad*i, 20)), lw)
    if size != BASE_SIZE:
        img = cv2.resize(img, (size, size))
    if augmentation:
        if random.random() > 0.5:
            img = np.fliplr(img)
    return img


# # Data generators
# 
# Data generators are based on Beluga kernel. In original data set classes are splitted to separate files. To improve performance files were merged in separete kernels to single files cointaining all classes examples in random order. It gave much faster generator and time to generate single batch decreased 2 times.

# In[ ]:




def image_generator(size, batchsize, lw=6, augmentation = False):
    while True:
        for filename in ALL_FILES:
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(eval)
                x = np.zeros((len(df), size, size,3))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i] = draw_cv2(raw_strokes, size=size, lw=lw, augmentation = augmentation)
                x = x / 255.
                x = x.reshape((len(df), size, size, 3)).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y

def valid_generator(valid_df, size, batchsize, lw=6):
    while(True):
        for i in range(0,len(valid_df),batchsize):
            chunk = valid_df[i:i+batchsize]
            x = np.zeros((len(chunk), size, size,3))
            for i, raw_strokes in enumerate(chunk.drawing.values):
                x[i] = draw_cv2(raw_strokes, size=size, lw=lw)
            x = x / 255.
            x = x.reshape((len(chunk), size, size,3)).astype(np.float32)
            y = keras.utils.to_categorical(chunk.y, num_classes=NCATS)
            yield x,y
        
def test_generator(test_df, size, batchsize, lw=6):
    for i in range(0,len(test_df),batchsize):
        chunk = test_df[i:i+batchsize]
        x = np.zeros((len(chunk), size, size,3))
        for i, raw_strokes in enumerate(chunk.drawing.values):
            x[i] = draw_cv2(raw_strokes, size=size, lw=lw)
        x = x / 255.
        x = x.reshape((len(chunk), size, size, 3)).astype(np.float32)
        yield x
        
        
train_datagen = image_generator(size=IMG_SIZE, batchsize=BATCH_SIZE, augmentation = AUGMENTATION)

valid_df = pd.read_csv(VALIDATION_FILE)
valid_df['drawing'] = valid_df['drawing'].apply(eval)
validation_steps = len(valid_df)//BATCH_SIZE
valid_datagen = valid_generator(valid_df, size=IMG_SIZE, batchsize=BATCH_SIZE)


# # Visualization of image encoding for "ambulance" class
# 
# Below is visualization of encoding for some ambulance images. First 3 columns represent described earlier channels and last 4th column is preview of whole image in RGB scale.<br>
# In second column we can see that our assumption that people draw an outline first is right. Body of car have brighter lines than wheels or cross on the side. Also we can notice that there is pattern in drawing a wheel. Most people starts at top and sketch in anticlockwise direction.

# In[ ]:


single_class_df = valid_df[valid_df['y'] == 2]
single_class_gen = valid_generator(single_class_df, size=IMG_SIZE, batchsize=BATCH_SIZE)
x, y = next(single_class_gen)
plot_batch(x)


# # Visualization of trainging batch

# In[ ]:


x, y = next(train_datagen)
plot_batch(x)


# # Visualization of validation batch

# In[ ]:


x, y = next(valid_datagen)
plot_batch(x)


# # Model definition
# 
# As I mentioned before, I used MobileNet from keras.application package. To train it for more epochs I piplined kernels to load previously trained model and continue whole process. It was annoying, but it helps me to save a lot of money on AWS or other cloud computing.<br>
# 
# Because network were trained from generator and there were a lot of data (3.4kk), each epoch means 500 batches for 488 examples.<br>
# For the first time I used ReduceLROnPlateau callback and have learned that properly reduced LR can provide much better results. After 10 epochs without progress its reduced learning rate by half which usually gives noticable drop of loss value and helps network to converge.

# In[ ]:


from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model

def top_3_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
checkpointer = ModelCheckpoint(filepath='mobileNet-best.hdf5', verbose=0, save_best_only=True)
model = load_model('../input/mobilenet-126x126x3-100k-per-class/mobileNet.hdf5', custom_objects = {'top_3_accuracy':top_3_accuracy})
opt = Adam(lr = LEARNING_RATE)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy', top_3_accuracy])
model.summary()


# # Training

# In[ ]:


history = model.fit_generator(train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS , validation_data=valid_datagen, validation_steps = validation_steps, callbacks= [checkpointer, reducer])
model.save('mobileNet.hdf5')


# In[ ]:


#merged log.csv files from each kernel version
log_df = pd.read_csv('../input/whole-training-log/log.csv')


# In[ ]:


p = log_df[['loss','val_loss']].plot(figsize = (7,7))
p.set_xlabel('Epochs')
p


# In[ ]:


p = log_df[['acc','val_acc', 'top_3_accuracy','val_top_3_accuracy']].plot(figsize = (7,7))
p.set_xlabel('Epochs')
p


# In[ ]:


p = log_df[['lr']].plot(figsize = (5,5))
p.set_xlabel('Epochs')
p


# Below we can observe mentioned loss drop after reducing LR.

# In[ ]:


p = log_df.iloc[150:200][['loss','val_loss']].plot(figsize = (7,7))
p.set_xlabel('Epochs')
p


# In[ ]:


log = pd.DataFrame.from_dict(history.history)
log.to_csv('train_log.csv', index=False)


# # Evaluation of model using MAP3

# In[ ]:


gen = test_generator(valid_df, size=IMG_SIZE, batchsize=BATCH_SIZE)
valid_predictions = model.predict_generator(gen, steps = validation_steps, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('Map3: {:.3f}'.format(map3))


# In[ ]:


submission_df = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
submission_df['drawing'] = submission_df['drawing'].apply(eval)
submission_datagen = test_generator(submission_df, size=IMG_SIZE, batchsize=BATCH_SIZE)
submission_predictions = model.predict_generator(submission_datagen, math.ceil(len(submission_df)/BATCH_SIZE))
cats = list_all_categories()
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
top3 = preds2catids(submission_predictions)
top3cats = top3.replace(id2cat)
submission_df['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = submission_df[['key_id', 'word']]
submission.to_csv('submission.csv', index=False)


# In[ ]:




