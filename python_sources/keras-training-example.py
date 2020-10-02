#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## EfficientNet
from efficientnet import EfficientNetB0

effnet = EfficientNetB0(include_top=False, weights=None, pooling='avg')
effnet.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b0_imagenet_1000_notop.h5')


# In[ ]:


## Globals
SZ = 224
SEED = 4


# In[ ]:


## Prepare Data
import numpy as np
import pandas as pd

train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train.diagnosis.hist(); train.diagnosis.value_counts()


# Our models will have a tendency to overfit to classes 0 and 2. A stratified train/test split will provide a validation set with a similar distribution as the training data, but will improperly test the model's ability to classify the disease intensity across the board. We need a validation set that will represent the model's true discriminative power. 
# 
# Let's take 15% of each class stratified, but limit the over-represented classes (0 and 2) to 15% of the least-represented class (3). 

# In[ ]:


# Train/Test Split
np.random.seed(SEED)

limit = int(193 * 0.15)
zer = train[train.diagnosis == 0].sample(n=limit)
one = train[train.diagnosis == 1].sample(frac=0.15)
two = train[train.diagnosis == 2].sample(n=limit)
thr = train[train.diagnosis == 3].sample(frac=0.15)
fou = train[train.diagnosis == 4].sample(frac=0.15)
valid = pd.concat([zer, one, two, thr, fou])
train = train.drop(valid.index)

xt, yt = train.id_code, train.diagnosis
xv, yv = valid.id_code, valid.diagnosis

yt.hist(); yv.hist()


# In[ ]:


# Extract
from keras.utils.data_utils import _extract_archive

def extract_imgs(ds):
    path = f'../input/aptos2019-224/{ds}_images.tar.gz'
    _extract_archive(path)

def append_ext(fname):
    if '_' in fname:
        return fname + '.jpeg'
    return fname + '.png'

extract_imgs('train')


# In[ ]:


# Manage Preprocessing
from keras.applications.imagenet_utils import preprocess_input
def preprocess(img):
    img = preprocess_input(img, mode='torch')
    return img


# In[ ]:


# Data Generator
from keras.preprocessing.image import ImageDataGenerator

bs = 128
shape = (SZ, SZ, 3)
train = pd.concat((xt, yt), axis=1).astype(str)
valid = pd.concat((xv, yv), axis=1).astype(str)
train.id_code = train.id_code.apply(append_ext)
valid.id_code = valid.id_code.apply(append_ext)
train_dir = './train_images'

def dataflow(df, mode='t'):
    assert mode in ['t', 'v']
    tfms = {'rotation_range': 360,
            'preprocessing_function': preprocess}
    if mode == 't': tfms.update({'zoom_range': 0.1,
                                 'horizontal_flip': True,
                                 'vertical_flip': True})
    datagen = ImageDataGenerator(**tfms)
    return datagen.flow_from_dataframe(
        df, directory=train_dir, x_col='id_code', y_col='diagnosis',
        target_size=(SZ, SZ), class_mode='sparse',
        batch_size=bs, shuffle=True, seed=SEED,
        validate_filenames=False, drop_duplicates=False)


# In[ ]:


## Training
# Metrics
# https://www.kaggle.com/carlolepelaars/efficientnetb3-with-keras-aptos-2019
from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score

def get_preds_and_labels(model, datagen):
    """
    Get predictions and labels from the data generator.
    Referencing `EmbedWrapper`, note that our data
    generator outputs two y targets and our model does
    the same.
    """
    y_pred, y_true = [], []
    for _ in range(datagen.samples // bs):
        x, y = next(datagen)
        yh = model.predict(x)
        yh = np.argmax(yh, axis=1)
        y_pred.append(yh)
        y_true.append(y)
    return np.concatenate(y_pred).ravel(), np.concatenate(y_true).ravel()

class QWK(Callback):
    def on_train_begin(self, logs={}):
        """Initialize best score variable."""
        self.best = 0.
    
    def on_epoch_end(self, epoch, logs={}):
        """
        Gets QWK score on the validation data and saves best weights.
        Assumes both the model and validation data generator exist.
        """
        y_pred, y_true = get_preds_and_labels(model, dataflow(valid, mode='v'))
        y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)
        score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        if score > self.best:
            self.best = score
            self.model.save_weights('weights.h5')
            print(f'val_qwk: {score:.4f}')
        return


# In[ ]:


# Model
from keras import Input, Model
from keras.layers import Dropout, Dense
from keras.optimizers import SGD

sgd = SGD(1e-4, decay=1e-5, momentum=0.9, nesterov=True, clipnorm=10.)
def effnet_aptos(shape):
    i = Input(shape=shape)
    x = effnet(i)
    x = Dropout(0.2, seed=SEED)(x)
    o = Dense(5, activation='softmax', kernel_initializer='zeros')(x)
    model = Model(inputs=i, outputs=o)
    for layer in model.layers[:-1]:
        layer.trainable = False
    return model


# In[ ]:


# Compile Params
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']


# In[ ]:


# Compile
model = effnet_aptos(shape)
model.compile(loss=loss, optimizer=sgd, metrics=metrics)
model.summary()


# In[ ]:


# Train Top
tsteps = np.ceil(len(train) / bs)
vsteps = np.ceil(len(valid) / bs)

history = model.fit_generator(dataflow(train), epochs=3, steps_per_epoch=tsteps)


# In[ ]:


# Unfreeze
for layer in model.layers:
    layer.trainable = True
model.compile(loss=loss, optimizer=sgd, metrics=metrics)
model.summary()


# In[ ]:


# Cosine Anneal, Model Checkpoint, QWK
from cosineanneal import CosineAnnealingScheduler
from keras.callbacks import ModelCheckpoint

epochs = 30
callbacks = [CosineAnnealingScheduler(T_max=epochs, eta_max=1e-2, eta_min=1e-3),
             QWK()]

# Train Full
history = model.fit_generator(dataflow(train),
                              epochs=epochs,
                              steps_per_epoch=tsteps,
                              validation_data=dataflow(valid, mode='v'),
                              validation_steps=vsteps,
                              callbacks=callbacks)


# In[ ]:


# Cleanup
import shutil
shutil.rmtree('./train_images')

