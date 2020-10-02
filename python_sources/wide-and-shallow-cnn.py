#!/usr/bin/env python
# coding: utf-8

# # Pytorch sucks am I right (the sequel)

# In[ ]:


import os
import gc
import pandas as pd
import numpy as np
import transformers
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import AdaBoostClassifier


# Loading training/validation/test data.

# In[ ]:


kaggle = '/kaggle/input/undersampling-xlmtokenized/'
x_train = np.load(f'{kaggle}x_train.npy')
y_train = np.load(f'{kaggle}y_train.npy')

x_valid = np.load(f'{kaggle}x_valid.npy')
y_valid = np.load(f'{kaggle}y_valid.npy')

x_test = np.load(f'{kaggle}x_test.npy')


# ### TPU Configuration

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


# Global Variables
BATCHSIZE = 128
EPOCHS = 10
AUTO = tf.data.experimental.AUTOTUNE
NUMCORES = strategy.num_replicas_in_sync


# Helper function to build model.

# In[ ]:


def build(transformer, maxlen=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_ids = Input(shape=(maxlen,), dtype=tf.int32, name="input_word_ids")
    cls_token = transformer(input_ids)[0][:,0,:]
    out = tf.reshape(cls_token, [-1, cls_token.shape[1], 1])
    out = Dropout(0.25)(out)
    out = Conv1D(100, 3, padding='valid', activation='relu', strides=1)(out)
    out = BatchNormalization(axis=2)(out)
    out = Conv1D(100, 4, padding='valid', activation='relu', strides=1)(out)
    out = BatchNormalization(axis=2)(out)
    out = Conv1D(100, 5, padding='valid', activation='relu', strides=1)(out)
    out = BatchNormalization(axis=2)(out)
    out = GlobalMaxPooling1D()(out)
    out = Dropout(0.5)(out)
    out = Dense(1, activation='sigmoid')(out)
    model = Model(inputs=input_ids, outputs=out)
    model.compile(Adam(lr=1e-5), 
                  loss=BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy', AUC()])
    return model


# Create TF dataset objects.

# In[ ]:


train_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .repeat()
        .shuffle(x_train.shape[0])
        .batch(BATCHSIZE)
        .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_valid, y_valid))
        .batch(BATCHSIZE)
        .cache()
        .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
        .from_tensor_slices(x_test)
        .batch(BATCHSIZE)
)


# Load model into the TPU

# In[ ]:


get_ipython().run_cell_magic('time', '', "with strategy.scope():\n    transformer_layer = transformers.TFAutoModel.from_pretrained('jplu/tf-xlm-roberta-large')\n    model = build(transformer_layer, maxlen=192)\nmodel.summary()")


# ## Train Model

# In[ ]:


n_steps = 680

# Treat every instance of class 1 as 20 instances of class 0.
# 5% of training data is composed of toxic data.
# CLASS_WEIGHT = [1., 20.]
training_history = model.fit(train_dataset, 
                             steps_per_epoch=n_steps, 
                             validation_data=valid_dataset, 
                             epochs=EPOCHS)


# After saturating the learning potential of the model on english data only, we can train it for a few more epochs on the validation set to fine tune the translation process.

# In[ ]:


n_steps = x_valid.shape[0] // 128

# Treat every instance of class 1 as 7 instances of class 0.
# 15% of validation data is composed of toxic data
CLASS_WEIGHT = [1., 7.]
valid_history = model.fit(valid_dataset.repeat(), 
                          steps_per_epoch=n_steps, 
                          epochs=4,
                          class_weight=CLASS_WEIGHT)


# ## Plotting Training History

# In[ ]:


import matplotlib.pyplot as plt


plt.plot(training_history.history['auc'])
plt.plot(training_history.history['val_auc'])
plt.title('Model AUC vs. Epoch')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])
plt.title('Model Accuracy vs. Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('Model Loss vs. Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()


# ## Test Model
# 

# In[ ]:


sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
sub['toxic'] = model.predict(test_dataset, verbose=1)[0:len(sub['toxic'])]
sub.to_csv('submission.csv', index=False)

