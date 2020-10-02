#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[ ]:


from IPython.utils import io

with io.capture_output() as out:
    get_ipython().system('pip install -U efficientnet')


# In[ ]:


import os
import gc # Garbage Collector

from kaggle_datasets import KaggleDatasets

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams as mrc
import seaborn as sns

from sklearn.metrics import roc_auc_score

import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, model_from_json

from tensorflow.keras.optimizers import Adam, RMSprop, Adamax
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint

from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras import backend as K

# Load the TensorBoard extension
get_ipython().run_line_magic('load_ext', 'tensorboard')

print('tf-version :', tf.__version__)


# # TPU - Config

# In[ ]:


try:

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  
    print('Running on TPU ', resolver.master())

except ValueError:
    resolver = None

if resolver:
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

else:
    strategy = tf.distribute.get_strategy() 

n_devices = strategy.num_replicas_in_sync

print('Number of Devices: ', n_devices)

devices =  tf.config.list_logical_devices('TPU')

print('#' * 50)

for i in range(len(devices)):
    print(f'Device {i+1}: {devices[i]}')


# # Config

# In[ ]:


pd.set_option('chained_assignment', None)

mrc['figure.figsize'] = (10, 6)

plt.style.use('ggplot')


# In[ ]:


# Define colors palette, for data visualization (EDA)

colors = sns.color_palette("hls", 10)

sns.palplot(colors)

class_p = [colors[3], colors[0]]


# In[ ]:


path = '../input/siim-isic-melanoma-classification/'

out_path = '../working/'

# CSV
csv_train = path + 'train.csv'

csv_test = path + 'test.csv'

csv_sample = path + 'sample_submission.csv'


# In[ ]:


# Access Data, Google Cloud Storage (GCS)
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

trainRec = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')
testRec  = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')


# In[ ]:


default = 1024 # The default images shape

img_width  = 512
img_height = 512

SHAPE = (img_width, img_height) # New images shape

CLASSES = ['benign', 'malignant']


# # Functions Image Data

# In[ ]:


def plot(x, y, classes, nrows=4, ncols=8, grayscale=False, stack=None):
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    ax = ax.flatten()
    
    for i in range(len(x)):
        
        image = x[i]
        
        if stack != None:
            image = image[stack]
            
        if grayscale:
            image = tf.image.rgb_to_grayscale(image)[:, :, 0] # rgb_to_grayscale.squeeze()
        
        ax[i].imshow(image)
        ax[i].set_title(classes[int(y[i])])
        ax[i].grid('off')
        ax[i].axis('off')

    fig.tight_layout()
    plt.show()


# In[ ]:


def decode(encoded_image):

    image = tf.io.decode_jpeg(encoded_image, channels=3)
    image = tf.reshape(image, [default, default, 3])  
    image = tf.image.resize(image, SHAPE)
    
    
    return image


# In[ ]:


def read_tfrecord(example):
    """
    Parses an image and label from the given `example`.
    """

    features = {
      'image': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.int64),
      }

    # parser
    example = tf.io.parse_single_example(example, features)

    # decode the image
    image = decode(example['image'])

    # cast labels
    label = tf.cast(example['target'], tf.float32)

    return image , label


# In[ ]:


def read_unlabeled_tfrecord(example):
    """
    Parses an image and image_id from the given `example`.
    """

    features = {
      'image': tf.io.FixedLenFeature([], tf.string),
      'image_name': tf.io.FixedLenFeature([], tf.string),
      }

    # parser
    example = tf.io.parse_single_example(example, features)

    # decode the image
    image = decode(example['image'])

    # image_id
    image_id = example['image_name']

    return image , image_id


# In[ ]:


def preprocessing(image, label):

    image = tf.cast(image, tf.float32) * (1. / 255)
    
    return image, label


# # **Tabular Data EDA**

# In[ ]:


# CSV

train = pd.read_csv(csv_train)
test = pd.read_csv(csv_test)

sample = pd.read_csv(csv_sample)

m, n = train.shape


# In[ ]:


train.dropna(axis=0, inplace=True)
test.dropna(axis=0, inplace=True)


# In[ ]:


print(train.shape)
print(train.isna().sum())

train.head(3)


# In[ ]:


print(test.shape)

print(test.isna().sum())


# In[ ]:


count = np.unique(train['benign_malignant'], return_counts=True)
print(count[0], count[1])

# The dataset is high imbalanced.


# In[ ]:


sns.countplot(x='benign_malignant', data=train, palette=class_p)
plt.show()


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

ax = ax.flatten()

# benign
benign = train[train['target'] == 0].sort_values(by='anatom_site_general_challenge')

sns.countplot(x='anatom_site_general_challenge', hue='benign_malignant', data=benign, ax=ax[0], palette=[colors[3]])
ax[0].legend(loc='upper right')

# malignant
malignant = train[train['target'] == 1].sort_values(by='anatom_site_general_challenge')

sns.countplot(x='anatom_site_general_challenge', hue='benign_malignant', data=malignant, ax=ax[1], palette=[colors[0]])
ax[1].legend(loc='upper right')

plt.tight_layout()
plt.show()


# In[ ]:


sns.countplot(x='sex', hue='benign_malignant', data=train, palette=class_p)
plt.show()


# In[ ]:


sns.countplot(x='age_approx', hue='benign_malignant', data=train, palette=class_p)
plt.show()


# In[ ]:


sns.countplot(x='sex', hue='diagnosis', data=train)
plt.legend(loc='upper right')
plt.show()


# In[ ]:


sns.countplot(x='age_approx', hue='diagnosis', data=train)
plt.tight_layout()
plt.show()


# In[ ]:


sns.countplot(x='sex', hue='anatom_site_general_challenge', data=train)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# In[ ]:


sns.countplot(x='age_approx', hue='anatom_site_general_challenge', data=train)
plt.show()


# In[ ]:


# Probability (target w.r.t anatom_site_general_challenge)

map_target = {}
target_prob = {}

count_map = np.unique(train['anatom_site_general_challenge'], return_counts=True)

prob = count_map[1] / train.shape[0]

print(prob)

for cat in count_map[0]:
    map_target[cat] =  np.unique(train[train['anatom_site_general_challenge'] == cat]['target'], return_counts=True)
    target_prob[f'{cat}_prob'] = map_target[cat][1] / train.shape[0]

target_prob = pd.DataFrame(target_prob)

target_prob


# In[ ]:


# Negative Entropy (target w.r.t anatom_site_general_challenge)

target_en = {}

for cat in count_map[0]:
    map_target[cat] =  np.unique(train[train['anatom_site_general_challenge'] == cat]['target'], return_counts=True)
    target_en[f'{cat}_entropy(-)'] = map_target[cat][1] / train.shape[0]
    
    target_en[f'{cat}_entropy(-)'] = -1 * target_en[f'{cat}_entropy(-)'] * np.log(target_en[f'{cat}_entropy(-)'])
    
target_en = pd.DataFrame(target_en)

target_en

# To be continued :) (:.


# # TFRecordDataset

# In[ ]:


BATCH_SIZE = 128
VAL_BATCH_SIZE = 128

BUFFER_SIZE = 2048

AUTOTUNE = tf.data.experimental.AUTOTUNE

ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False


# In[ ]:


tfData = tf.data.TFRecordDataset(trainRec, num_parallel_reads=AUTOTUNE)
tfData = tfData.with_options(ignore_order)

tfData = tfData.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
tfData = tfData.map(preprocessing, num_parallel_calls=AUTOTUNE)


# In[ ]:


testData = tf.data.TFRecordDataset(testRec, num_parallel_reads=AUTOTUNE)
testData = testData.with_options(ignore_order)

testData = testData.map(read_unlabeled_tfrecord, num_parallel_calls=AUTOTUNE)
testData = testData.map(preprocessing, num_parallel_calls=AUTOTUNE)

testData = testData.batch(BATCH_SIZE)
testData = testData.prefetch(AUTOTUNE)

testData = testData.cache()


# # K-Fold Cross Validation

# In[ ]:


"""
Implements K-Fold cross validation.
"""

def get_kfold(tfData, size, k=6, batch=False):
        
    step = size // k
    rest = size % k 

    kfold_train = {}
    kfold_val = {}
        
    
    for i in range(k-1):
        
        """
        For training the required sequence is (take ---> skip ----> take), 
        For validation the required sequence is (skip ----> take a step)
        """
        take_i = step * (i + 1)
        skip_i = step * (i + 2)

        # train
        kfold_train[i] = tfData.take(take_i)
        kfold_train[i] = kfold_train[i].concatenate(tfData.skip(skip_i))
        
        # validation
        kfold_val[i] = tfData.skip(take_i)
        kfold_val[i] = kfold_val[i].take(step)
    
        
        if batch:
            
            # train
            kfold_train[i] = kfold_train[i].batch(BATCH_SIZE)
            kfold_train[i] = kfold_train[i].prefetch(AUTOTUNE)
              
            # validation
            kfold_val[i] = kfold_val[i].batch(VAL_BATCH_SIZE)
            kfold_val[i] = kfold_val[i].prefetch(AUTOTUNE)
            kfold_val[i] = kfold_val[i].cache()

        print(f'K = {i+1}, Max Range: {(step) * (i + 1)}')
    
    # train at k
    kfold_train[(k-1)] = tfData.skip(step)

    # validation at k
    kfold_val[(k-1)] = tfData.take(step)
    
    # merge rest to (train at k)
    tfRest = tfData.skip(m - rest)
    kfold_train[(k-1)] = kfold_train[(k-1)].concatenate(tfRest)

    if batch:
        
        # train
        kfold_train[(k-1)] = kfold_train[(k-1)].batch(BATCH_SIZE)
        kfold_train[(k-1)] = kfold_train[(k-1)].prefetch(AUTOTUNE)

        # validation
        kfold_val[(k-1)] = kfold_val[(k-1)].batch(BATCH_SIZE)
        kfold_val[(k-1)] = kfold_val[(k-1)].prefetch(AUTOTUNE)
        kfold_val[(k-1)] = kfold_val[(k-1)].cache()
        
    print(f'K = {k}, Max Range: {(step + rest) * (k)}')
    
    if rest > 0:
        print('=' * 50)

        print('Rest (size % k) =', rest)

    return kfold_train, kfold_val


# In[ ]:


# Validate get_kfold() function, (i.e. get_kfold() gives right indice At k=x)

def test_kfold(test_size, k=6, batch=False):
    
    example = tf.data.Dataset.range(test_size)
    
    ex_k_train, ex_k_val = get_kfold(example, test_size, k=k, batch=batch)
    
    print('-' * 50)
    
    for i in range(k):
        
        print(f'K = {i+1} :\n')
        
        if batch:
            print('Examples/Batch:\n')
            print('#' * 50)

        print(f'Train :{np.array(list(ex_k_train[i].as_numpy_iterator())).shape[0]} example')
        
        if not batch:
            print(f' Min index :{np.array(list(ex_k_train[i].as_numpy_iterator())).min()}')
            print(f' Max index :{np.array(list(ex_k_train[i].as_numpy_iterator())).max()}')

        print('-' * 50)
        
        print(f'Test :{np.array(list(ex_k_val[i].as_numpy_iterator())).shape[0]} example')
        
        if not batch:
            print(f' Min index :{np.array(list(ex_k_val[i].as_numpy_iterator())).min()}')
            print(f' Max index :{np.array(list(ex_k_val[i].as_numpy_iterator())).max()}')    
        
        print('=' * 50)


# In[ ]:


# size divisible by k, k=4
test_kfold(test_size=1024,k=4, batch=False)


# In[ ]:


# size not divisible by k, k=9
test_kfold(test_size=1024,k=9, batch=False)


# In[ ]:


# size divisible by k, k=4 (batch = True)
test_kfold(test_size=1024, k=4, batch=True)


# # Model Config (Train/Validation)

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nkfold = 6\n\nk_tfTrain, k_tfVal = get_kfold(tfData, size=m, k=kfold, batch=True)')


# # **Images EDA**

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# training examples (sample)\nsample_x, sample_y = k_tfTrain[0].as_numpy_iterator().next()\n\nprint(sample_x.shape)')


# In[ ]:


# grayscale
plot(sample_x[:32], sample_y[:32], CLASSES, grayscale=True, stack=None)


# In[ ]:


# RBG
plot(sample_x[:32], sample_y[:32], CLASSES, grayscale=False, stack=None)


# In[ ]:


try:
   
    del sample_x, sample_y
    gc.collect()
    
except:
    print('Memory has been Freed')


# # Model

# In[ ]:


"""
ModelConfig (could be used for parameters tuning),
by apply grid or random search, then save (self.opt_config as csv file)
"""

class ModelConfig:
    
    def __init__(self, classes, base_model=None, opt='adam',  opt_config={}, tol=0.001, 
                 logs_dir='./logs', save_dir="./model", csv_info_dir="./csv"):
        
        self.base_model = base_model
        
        self.opt = {'adam': Adam, 
                    'adamax':Adamax, 
                    'rmsprop':RMSprop}
        
        self.classes = classes
        
        # (opt_config) dictionary, learning_rate, beta_1, beta_2, ..etc.
        self.opt_config = opt_config
        
        # optimizer key
        self.opt_key = opt
        
        self.loss = None
        self.auc = None
        self.optimizer = None
        
        self.count_logs = 0
        self.logs_dir = logs_dir
        
        self.callbacks = [EarlyStopping(monitor='val_loss', min_delta=tol,
                                        mode='auto', restore_best_weights=True), 
                          ModelCheckpoint(filepath=save_dir, monitor='val_loss', save_best_only=True, 
                                          save_weights_only=True, save_freq='epoch'),
                          TensorBoard(log_dir=self.logs_dir, histogram_freq=1, write_images=True),
                          CSVLogger(filename=csv_info_dir, append=True)]
        

    def build_model(self):
        
        assert self.base_model != None, 'base_model is None'
                   
        self.optimizer = self.opt[self.opt_key](**self.opt_config)
        self.auc = tf.keras.metrics.AUC()
        self.loss = tf.keras.losses.BinaryCrossentropy()

        model = Sequential([self.base_model 
                                 ,Dense(self.classes, 'sigmoid')])

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.auc])
        
        return model
    
    def update_logs_dir(self):
        
        self.callbacks[2].log_dir = self.logs_dir + '_' + str(self.count_logs)
        self.count_logs +=1
    
    @staticmethod
    def save_model(model, out_path):
    
        model_json = model.to_json()

        with open(out_path + "model.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights("model.h5")

        print("Model Saved")
    
    @staticmethod
    def load(path):
    
        json_file = open(path + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model.model = model_from_json(loaded_model_json)
        self.model.model.load_weights("model.h5")

        return model


# # Train Model

# In[ ]:


# ### same as version 3.0


# In[ ]:


# ### same as version 3.0

# config = {'learning_rate': 0.01, 
#           'beta_1':0.9, 
#           'beta_2':0.999}


# In[ ]:


# ### same as version 3.0

# # tol -- > tolerance validation loss
# with strategy.scope():
    
#     base_model = EfficientNetB0(include_top=False, weights='imagenet', 
#                                 input_shape=(*SHAPE, 3), pooling='avg')
    
#     model_config = ModelConfig(classes=1, base_model=base_model, opt='adam', opt_config=config, tol=0.001, 
#                                logs_dir='./logs', save_dir="./model", csv_info_dir="./info.csv")

#     model = model_config.build_model()

# model.summary()


# In[ ]:


# ### same as version 3.0

# %%time

# # Test model, At - k = 0

# trainData, valData = k_tfTrain[0], k_tfVal[0]
# model.fit(trainData, epochs=1, steps_per_epoch=BATCH_SIZE, validation_data=valData)


# In[ ]:


# ## same as version 3.0

# %%time

# # Test model, For all K-Folds

# # Callbacks - EarlyStopping
# callbacks = model_config.callbacks[0]

# for i in range(kfold):
    
#     # Releases (clear) the global graph 
#     K.clear_session()
    
#     # Model reinitialization
#     with strategy.scope():
#         model = model_config.build_model()

#     print(f'Model : {i+1}')
#     print('=' * 80)
    
#     trainData, valData = k_tfTrain[i], k_tfVal[i]
    
#     model.fit(trainData, epochs=2, validation_data=valData, callbacks=callbacks)
    
#     print('#' * 80)
    
    
# """
# For a better performance, we could use @tf.function decorator 
# for (AutoGraph Transformations). https://www.tensorflow.org/api_docs/python/tf/function
# """

# print()


# # Parameters Tuning (ModelConfig Class)

# In[ ]:


### Update version 6.0

def run_search(tfTrain, tfVal, params_grid: list):
    
    history = {}
    
    # tol -- > tolerance validation loss
    with strategy.scope():

        base_model = EfficientNetB0(include_top=False, weights='imagenet', 
                                input_shape=(*SHAPE, 3), pooling='avg')
        
        # opt_config = {}, opt = ''
        model_config = ModelConfig(classes=1, base_model=base_model, opt='', opt_config={}, tol=0.001, 
                                    logs_dir='./logs', save_dir="./model", csv_info_dir="./info.csv")
    
    for i in range(len(params_grid)):
        
        # model config at i
        config = params_grid[i]
        
        # Releases (clear) the global graph 
        K.clear_session()

        ## update model parameters
        model_config.opt_key    = config['optimizer']
        model_config.opt_config = config['params']

        epochs = config['epochs']
        ##

        # Model reinitialization
        with strategy.scope():
            model = model_config.build_model()

        print(f'Model : {i+1}')
        print('=' * 80)
        
        
        # cache: model (train data, validation data) loss & AUC
        results = model.fit(tfTrain, epochs=epochs, validation_data=tfVal)
        
        # store ith model (config & results)
        history[i] = {'config':config, 'results':results.history}
        
        print('#' * 80)
    
    return history

"""
built-in
--------

Tune : https://docs.ray.io/en/latest/tune.html

or 

Tensorflow : https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
"""

print()


# > Usually it's preferred to use random search in case of tunning a large number of hyperparameters

# In[ ]:


"""
get_random_params() for EfficientNetB0, gives a quite useless (poor) n-models,
therefore, choosing parameters, by intuition or grid search, gives satisfying results 
for a small number of hyperparameters.
"""
def get_random_params(sample_size=5):
    pass


# In[ ]:


# returns best params and model result, based on (monitor)
def get_best_params(history, monitor='val_auc'):
    
    """
    best_params (tuple)
    """
    
    best_params = (-1, None)
    
    for i in range(len(history)):
        
        # last epoch val_auc (monitor) result
        resulte =  history[i]['results'][monitor][-1]
        
        config  =  history[i]['config']
        
        # current choice
        choice = (resulte, config)
        
        """
        max(t1, t2) --> keep best_params, based on maximum result of monitor
        (val_auc), i.e. t1[0] or t2[0]
        """
        best_params = max(best_params, choice)
    
    return best_params


# In[ ]:


# optional, in case of searching, for n-days
def save_history(history, fname):
    
    params_df = pd.DataFrame(history).T
    params_df.to_csv(f'{fname}.csv', index=False, encoding='utf-8')


# In[ ]:


# Grid Search |  dummy parameters (example)

params = [{'optimizer': 'adam', 'params':{'learning_rate': 0.002, 'beta_1':0.96, 'beta_2':0.999}, 'epochs':3},
          {'optimizer': 'adam', 'params':{'learning_rate': 0.002, 'beta_1':0.9, 'beta_2':0.999},  'epochs': 3},
          {'optimizer': 'adam', 'params':{'learning_rate': 0.001, 'beta_1':0.96, 'beta_2':0.999}, 'epochs': 3},
          {'optimizer': 'adam', 'params':{'learning_rate': 0.001, 'beta_1':0.9, 'beta_2':0.999},  'epochs': 3}]


# In[ ]:


# run search (k-fold - at k = 0)
trainData, valData = k_tfTrain[0], k_tfVal[0]

history = run_search(trainData, valData, params)


# In[ ]:


monitor = 'val_auc'

# get best parameters
best_params = get_best_params(history, monitor=monitor)

print('best_parameters :', best_params[-1], f'\n{monitor} :', best_params[0])


# In[ ]:


save_history(history, 'params_file_1')


# # Train - Best Parameters Model

# In[ ]:


optimizer = best_params[-1]['optimizer']
config = best_params[-1]['params']

epochs = best_params[-1]['epochs']


# In[ ]:


with strategy.scope():
    
    base_model = EfficientNetB0(include_top=False, weights='imagenet', 
                                input_shape=(*SHAPE, 3), pooling='avg')
    
    model_config = ModelConfig(classes=1, base_model=base_model, opt=optimizer, opt_config=config, tol=0.001, 
                               logs_dir='./logs', save_dir="./model", csv_info_dir="./info.csv")

    model = model_config.build_model()

model.summary()


# In[ ]:


# Test: fit model - (k-fold - at k = 0)
trainData, valData = k_tfTrain[0], k_tfVal[0]

# cache: model (train data, validation data) loss & AUC
history = model.fit(trainData, epochs=epochs, validation_data=valData)

"""
run_search() have to be over all k-folds and the choice is the average over 
the monitor, which might be the 'val_auc'
"""


# # Prediction

# In[ ]:


def  to_csv(prediction: dict):
    
    dataframe = pd.DataFrame({'image_name':np.array(list(prediction.keys())).astype(str), 
                              'target':np.array(list(prediction.values())).squeeze().astype('float')})
    
    dataframe.to_csv('submission.csv', encoding='utf-8', index=False)
    
    return dataframe


# In[ ]:


def predict(testData, model):
    
    prediction = {}

    for x, names in testData.as_numpy_iterator():

        """
        predict_on_batch() rather than predict(), as testData was batched,
        therefore there's no need to use predict()
        """
        pred_step = model.predict_on_batch(x)

        for i in range(len(names)):
            prediction[names[i]] = pred_step[i]
    
    return prediction


# In[ ]:


get_ipython().run_cell_magic('time', '', 'prediction = predict(testData, model)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nsubmission = to_csv(prediction)\n\nsubmission.head(10)')

