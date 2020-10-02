#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from textblob import TextBlob

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# import wandb
import transformers
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


# # TPU Config

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


#IMP DATA FOR CONFIG

AUTO = tf.data.experimental.AUTOTUNE


# # Configuration
EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192
MODEL = 'jplu/tf-xlm-roberta-large'
D = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/'
D_TRANS = '/kaggle/input/jigsaw-train-multilingual-coments-google-api/'


# In[ ]:


def fast_encode(texts, tokenizer, maxlen=512):
    '''
    Function for encoding
    '''
    all_ids = []
    encs = tokenizer.batch_encode_plus(texts, max_length=maxlen, pad_to_max_length = True,
                                          return_token_type_ids=False, return_attention_masks=False)
        
    return np.array(encs['input_ids'])


# In[ ]:


fast_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)


# In[ ]:


def load_jigsaw_trans(langs=['tr','it','es','ru','fr','pt'], 
                      columns=['comment_text', 'toxic']):
    train_6langs=[]
    for i in range(len(langs)):
        fn = D_TRANS+'jigsaw-toxic-comment-train-google-%s-cleaned.csv'%langs[i]
        train_6langs.append(downsample(pd.read_csv(fn)[columns]))

    return train_6langs

def downsample(df):
    """Subsample the train dataframe to 50%-50%"""
    ds_df= pd.concat([
        df.query('toxic==1'),
        df.query('toxic==0').sample(sum(df.toxic))
    ])
    
    return ds_df
    

train = pd.concat(load_jigsaw_trans()) 


# In[ ]:


train1 = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')
train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
# valid = pd.read_csv('../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv')
# test2 = pd.read_csv('../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')
train2.toxic = train2.toxic.round().astype(int)


# In[ ]:


train3 = pd.concat([
    train1[['comment_text','toxic']].query('toxic==1'),
    train1[['comment_text','toxic']].query('toxic==0').sample(sum(train1.toxic), random_state=0),
    train[['comment_text','toxic']],
    train2[['comment_text','toxic']].query('toxic==1'),
    train2[['comment_text','toxic']].query('toxic==0').sample(sum(train2.toxic), random_state=0)
])


# In[ ]:


train3[['toxic']].query('toxic==0').count()


# In[ ]:


x_train = fast_encode(train.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_valid = fast_encode(validation.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)


# In[ ]:


y_train = train.toxic.values
y_valid = validation.toxic.values


# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train,y_train))
    .repeat()
    .shuffle(len(x_train))
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid,y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)


# In[ ]:


# class LRFinder(Callback):
    
#     '''
#     A simple callback for finding the optimal learning rate range for your model + dataset. 
    
#     # Usage
#         ```python
#             lr_finder = LRFinder(min_lr=1e-5, 
#                                  max_lr=1e-2, 
#                                  steps_per_epoch=np.ceil(epoch_size/batch_size), 
#                                  epochs=3)
#             model.fit(X_train, Y_train, callbacks=[lr_finder])
            
#             lr_finder.plot_loss()
#         ```
    
#     # Arguments
#         min_lr: The lower bound of the learning rate range for the experiment.
#         max_lr: The upper bound of the learning rate range for the experiment.
#         steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
#         epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
#     # References
#         Blog post: jeremyjordan.me/nn-learning-rate
#         Original paper: https://arxiv.org/abs/1506.01186
#     '''
    
#     def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
#         super().__init__()
        
#         self.min_lr = min_lr
#         self.max_lr = max_lr
#         self.total_iterations = steps_per_epoch * epochs
#         self.iteration = 0
#         self.history = {}
        
#     def clr(self):
#         '''Calculate the learning rate.'''
#         x = self.iteration / self.total_iterations 
#         return self.min_lr + (self.max_lr-self.min_lr) * x
        
#     def on_train_begin(self, logs=None):
#         '''Initialize the learning rate to the minimum value at the start of training.'''
#         logs = logs or {}
#         K.set_value(self.model.optimizer.lr, self.min_lr)
        
#     def on_batch_end(self, epoch, logs=None):
#         '''Record previous batch statistics and update the learning rate.'''
#         logs = logs or {}
#         self.iteration += 1

#         self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
#         self.history.setdefault('iterations', []).append(self.iteration)

#         for k, v in logs.items():
#             self.history.setdefault(k, []).append(v)
            
#         K.set_value(self.model.optimizer.lr, self.clr())
 
#     def plot_lr(self):
#         '''Helper function to quickly inspect the learning rate schedule.'''
#         plt.plot(self.history['iterations'], self.history['lr'])
#         plt.yscale('log')
#         plt.xlabel('Iteration')
#         plt.ylabel('Learning rate')
#         plt.show()
        
#     def plot_loss(self):
#         '''Helper function to quickly observe the learning rate experiment results.'''
#         plt.plot(self.history['lr'], self.history['loss'])
#         plt.xscale('log')
#         plt.xlabel('Learning rate')
#         plt.ylabel('Loss')
#         plt.show()


# In[ ]:


# from tf.keras.callbacks import LearningRateScheduler
lr_scheds = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * (0.4 ** np.floor(epoch / 2)))
# lr_scheds = LRFinder(min_lr=2e-5,
#                      max_lr=1e-5, 
#                      steps_per_epoch=np.ceil(x_train.shape[0]/BATCH_SIZE), 
#                      epochs=3)


# In[ ]:


def build_model(transformer, maxlen=512):
    input_word_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name='input_word_ids')
    sequence_output = transformer(input_word_ids)[0]
    
    clf_output = sequence_output[:,0,:]
    out = tf.keras.layers.Dense(1, activation='sigmoid')(clf_output)
    
    model = tf.keras.models.Model(inputs = input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


with strategy.scope():
    transformer_layer = (
        transformers.TFAutoModelWithLMHead
        .from_pretrained(MODEL)
    )
    model = build_model(transformer_layer, maxlen=MAX_LEN)
    
model.summary()


# In[ ]:


n_steps = x_train.shape[0]//BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch = n_steps,
    validation_data = valid_dataset,
    epochs = EPOCHS,
    callbacks=[lr_scheds]
)


# In[ ]:


from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.metrics import roc_auc_score
n_steps_valid = x_valid.shape[0] //BATCH_SIZE
cv_scores = []
auc_scores = []
skfold = KFold(5, True, 1)
for train_index, test_index in skfold.split(x_valid,y_valid):
    x_vtrain, x_vtest = x_valid[train_index], x_valid[test_index]
    y_vtrain, y_vtest = y_valid[train_index], y_valid[test_index]
    
    train_vdata = (
    tf.data.Dataset
    .from_tensor_slices((x_vtrain,y_vtrain))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
    ) 
    
    test_data = (
    tf.data.Dataset
    .from_tensor_slices((x_vtest,y_vtest))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
    )
    valid_history = model.fit(
    train_vdata.repeat(),
    steps_per_epoch = n_steps_valid,
    epochs = EPOCHS
    )
    auc_score = roc_auc_score(y_vtest, model.predict(x_vtest))
    auc_scores.append(auc_score)
    score = model.evaluate(test_data,verbose=0)
    cv_scores.append(score)


# In[ ]:


np.mean(cv_scores[1][1])
# np.std(cv_scores)


# In[ ]:


auc_scores


# In[ ]:


sub = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
sub['toxic'] = model.predict(test_dataset, verbose=1)
# sub.to_csv('submission.csv', index=False )


# In[ ]:


ensemble = pd.read_csv('../input/multitpu-inference/submission-ensemble.csv', index_col='id')


# In[ ]:


def scale_min_max_submission(submission):
    min_, max_ = submission['toxic'].min(), submission['toxic'].max()
    submission['toxic'] = (submission['toxic'] - min_) / (max_ - min_)
    return submission


# In[ ]:


sub['toxic'] = (scale_min_max_submission(sub)['toxic'] + scale_min_max_submission(ensemble)['toxic']) /2
sub.to_csv('submission.csv', index=False )


# In[ ]:


test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')


# In[ ]:


test['toxic'] = sub['toxic']


# In[ ]:


sub.drop(columns=['toxic'], axis=1, inplace=True)


# In[ ]:


# # test.drop(columns=['content'], axis=1, inplace=True)
test.rename(columns = {'content':'comment_text'}, inplace=True)


# In[ ]:


test.toxic = test.toxic.round().astype(int)


# In[ ]:


data_new = pd.concat([train[['comment_text','toxic']].query('toxic==0').sample(n=39548, random_state=42),
                      train[['comment_text','toxic']].query('toxic==1').sample(n=79764, random_state=42),
                       test[['comment_text','toxic']]])


# In[ ]:


data_new[['toxic']].query('toxic==1').count()


# In[ ]:


y_data_new = data_new.toxic.values


# In[ ]:


x_train_new = fast_encode(data_new.comment_text.astype(str), fast_tokenizer, MAX_LEN)


# In[ ]:


# tf.keras.backend.clearsession() 
# tf.tpu.experimental.shutdowntpusystem() 
# tf.tpu.experimental.initializetpu_system()


# In[ ]:


train_dataset_new = (
    tf.data.Dataset
    .from_tensor_slices((x_train_new,y_data_new))
    .repeat()
    .shuffle(len(x_train_new))
    .batch(64)
    .prefetch(AUTO)
)


# In[ ]:


with strategy.scope():
    transformer_layers = (
        transformers.TFAutoModelWithLMHead
        .from_pretrained(MODEL)
    )
    model_new = build_model(transformer_layers, maxlen=MAX_LEN)
    
model_new.summary()


# In[ ]:


# from tf.keras.callbacks import LearningRateScheduler
# lr_sched_new = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * (0.2 ** np.floor(epoch / 2)))


# In[ ]:


n_steps = x_train_new.shape[0]//64
train_history = model_new.fit(
    train_dataset_new,
    steps_per_epoch = n_steps,
    validation_data = valid_dataset,
    epochs = 2,
)


# In[ ]:


sub['toxic'] = model_new.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False )

