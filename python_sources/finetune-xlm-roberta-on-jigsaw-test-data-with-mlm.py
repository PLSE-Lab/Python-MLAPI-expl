#!/usr/bin/env python
# coding: utf-8

# ## About this notebook
# 
# 
# Domain specific masked language model (MLM) fine tuning was reported to improve the performance of BERT models [(arxiv/1905.05583)](https://arxiv.org/abs/1905.05583)
# 
# MLM fine tuning is also regularly mentioned as one of the things winning teams have tried in previous kaggle NLP competitions (sometimes it is reported to help, sometimes not).
# 
# 
# Huggingface provides a pytorch script to fine tune models, however there is no TensorFlow solution which just works out of the box.
# 
# 
# Here I share a notebook with my implementation of MLM fine tuning on TPUs with a custom training loop in TensorFlow 2.
# 
# In my experience, starting from an MLM finetuned model makes training faster and more stable, and possibly more accurate.
# 
# Suggestions/improvements are appreciated!
# 
# ---
# 
# ### References:
# 
# - This notebook relies on the great [notebook]((https://www.kaggle.com/xhlulu//jigsaw-tpu-xlm-roberta) by, Xhulu: [@xhulu](https://www.kaggle.com/xhulu/) 
# - The tensorflow distrubuted training tutorial: [Link](https://www.tensorflow.org/tutorials/distribute/custom_training)

# In[ ]:


MAX_LEN = 128
BATCH_SIZE = 16 # per TPU core
TOTAL_STEPS = 2000  # thats approx 4 epochs
EVALUATE_EVERY = 200
LR =  1e-5

PRETRAINED_MODEL = 'jplu/tf-xlm-roberta-large'
D = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/'


import os
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.optimizers import Adam
import transformers
from transformers import TFAutoModelWithLMHead, AutoTokenizer
import logging
# no extensive logging 
logging.getLogger().setLevel(logging.NOTSET)

AUTO = tf.data.experimental.AUTOTUNE


# ## Connect to TPU

# In[ ]:


def connect_to_TPU():
    """Detect hardware, return appropriate distribution strategy"""
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

    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

    return tpu, strategy, global_batch_size


tpu, strategy, global_batch_size = connect_to_TPU()
print("REPLICAS: ", strategy.num_replicas_in_sync)


#  ## Load text data into memory
#  
#  - Now we will only use the multilingual test and validation data.

# In[ ]:


val_df = pd.read_csv(D+'validation.csv')
test_df = pd.read_csv(D+'test.csv')


# ## Tokenize  it with the models own tokenizer
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef regular_encode(texts, tokenizer, maxlen=512):\n    enc_di = tokenizer.batch_encode_plus(\n        texts, \n        return_attention_masks=False, \n        return_token_type_ids=False,\n        pad_to_max_length=True,\n        max_length=maxlen\n    )\n    \n    return np.array(enc_di['input_ids'])\n    \n\ntokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)\nX_val = regular_encode(val_df.comment_text.values, tokenizer, maxlen=MAX_LEN)\nX_test = regular_encode(test_df.content.values, tokenizer, maxlen=MAX_LEN)")


# ### Mask inputs and create appropriate labels for masked language modelling
# 
# Following the BERT recipee: 15% used for prediction. 80% of that is masked. 10% is random token, 10% is just left as is.

# In[ ]:


def prepare_mlm_input_and_labels(X):
    # 15% BERT masking
    inp_mask = np.random.rand(*X.shape)<0.15 
    # do not mask special tokens
    inp_mask[X<=2] = False
    # set targets to -1 by default, it means ignore
    labels =  -1 * np.ones(X.shape, dtype=int)
    # set labels for masked tokens
    labels[inp_mask] = X[inp_mask]
    
    # prepare input
    X_mlm = np.copy(X)
    # set input to [MASK] which is the last token for the 90% of tokens
    # this means leaving 10% unchanged
    inp_mask_2mask = inp_mask  & (np.random.rand(*X.shape)<0.90)
    X_mlm[inp_mask_2mask] = 250001  # mask token is the last in the dict

    # set 10% to a random token
    inp_mask_2random = inp_mask_2mask  & (np.random.rand(*X.shape) < 1/9)
    X_mlm[inp_mask_2random] = np.random.randint(3, 250001, inp_mask_2random.sum())
    
    return X_mlm, labels


# use validation and test data for mlm
X_train_mlm = np.vstack([X_test, X_val])
# masks and labels
X_train_mlm, y_train_mlm = prepare_mlm_input_and_labels(X_train_mlm)


# ## Create distributed tensorflow dataset
# 

# In[ ]:


def create_dist_dataset(X, y=None, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(X)

    ### Add y if present ###
    if y is not None:
        dataset_y = tf.data.Dataset.from_tensor_slices(y)
        dataset = tf.data.Dataset.zip((dataset, dataset_y))
        
    ### Repeat if training ###
    if training:
        dataset = dataset.shuffle(len(X)).repeat()

    dataset = dataset.batch(global_batch_size).prefetch(AUTO)

    ### make it distributed  ###
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    return dist_dataset
    
    
train_dist_dataset = create_dist_dataset(X_train_mlm, y_train_mlm, True)


# ## Build model from pretrained transformer
# 
# We just use the pretrained model loaded as a model with a language modelling head.
# 
# 
# 
# - Note: Downloading the model takes some time!
# - Note reported number of total params, and the numbers printed in the column Param # do not match, as the weights of the mbedding matrix are shared between the encoder and the decoder.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef create_mlm_model_and_optimizer():\n    with strategy.scope():\n        model = TFAutoModelWithLMHead.from_pretrained(PRETRAINED_MODEL)\n        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)\n    return model, optimizer\n\n\nmlm_model, optimizer = create_mlm_model_and_optimizer()\nmlm_model.summary()')


# ### Define stuff for the masked language modelling, and the custom training loop
# 
# We will need:
# - 1, MLM loss, and a loss metric. These need to be defined in the scope of the distributed strategy. The MLM loss is only evauluated at the 15% of the tokens, so we need to make a custom masked sparse categorical crossentropy. Labels with -1 value are ignored during loss calculation.
# - 2, A full training loop
# - 3, A distributed train step called in the training loop, which uses a single replica train step
# 

# In[ ]:


def define_mlm_loss_and_metrics():
    with strategy.scope():
        mlm_loss_object = masked_sparse_categorical_crossentropy

        def compute_mlm_loss(labels, predictions):
            per_example_loss = mlm_loss_object(labels, predictions)
            loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size = global_batch_size)
            return loss

        train_mlm_loss_metric = tf.keras.metrics.Mean()
        
    return compute_mlm_loss, train_mlm_loss_metric


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked,
                                                          y_pred_masked,
                                                          from_logits=True)
    return loss

            
            
def train_mlm(train_dist_dataset, total_steps=2000, evaluate_every=200):
    step = 0
    ### Training lopp ###
    for tensor in train_dist_dataset:
        distributed_mlm_train_step(tensor) 
        step+=1

        if (step % evaluate_every == 0):   
            ### Print train metrics ###  
            train_metric = train_mlm_loss_metric.result().numpy()
            print("Step %d, train loss: %.2f" % (step, train_metric))     

            ### Reset  metrics ###
            train_mlm_loss_metric.reset_states()
            
        if step  == total_steps:
            break


@tf.function
def distributed_mlm_train_step(data):
    strategy.experimental_run_v2(mlm_train_step, args=(data,))


@tf.function
def mlm_train_step(inputs):
    features, labels = inputs

    with tf.GradientTape() as tape:
        predictions = mlm_model(features, training=True)[0]
        loss = compute_mlm_loss(labels, predictions)

    gradients = tape.gradient(loss, mlm_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mlm_model.trainable_variables))

    train_mlm_loss_metric.update_state(loss)
    

compute_mlm_loss, train_mlm_loss_metric = define_mlm_loss_and_metrics()


# ## Finally train it
# 
# 
# - Note it takes some time
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_mlm(train_dist_dataset, TOTAL_STEPS, EVALUATE_EVERY)')


# ## Save finetuned model
# 
# This fine tuned model can be loaded just as the original to build a classification model from it.

# In[ ]:


mlm_model.save_pretrained('./')

