#!/usr/bin/env python
# coding: utf-8

# ## About this notebook
# 
# 
# The TPU v3-8 accelerators on Kaggle are amazingly fast, but unfortunately there is a usage limit on them.
# 
# On the other hand, TPU v2-8 accelerators are freely available in Google Colab, which are still super fast. However, these accelerators only have 8gb of memory on each TPU core, whch makes the high-performing Kaggle notebooks with the extremely powerful XLM-R large model fail on Colab. 
# 
# 
# Here I show how to make some small modifications on the workflow to enable training an XLM-R large model on free COlab TPU v2-8 accelerators.
# 
# I hope this will help people experiment more freely with their ideas, and push their score even higher.
# 
# ---
# 
# To try it, just 
# - 1, Download this notebook and upload it to Colab
# - 2, Upload the training/validation/test/submission csv files to your Google drive, mount Google drive on Colab and change path to data
# - 3, Change runtime to TPU and you are ready to go.
# 
# ---
# 
# This notebook is made for Colab, so I did not run it here. But it gets a very similar score to the pevious one, 0.933 validation AUC after the first stage, and 0.95 train AUC after stage 2.
# 
# ** Note: This notebook tries to be simple, and only uses a small amount of data, and it does not use translated datasets or other tricks. You need to add those yourself to squeeze out a good score.**
# 
# Suggestions/improvements are appreciated!
# 
# ---
# 
# ### References:
# 
# - This notebook directly reuses code from my previous notebook showing how to use a custom training loop [link](https://www.kaggle.com/riblidezso/tpu-custom-tensoflow2-training-loop)
# - This notebook heavily relies on the great [notebook]((https://www.kaggle.com/xhlulu//jigsaw-tpu-xlm-roberta) by, Xhulu: [@xhulu](https://www.kaggle.com/xhulu/) 
# - The tensorflow distrubuted training tutorial: [Link](https://www.tensorflow.org/tutorials/distribute/custom_training)

# ### We will need to install Transformers on Colab

# In[ ]:


get_ipython().system('pip install transformers  # necessary on colab')


# In[ ]:


MAX_LEN = 192
LR = 1e-5
BATCH_SIZE = 8 # per TPU core, reduced to fit on a TPUv2
TOTAL_STEPS_STAGE1 = 2000  # increased the number of steps for smaller batches
VALIDATE_EVERY_STAGE1 = 500
TOTAL_STEPS_STAGE2 = 1000
VALIDATE_EVERY_STAGE2 = 500

PRETRAINED_MODEL = 'jplu/tf-xlm-roberta-large'

# The path to the data on my drive
D = 'drive/My Drive/jigsaw/data/original/'

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import transformers
from transformers import TFAutoModel, AutoTokenizer
import logging
# no extensive logging 
logging.getLogger().setLevel(logging.NOTSET)

AUTO = tf.data.experimental.AUTOTUNE


# ## Collect functions from the previous notebook here
# 
# - Note we will redefine some of these later

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


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])


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


def create_model_and_optimizer():
    with strategy.scope():
        transformer_layer = TFAutoModel.from_pretrained(PRETRAINED_MODEL)                
        model = build_model(transformer_layer)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR, epsilon=1e-08)
    return model, optimizer


def build_model(transformer):
    inp = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
    # Huggingface transformers have multiple outputs, embeddings are the first one
    # let's slice out the first position, the paper says its not worse than pooling
    x = transformer(inp)[0][:, 0, :]  
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[inp], outputs=[out])
    
    return model


def define_losses_and_metrics():
    with strategy.scope():
        loss_object = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE, from_logits=False)

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size = global_batch_size)
            return loss

        train_accuracy_metric = tf.keras.metrics.AUC(name='training_AUC')

    return compute_loss, train_accuracy_metric



def train(train_dist_dataset, val_dist_dataset=None, y_val=None,
          total_steps=5000, validate_every=500):
    step = 0
    ### Training lopp ###
    for tensor in train_dist_dataset:
        distributed_train_step(tensor) 
        step+=1

        if (step % validate_every == 0):   
            ### Print train metrics ###  
            train_metric = train_accuracy_metric.result().numpy()
            print("Step %d, train AUC: %.5f" % (step, train_metric))   
            
            ### Test loop with exact AUC ###
            if val_dist_dataset:
                val_metric = roc_auc_score(y_val, predict(val_dist_dataset))
                print("     validation AUC: %.5f" %  val_metric)   

            ### Reset (train) metrics ###
            train_accuracy_metric.reset_states()
            
        if step  == total_steps:
            break



@tf.function
def distributed_train_step(data):
    strategy.experimental_run_v2(train_step, args=(data,))

def train_step(inputs):
    features, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy_metric.update_state(labels, predictions)




def predict(dataset):  
    predictions = []
    for tensor in dataset:
        predictions.append(distributed_prediction_step(tensor))
    ### stack replicas and batches
    predictions = np.vstack(list(map(np.vstack,predictions)))
    return predictions

@tf.function
def distributed_prediction_step(data):
    predictions = strategy.experimental_run_v2(prediction_step, args=(data,))
    return strategy.experimental_local_results(predictions)

def prediction_step(inputs):
    features = inputs  # note datasets used in prediction do not have labels
    predictions = model(features, training=False)
    return predictions


# ## Connect to TPU

# In[ ]:


tpu, strategy, global_batch_size = connect_to_TPU()
print("REPLICAS: ", strategy.num_replicas_in_sync)

compute_loss, train_accuracy_metric = define_losses_and_metrics()


# ## Prepare data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n### Load ###\ntrain_df = pd.read_csv(D+'jigsaw-toxic-comment-train.csv')\nval_df = pd.read_csv(D+'validation.csv')\ntest_df = pd.read_csv(D+'test.csv')\nsub_df = pd.read_csv(D+'sample_submission.csv')\n\n### subsample the train dataframe to 50%-50%  ###\ntrain_df = pd.concat([\n    train_df.query('toxic==1'),\n    train_df.query('toxic==0').sample(sum(train_df.toxic),random_state=42)\n])\n### shufle it just to make sure ###\ntrain_df = train_df.sample(frac=1, random_state = 42)\n\n### Tokenize  ###\ntokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)\nX_train = regular_encode(train_df.comment_text.values, tokenizer, maxlen=MAX_LEN)\nX_val = regular_encode(val_df.comment_text.values, tokenizer, maxlen=MAX_LEN)\nX_test = regular_encode(test_df.content.values, tokenizer, maxlen=MAX_LEN)\n\n### Make appropriate target shapes ###\ny_train = train_df.toxic.values.reshape(-1,1)\ny_val = val_df.toxic.values.reshape(-1,1)\n\n### Create datasets  ###\ntrain_dist_dataset = create_dist_dataset(X_train, y_train, True)\nval_dist_dataset   = create_dist_dataset(X_val)\ntest_dist_dataset  = create_dist_dataset(X_test)")


# ## Loading the pretrained transformer will crash a colab standard instance, it runs out of RAM
# 
# (It will not crash a high-ram instance available with Colab pro though)
# 
# This is somehow connected to the models being defined in the distributed starategy scope. But don't ask me why... 

# In[ ]:


# model, optimizer = create_model_and_optimizer()


# ### Workaround: Let's initialize the pretrained model from just the config file without loading weights.
# 
# #### And then load the pretrained weights outside the ditributed stategy and assign weights manually
# 
# 
# Loading the model takes a few minutes again

# In[ ]:


#Download the config file
get_ipython().system('wget https://s3.amazonaws.com/models.huggingface.co/bert/jplu/tf-xlm-roberta-large/config.json')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom transformers import PretrainedConfig, TFRobertaModel\nCONFIG_PATH = 'config.json'\n\n\ndef create_model_from_config():\n    with strategy.scope():\n        ### Load only config no weights ###\n        config = PretrainedConfig.from_json_file(CONFIG_PATH)                \n        transformer_layer = TFRobertaModel(config) \n\n        ### Make the cls model ###               \n        model = build_model(transformer_layer)\n        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)\n    model.summary()\n    return model, optimizer\n\n\ndef load_weights_workaround():\n    ### Load full pretrained model outside strategy scope ###\n    transformer_layer = TFAutoModel.from_pretrained(PRETRAINED_MODEL)\n\n    ### Assign weights \n    for tv1, tv2 in zip(model.layers[1].trainable_variables,\n                        transformer_layer.trainable_variables):\n        tv1.assign(tv2)\n\n\nmodel, optimizer = create_model_from_config()\nload_weights_workaround()\nmodel.summary()")


# ## Unfortunately training will still fail on a TPUv2 with ResourceExhausted error
# 
# Now matter how short the sequence length is or how small the batch is, it always happens.
# 
# Apparently there simply isn't enough memory to hold 3 copies of the weights/gradients for Adam  with a >2gb model, and only 8gb memory on each core.

# In[ ]:


#train(train_dist_dataset, val_dist_dataset, y_val,
#      TOTAL_STEPS_STAGE1, VALIDATE_EVERY_STAGE1)


# ### Workaround we need to use SGD (without momentum)
# 
# No extra copies of parameters, so it will fit.

# In[ ]:


with strategy.scope():
   optimizer = tf.keras.optimizers.SGD(learning_rate=LR)


# ### But SGD can be quite unstable with high noise so let's clip gradients
# 
# Clipping is a bit tricky though, because for some reason the Keras optimizers  just do not accept clipnorm in distributed training strategy.
# 
# So we need to apply it manually.
# 
# **Tensorflow 2.1 and 2.2 is a little dfferent here, this version is for 2.2 and Colab **

# In[ ]:


CLIP_NORM = 1  # agressive clipping

@tf.function
def train_step(data):
    inputs, targets = data
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = compute_loss(targets, predictions)

    ### There is an unused pooler head of the tranformer with None gradients
    ### we need to get rid of it before clipping
    trainable_variables = [v for v in model.trainable_variables 
                           if 'pooler' not in v.name]

    ### Calculate grads
    gradients = tape.gradient(loss, trainable_variables)
    
    ### We cannot clip replicas, it throws an error
    ### First we have to manually sum the gradients from the replicas
    gradients = tf.distribute.get_replica_context().all_reduce('sum', gradients)

    ### Clip by global norm, (do not change gradient direction)
    gradients, _ = tf.clip_by_global_norm(gradients, CLIP_NORM)

    ### Apply gradients
    ### NOTE: Only for tenforflow 2.2 on colab!!!!
    optimizer.apply_gradients(zip(gradients, trainable_variables),
                              experimental_aggregate_gradients=False)

    train_accuracy_metric.update_state(targets, predictions)


# Increase learning rate for agressively clipped gradients

# In[ ]:


optimizer.learning_rate.assign(0.01)


# ## Finally,  we are ready to train
# 

# In[ ]:


#%%time
train(train_dist_dataset, val_dist_dataset, y_val,
      TOTAL_STEPS_STAGE1, VALIDATE_EVERY_STAGE1)


# ## Finetune it on the validation data

# In[ ]:


get_ipython().run_cell_magic('time', '', '# make a new dataset for training with the validation data \n# with targets, shuffling and repeating\nval_dist_dataset_4_training = create_dist_dataset(X_val, y_val, training=True)\n\n# train again\ntrain(val_dist_dataset_4_training,\n      total_steps = TOTAL_STEPS_STAGE2, \n      validate_every = VALIDATE_EVERY_STAGE2)  # not validating but printing now')


# ## Make predictions and submission

# In[ ]:


get_ipython().run_cell_magic('time', '', "sub_df['toxic'] = predict(test_dist_dataset)[:,0]\nsub_df.to_csv('submission.csv', index=False)")

