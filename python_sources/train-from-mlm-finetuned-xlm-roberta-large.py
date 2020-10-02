#!/usr/bin/env python
# coding: utf-8

# ## About this notebook
# 
# This notebook trains from the XLM-Roberta large model which was finetuned with masked language modelling on the jigsaw test dataset [Link](https://www.kaggle.com/riblidezso/finetune-xlm-roberta-on-jigsaw-test-data-with-mlm).
# 
# This notebook also implements a few improvements compared to a previous starter notebook that I shared
# 
# * 1, It trains on translated data
# * 2, It uses different learning rate for the head layer and the transformer
# * 3, It restores the model weights after training to the checkpoint which had the highest validation score
# 
# Suggestions/improvements are appreciated!
# 
# ---
# 
# ### References:
# 
# - The shared XLM-Roberta large model, finetuned on the Jigsaw multilingual test data with masked language modelling Notebook [link]() / Dataset [link](https://www.kaggle.com/riblidezso/jigsaw-mlm-finetuned-xlm-r-large)
# - My previous starter notebook [link](https://www.kaggle.com/riblidezso/tpu-custom-tensoflow2-training-loop)
# - This notebook uses the translated versions of the training dataset too, big thanks to Michael Kazachok! [link](https://www.kaggle.com/miklgr500/jigsaw-train-multilingual-coments-google-api)
# - This notebook uses different learning rate for the transformer and the head, I got the ideas from the writeup of the winning team of the Google QUEST Q&A Labeling competition  [link](https://www.kaggle.com/c/google-quest-challenge/discussion/129840), I have seen it described to be useful elsewhere too.
# - This notebook heavily relies on the great [notebook]((https://www.kaggle.com/xhlulu//jigsaw-tpu-xlm-roberta) by, Xhulu: [@xhulu](https://www.kaggle.com/xhulu/) 
# - The tensorflow distrubuted training tutorial: [Link](https://www.tensorflow.org/tutorials/distribute/custom_training)

# In[ ]:


MAX_LEN = 192 
DROPOUT = 0.5 # use aggressive dropout
BATCH_SIZE = 16 # per TPU core
TOTAL_STEPS_STAGE1 = 2000
VALIDATE_EVERY_STAGE1 = 200
TOTAL_STEPS_STAGE2 = 200
VALIDATE_EVERY_STAGE2 = 10

### Different learning rate for transformer and head ###
LR_TRANSFORMER = 5e-6
LR_HEAD = 1e-3

PRETRAINED_TOKENIZER=  'jplu/tf-xlm-roberta-large'
PRETRAINED_MODEL = '/kaggle/input/jigsaw-mlm-finetuned-xlm-r-large'
D = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/'
D_TRANS = '/kaggle/input/jigsaw-train-multilingual-coments-google-api/'


import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import transformers
from transformers import TFRobertaModel, AutoTokenizer
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
#  - Traning data is englih + all translations. The reason to use english too is that people use english in their foreign language comments all the time.
#  - Not using the full dataset, downsampling negatives to 50-50%

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
    

train_df = pd.concat(load_jigsaw_trans()) 
val_df = pd.read_csv(D+'validation.csv')
test_df = pd.read_csv(D+'test.csv')
sub_df = pd.read_csv(D+'sample_submission.csv')


# ## Tokenize  it with the models own tokenizer
# 
# - Note it takes some time ( approx 5 minutes)
# - Note, we need to reshape the targets

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef regular_encode(texts, tokenizer, maxlen=512):\n    enc_di = tokenizer.batch_encode_plus(\n        texts, \n        return_attention_masks=False, \n        return_token_type_ids=False,\n        pad_to_max_length=True,\n        max_length=maxlen\n    )\n    \n    return np.array(enc_di['input_ids'])\n    \n\ntokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER)\nX_train = regular_encode(train_df.comment_text.values, tokenizer, maxlen=MAX_LEN)\nX_val = regular_encode(val_df.comment_text.values, tokenizer, maxlen=MAX_LEN)\nX_test = regular_encode(test_df.content.values, tokenizer, maxlen=MAX_LEN)\n\ny_train = train_df.toxic.values.reshape(-1,1)\ny_val = val_df.toxic.values.reshape(-1,1)")


# ## Create distributed tensorflow datasets
# 
# - Note, validation dataset does not contain labels, we keep track of it ourselves

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
    
    
train_dist_dataset = create_dist_dataset(X_train, y_train, True)
val_dist_dataset   = create_dist_dataset(X_val)
test_dist_dataset  = create_dist_dataset(X_test)


# ## Build model from pretrained transformer
# 
# 
# Let's use a different learning rate for the head and the transformer like the winning team of the Google QUEST Q&A Labeling competition  [link](https://www.kaggle.com/c/google-quest-challenge/discussion/129840). 
# 
# The reasoning is the following, the transformer is trained for super long time and has a very good multilingual representaton, which we only want to change a little, while the head needs to be trained from scratch.
# 
# We define 2 separate optimizers for the transofmer and the head layer. This is a simple way to use different learning rate for the transformer and the head. The caffe style "lr_multiplier" option would be more elegant but that is not available in keras.
# 
# We add the name 'custom' to the head layer, so that we can find it later and use a different learning rate with this layer
# 
# - Note: Downloading the model takes some time!

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef create_model_and_optimizer():\n    with strategy.scope():\n        transformer_layer = TFRobertaModel.from_pretrained(PRETRAINED_MODEL)                \n        model = build_model(transformer_layer)\n        optimizer_transformer = Adam(learning_rate=LR_TRANSFORMER)\n        optimizer_head = Adam(learning_rate=LR_HEAD)\n    return model, optimizer_transformer, optimizer_head\n\n\ndef build_model(transformer):\n    inp = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")\n    # Huggingface transformers have multiple outputs, embeddings are the first one\n    # let\'s slice out the first position, the paper says its not worse than pooling\n    x = transformer(inp)[0][:, 0, :]  \n    x = Dropout(DROPOUT)(x)\n    ### note, adding the name to later identify these weights for different LR\n    out = Dense(1, activation=\'sigmoid\', name=\'custom_head\')(x)\n    model = Model(inputs=[inp], outputs=[out])\n    \n    return model\n\n\nmodel, optimizer_transformer, optimizer_head = create_model_and_optimizer()\nmodel.summary()')


# ### Define stuff for the custom training loop
# 
# We will need:
# - 1, losses, and  optionally a training AUC metric here: these need to be defined in the scope of the distributed strategy. 
# - 2, A full training loop
# - 3, A distributed train step called in the training loop, which uses a single replica train step
# - 4, A prediction loop with dstibute 
# 
# 
# At the end of training we restore the parameters which had the best validation score.
# 
# 
# For the different learning rate we need to apply gradients in two steps, check the train_step function for details.
# 
# 
# 
# - Note, we are using exact AUC, for the valdationdata, and approximate AUC for the training data

# In[ ]:


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
          total_steps=2000, validate_every=200):
    best_weights, history = None, []
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
                print("Step %d,   val AUC: %.5f" %  (step,val_metric))   
                
                # save weights if it is the best yet
                history.append(val_metric)
                if history[-1] == max(history):
                    best_weights = model.get_weights()

            ### Reset (train) metrics ###
            train_accuracy_metric.reset_states()
            
        if step  == total_steps:
            break
    
    ### Restore best weighths ###
    model.set_weights(best_weights)



@tf.function
def distributed_train_step(data):
    strategy.experimental_run_v2(train_step, args=(data,))

def train_step(inputs):
    features, labels = inputs
    
    ### get transformer and head separate vars
    # get rid of pooler head with None gradients
    transformer_trainable_variables = [ v for v in model.trainable_variables 
                                       if (('pooler' not in v.name)  and 
                                           ('custom' not in v.name))]
    head_trainable_variables = [ v for v in model.trainable_variables 
                                if 'custom'  in v.name]

    # calculate the 2 gradients ( note persistent, and del)
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(features, training=True)
        loss = compute_loss(labels, predictions)
    gradients_transformer = tape.gradient(loss, transformer_trainable_variables)
    gradients_head = tape.gradient(loss, head_trainable_variables)
    del tape
        
    ### make the 2 gradients steps
    optimizer_transformer.apply_gradients(zip(gradients_transformer, 
                                              transformer_trainable_variables))
    optimizer_head.apply_gradients(zip(gradients_head, 
                                       head_trainable_variables))

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


compute_loss, train_accuracy_metric = define_losses_and_metrics()


# ## Finally train it on english comments
# 
# 
# - Note it takes some time
# - Don't mind the warning: "Converting sparse IndexedSlices to a dense Tensor"

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train(train_dist_dataset, val_dist_dataset, y_val,\n      TOTAL_STEPS_STAGE1, VALIDATE_EVERY_STAGE1)')


# ## Finetune it on the validation data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# decrease LR for second stage in the head\noptimizer_head.learning_rate.assign(1e-4)\n\n# split validation data into train test\nX_train, X_val, y_train, y_val = train_test_split(X_val, y_val, test_size = 0.1)\n\n# make a datasets\ntrain_dist_dataset = create_dist_dataset(X_train, y_train, training=True)\nval_dist_dataset = create_dist_dataset(X_val, y_val)\n\n# train again\ntrain(train_dist_dataset, val_dist_dataset, y_val,\n      total_steps = TOTAL_STEPS_STAGE2, \n      validate_every = VALIDATE_EVERY_STAGE2)  # not validating but printing now')


# ## Make predictions and submission

# In[ ]:


get_ipython().run_cell_magic('time', '', "sub_df['toxic'] = predict(test_dist_dataset)[:,0]\nsub_df.to_csv('submission.csv', index=False)")

