#!/usr/bin/env python
# coding: utf-8

# ## About this notebook
# 
# 
# I have seen many great notebooks using Pytorch and the Keras built in training loop, but I wanted to share one which uses a custom training loop in TensorFlow 2.
# 
# I hope this starter will allow more people to start experimenting with their unique ideas for tweaking.
# 
# As an example, a custom loop allows us to use the exact AUC for validation instead of the  (very convenient) [approximate value used in Keras](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC). The two values may differ if predictions are close to each other, and not uniformly distributed (both are happening here).
# 
# ** Note: This notebook tries to be simple, and only uses a small amount of data, and it does not use translated datasets or other tricks. You need to add those yourself to squeeze out a good score.**
# 
# Suggestions/improvements are appreciated!
# 
# ---
# 
# ### References:
# 
# 
# - This notebook heavily relies on the great [notebook]((https://www.kaggle.com/xhlulu//jigsaw-tpu-xlm-roberta) by, Xhulu: [@xhulu](https://www.kaggle.com/xhulu/) 
# - The tensorflow distrubuted training tutorial: [Link](https://www.tensorflow.org/tutorials/distribute/custom_training)

# In[ ]:


MAX_LEN = 192  #Reduced for quicker execution
LR = 1e-5
BATCH_SIZE = 16 # per TPU core
TOTAL_STEPS_STAGE1 = 300
VALIDATE_EVERY_STAGE1 = 100
TOTAL_STEPS_STAGE2 = 200
VALIDATE_EVERY_STAGE2 = 100

PRETRAINED_MODEL = 'jplu/tf-xlm-roberta-large'
D = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/'

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

# In[ ]:


train_df = pd.read_csv(D+'jigsaw-toxic-comment-train.csv')
val_df = pd.read_csv(D+'validation.csv')
test_df = pd.read_csv(D+'test.csv')
sub_df = pd.read_csv(D+'sample_submission.csv')

# subsample the train dataframe to 50%-50%
train_df = pd.concat([
    train_df.query('toxic==1'),
    train_df.query('toxic==0').sample(sum(train_df.toxic),random_state=42)
])
# shufle it just to make sure
train_df = train_df.sample(frac=1, random_state = 42)


# ## Tokenize  it with the models own tokenizer
# 
# - Note it takes some time!
# - Note, we need to reshape the targets

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef regular_encode(texts, tokenizer, maxlen=512):\n    enc_di = tokenizer.batch_encode_plus(\n        texts, \n        return_attention_masks=False, \n        return_token_type_ids=False,\n        pad_to_max_length=True,\n        max_length=maxlen\n    )\n    \n    return np.array(enc_di['input_ids'])\n    \n\ntokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)\nX_train = regular_encode(train_df.comment_text.values, tokenizer, maxlen=MAX_LEN)\nX_val = regular_encode(val_df.comment_text.values, tokenizer, maxlen=MAX_LEN)\nX_test = regular_encode(test_df.content.values, tokenizer, maxlen=MAX_LEN)\n\ny_train = train_df.toxic.values.reshape(-1,1)\ny_val = val_df.toxic.values.reshape(-1,1)")


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
# - Note: Downloading the model takes some time!

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef create_model_and_optimizer():\n    with strategy.scope():\n        transformer_layer = TFAutoModel.from_pretrained(PRETRAINED_MODEL)                \n        model = build_model(transformer_layer)\n        optimizer = tf.keras.optimizers.Adam(learning_rate=LR, epsilon=1e-08)\n    return model, optimizer\n\n\ndef build_model(transformer):\n    inp = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")\n    # Huggingface transformers have multiple outputs, embeddings are the first one\n    # let\'s slice out the first position, the paper says its not worse than pooling\n    x = transformer(inp)[0][:, 0, :]  \n    out = Dense(1, activation=\'sigmoid\')(x)\n    model = Model(inputs=[inp], outputs=[out])\n    \n    return model\n\n\nmodel, optimizer = create_model_and_optimizer()\nmodel.summary()')


# ### Define stuff for the custom training loop
# 
# We will need:
# - 1, losses, and  optionally a training AUC metric here: these need to be defined in the scope of th distributed strategy. 
# - 2, A full training loop
# - 3, A distributed train step called in the training loop, which uses a single replica train step
# - 4, A prediction loop with dstibute 
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


get_ipython().run_cell_magic('time', '', '# make a new dataset for training with the validation data \n# with targets, shuffling and repeating\nval_dist_dataset_4_training = create_dist_dataset(X_val, y_val, training=True)\n\n# train again\ntrain(val_dist_dataset_4_training,\n      total_steps = TOTAL_STEPS_STAGE2, \n      validate_every = VALIDATE_EVERY_STAGE2)  # not validating but printing now')


# ## Make predictions and submission

# In[ ]:


get_ipython().run_cell_magic('time', '', "sub_df['toxic'] = predict(test_dist_dataset)[:,0]\nsub_df.to_csv('submission.csv', index=False)")

