#!/usr/bin/env python
# coding: utf-8

# ### To pick samples for training and validation use the Adversarial Validation, the idea successfully used often across Kaggle (e.g. [Quora Adversarial Validation](https://www.kaggle.com/tunguz/quora-adversarial-validation) and [Adversarial validation](https://www.kaggle.com/konradb/adversarial-validation)) - the idea of training a model that predicts by how much a given sample is different from the test set.
# 
# - The public notebook used as a base: Xhlulu @xhlulu - [Jigsaw TPU: XLM-Roberta](https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta)
# 
# - Roberta XML-Base Model - This notebook run both on TPU (for x minutes - this version run on TPU) and Kaggle GPU (for up to 7 hours)
# 
# 
# ### The workflow:
# - Label test data set as 1 and the training set as 0. The training set consists of the randomly sampled Michael Kazachok's [translations](https://www.kaggle.com/miklgr500/jigsaw-train-multilingual-coments-google-api) to 6 languages of train1 (toxic-comment-calssification-training-data), of train2 (unbiased-bias-training-data) and val_8k (validation.csv) that had been encoded in a separate notebook without use of GPU/TPU;
# - Train classification task with pretrained XLM-Roberta-base model (make sure the same sample's translations are in the same fold);
# - Save the model and the oof predictions of the train1, train2, val_8k;
# - Out of this notebook: predict all train1 and train2 samples using  this model (except for those rows used for training)
# - Use those predictions (and the oof predictions of those sampled used for training) to rank samples as to how much they are similar to test set.
# - use threshold 'probability' to sample the desired number of samples for training and adding to validation set
# 
# ### Observations:
# 
# - most of the translations are easily separated from test set - validation set auc = 0.98 => they are very different from test set.
# - there are about only 4k samples in train1 that has predictions ('probability' of being from test set according to classifier) larger than 0.4 which can be added to validation set; most of the samples are well below 0.001
# - selecting samples this way, at least, does not make the performance worse than lucky picks of random sampling
# 
# 
# 

# In[ ]:


import os, time, gc

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm

from sklearn.model_selection import GroupKFold

osj = os.path.join; osdir = os.listdir


# ## TPU Configs

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Detect hardware, return appropriate distribution strategy\ndef tpu_init():\n    try:\n        # TPU detection. No parameters necessary if TPU_NAME environment variable is\n        # set: this is always the case on Kaggle.\n        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()\n        print(\'Running on TPU \', tpu.master())\n    except ValueError:\n        tpu = None\n\n    if tpu:\n        tf.config.experimental_connect_to_cluster(tpu)\n        tf.tpu.experimental.initialize_tpu_system(tpu)\n        strategy = tf.distribute.experimental.TPUStrategy(tpu)\n    else:\n        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.\n        strategy = tf.distribute.get_strategy()\n\n    print("REPLICAS: ", strategy.num_replicas_in_sync)\n\n    return strategy')


# In[ ]:


def build_model(transformer, max_len=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    
    cls_token = sequence_output[:, 0, :]
    
    out = Dense(1, activation='sigmoid')(cls_token)
        
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy',
                                      metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model


# In[ ]:


def tpu_bs(BATCH_SIZE_MULTIPLIER):
    strategy = tpu_init()
    bs = BATCH_SIZE_MULTIPLIER * strategy.num_replicas_in_sync
    
    return strategy, bs


# In[ ]:


debug = False  # True # False
n_rows = 100_000_000 if not debug else 16*8*2

n_splits = 4
# load models of folds 0,1,4
folds_to_train = [0,1]
folds_to_save = [0,1]
seed_num = 1
seed_model = 2020

AUTO = tf.data.experimental.AUTOTUNE

# Configuration
epochs = 2

BATCH_SIZE_MULTIPLIER = 24  # 32  # 24  # 16
MAX_LEN = 192
MODEL = 'jplu/tf-xlm-roberta-base'

out_path = './'
assert os.path.exists(out_path)

datetime_str = time.strftime("%d_%m_time_%H_%M", time.localtime())

t0 = time.time()

def keras_seed_everything(seed):
    # import tensorflow as tf
    # import os
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

keras_seed_everything(seed_model)

strategy, bs = tpu_bs(BATCH_SIZE_MULTIPLIER)

tokenizer = AutoTokenizer.from_pretrained(MODEL)


# ## Load data
# 

# In[ ]:


train_trans = pd.read_csv(
    '../input/jig-ds-noneng-raw-begin-data-maxl-192/train_trans_raw_begin_enc_rows_1326360_maxl_192.csv', nrows=n_rows)

train_trans = train_trans.sample(n=60_000 if not debug else 60, random_state=seed_num)

train_trans = train_trans[train_trans['df_name']=='trans']

val_8k = pd.read_csv(
    '../input/jig-ds-noneng-raw-begin-data-maxl-192/val_8k_raw_begin_enc_nrows_63812_maxl_192.csv', nrows=n_rows)

test = pd.read_csv(
    '../input/jig-ds-noneng-raw-begin-data-maxl-192/test_raw_begin_enc_nrows_63812_maxl_192.csv', nrows=n_rows)

train2_trans = pd.read_csv('../input/jig-train2-trans-enc-raw-similar-to-test-preds/train2_trans_similar_test_nrows_838658.csv')
train2_trans = train2_trans.sample(n=176_000 if not debug else 70, random_state=seed_num)
train2_trans.drop('val_fold', axis=1, inplace=True)
train2_trans['toxic'] = (train2_trans['toxic']>0.5).astype(int)

train2_trans['df_name'] = 'tr2_trans'
val_8k['df_name']='val_8k'
test['df_name'] = 'test'

enc_cols = [col for col in test.columns if col.startswith('enc_')]
cols_select = ['id','lang','df_name','toxic'] + enc_cols
train_trans = pd.concat([train_trans[cols_select], train2_trans[cols_select]])
train_trans.head(2)


# In[ ]:


def print_df_stats(df, df_name='df', text_col = 'comment_text'):
    print("="*30)
    print(f"\n{df_name}.shape:", df.shape)
    if 'toxic' in df.columns:
        print(f"\n{df_name}['toxic'].value_counts:\n", df['toxic'].value_counts())
    if 'lang' in df.columns:
        print(f"\n{df_name}['lang'].value_counts:\n", df['lang'].value_counts())
    if 'target' in df.columns:
        print(f"\n{df_name}['target'].value_counts:\n", df['target'].value_counts())

#print_df_stats(osub, 'osub');
print_df_stats(train_trans, 'train_trans'); print_df_stats(val_8k,'val_8k'); print_df_stats(test,'test', text_col = 'content') 


# In[ ]:


enc_cols = [col for col in test.columns if col.startswith('enc_')]

train_trans['target']=0
val_8k['target']=0

test['target']=1

sel_cols = ['id','lang','df_name','target']+enc_cols
train = pd.concat([train_trans[sel_cols], val_8k[sel_cols], test[sel_cols]])

train = train.sample(frac=1, replace=False, random_state=seed_num)
print_df_stats(train, 'train')
del train_trans, test, val_8k; _=gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'x_train = train[enc_cols].values.astype(\'int\')[:n_rows]\ny_train = train[\'target\'].values.astype(\'int\')[:n_rows]\n\ntrain.drop(enc_cols, axis=1, inplace=True)\ntrain = train[:n_rows]\n#x_test = test[enc_cols].values.astype(\'int\')\n\nprint("x_train.shape", x_train.shape)\n\ntrain[\'lang\'] = train[\'lang\'].astype(\'category\')\ntrain[\'target\'] = train[\'target\'].astype(\'int\')\ntrain.dtypes\n\nt_preproc = time.time()\nprint(f"Finished preprocessing in {(t_preproc-t0)/60:.2f} min.")')


# In[ ]:


def tpu_bs(BATCH_SIZE_MULTIPLIER):
    strategy = tpu_init()
    bs = BATCH_SIZE_MULTIPLIER * strategy.num_replicas_in_sync
    
    return strategy, bs

strategy, bs = tpu_bs(BATCH_SIZE_MULTIPLIER)


# ## Divide train to groups: the same 'id' - one group

# In[ ]:


id_to_group = {k: v for (k,v) in zip(train['id'].unique(), range(len(train['id'].unique())))}
groups = train['id'].map(id_to_group).values
del id_to_group; _=gc.collect()
train['id'].value_counts()


# ## Train Model

# In[ ]:


train['preds'] = 999.99
hist1_ls, hist2_ls = [], []
t0 = time.time()

gkf = GroupKFold(n_splits=n_splits)

for fold, (train_idx, valid_idx) in enumerate(gkf.split(x_train, y_train, groups)):
    
    if not (fold in folds_to_train):
        continue
        
    t1 = time.time()
    
    train_dataset = ( tf.data.Dataset.from_tensor_slices((x_train[train_idx], y_train[train_idx]))
                                            .repeat().shuffle(2048).batch(bs, drop_remainder=True).prefetch(AUTO)
                    )
    valid_dataset = (  tf.data.Dataset.from_tensor_slices((x_train[valid_idx],
                                                           y_train[valid_idx]))
                            .batch(bs).prefetch(AUTO) )
    
    n_steps = int( max(1, x_train[train_idx].shape[0] // bs) )
    
    print(f"1: Num train samples {len(x_train[train_idx])}, num valid samples = {len(x_train[valid_idx])}")
    
    with strategy.scope():
        transformer_layer = TFAutoModel.from_pretrained(MODEL)
        model = build_model(transformer_layer, max_len=MAX_LEN)
        
    train_history_1 = model.fit(train_dataset,
                            steps_per_epoch=n_steps,
                            validation_data=valid_dataset,
                            epochs=epochs)

    
    t2 = time.time()
    print(f"\nTrained fold {fold}, in {(t2-t1)/60:.2f} min.")
   
    
    hist1_df = pd.DataFrame(train_history_1.history)
    hist1_ls.append(hist1_df)
    
    # save model
            
    new_model_filename = f'model_fl{fold}_auc_{hist1_df.iloc[-1,-1]:.5f}_eps_{epochs}.h5'
    checkpoint_path_fn = os.path.join(out_path, new_model_filename)
    if fold in folds_to_save:
        model.save_weights(checkpoint_path_fn)
        print(f"Saved fold {fold} weights")
    
    t3 = time.time()
    
    train['preds'].iloc[valid_idx]  =  model.predict(valid_dataset, verbose=1).squeeze()
    train[['id','lang','preds']].to_csv(f'train_preds_after_fold_{fold}.csv', index=False)
    t4 = time.time()
    print(f"Predicted fold {fold} valid_idx - English in {(t4-t3)/60:.2f} min.")
    
    t5 = time.time()
    print(f"Predicted fold {fold} valid_idx - English in {(t5-t4)/60:.2f} min.")
    
    print(f"\nFOLD {fold}, TOTAL TIME: {(time.time()-t1)/60:.2f} min. \n =================== END of fold {fold} ====================\n")
    
    if fold != folds_to_train[-1]:
        #print(model.summary())
        del model;  _=gc.collect()

        strategy, bs = tpu_bs(BATCH_SIZE_MULTIPLIER)
        tf.keras.backend.clear_session()


# In[ ]:


hist_df = pd.DataFrame( np.concatenate([h.values for h in hist1_ls]), columns = hist1_ls[0].columns ,
                      )
hist_df.to_csv('hist_df.csv', index=False)
hist_df


# In[ ]:


# leave train with predictions and remove test
train_valid = train.loc[(train['preds']<=1)&(train['target']==0), 
                            ['id', 'lang','df_name','target','preds']]
print_df_stats(train_valid, df_name='train_valid')
train_valid.head(10)


# In[ ]:


train_valid['df_name'].value_counts()


# In[ ]:


train[train['df_name']=='test'].sort_values(by='preds').head(10)


# In[ ]:


train_valid['preds'].hist(bins=100)


# ### Distribution of predictions

# In[ ]:


thresholds = [1e-7, 1e-5, 1e-4, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7]

for prob_thresh in thresholds:
    train_valid_thresh = train_valid[train_valid['preds']>prob_thresh]
    #train_valid_thresh.to_csv(f"train_valid_thresh_{str(prob_thresh).replace('.','_')}_nrows_{train_valid_thresh.shape[0]}.csv")
    print(f"thresh={prob_thresh}: \tnum samples with preds > thresh : = {train_valid_thresh.shape[0]:,d}")
    #      .describe()
    #print(train_valid_thresh['lang'].train_validue_counts())
    print(train_valid_thresh['df_name'].value_counts())
    #print("\nLanguages train_validue_counts in 'train' train_valid_thresh:")
    #print(train_valid_thresh.loc[train_valid_thresh['df_name']=='train', 'lang'].train_validue_counts())
    print("="*30)


# In[ ]:


# save train_valid
train_valid[['id','lang','df_name','target','preds']].to_csv(f'train_valid_folds_{folds_to_train}.csv', index=False)


# In[ ]:


get_ipython().system('du -ha ./')


# In[ ]:




