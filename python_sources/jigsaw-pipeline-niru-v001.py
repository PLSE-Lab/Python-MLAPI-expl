#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc
import re
import keras
import pickle
import string
import random
import warnings
import matplotlib
import numpy as np
import pandas as pd
import transformers
import seaborn as sns
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

from tokenizers import *
from transformers import *
from sklearn.metrics import *
from tqdm.notebook import tqdm
from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import Callback


warnings.filterwarnings("ignore")


# In[ ]:


data_path = "../input/jigsaw-multilingual-toxic-comment-classification/"
test_en_path = "../input/jigsaw-ml-laser-embed-without-cleaning-en/"
val_en_path = "../input/val-en-df/"

laser_path = "../input/jigsaw-ml-laser-embed-without-cleaning/"
use_path = "../input/jigsaw-multilingual-use-embeddings/"


# In[ ]:


PRETRAINED_TOKENIZER = 'jplu/tf-xlm-roberta-large'
PRETRAINED_MODEL     = '/kaggle/input/jigsaw-ml-xlm-roberta-finetune'


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


# ## TPU Setup

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
AUTO = tf.data.experimental.AUTOTUNE


# In[ ]:


class Config:
    seed = 42
    
    # Architecture
    n_laser = 1024
    n_use = 512
    
    laser_ft = 1024
    use_ft = 512
    logit_ft = 2048
    
    # Training
    k = 4
    
    batch_size = 16 * strategy.num_replicas_in_sync
    max_len = 192
    epochs = 1
    
    lr = 8e-6
    min_lr = 8e-6
    warmup_prop = 0.1
    weight_decay = 0.
    
config = Config()
seed_everything(config.seed)


# # Data

# In[ ]:


def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x

def write_pickle_to_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
        
def read_embed(path, name, file_name):
    em = read_pickle_from_file(path + file_name)

    columns = ['{}_{}'.format(name, i) for i in range(em.shape[1])]
    df = pd.DataFrame(em, columns=columns)
    del em  
    
    return df


# In[ ]:


jigsaw_toxic_df = pd.read_csv(data_path + "jigsaw-toxic-comment-train.csv")
jigsaw_toxic_df['lang'] = 'en'

# jigsaw_bias_df = pd.read_csv(data_path + "jigsaw-unintended-bias-train.csv")
# jigsaw_bias_df['toxic'] = jigsaw_bias_df['toxic'].round().astype(int)
# jigsaw_bias_df['lang'] = 'en'


valid_df = read_pickle_from_file(test_en_path + 'valid_en_df.pkl')
test_df = read_pickle_from_file(test_en_path + 'test_en_df.pkl')
sub_df = pd.read_csv(data_path + 'sample_submission.csv')


# In[ ]:


print(f"Jigsaw toxic : {len(jigsaw_toxic_df)} texts")
# print(f"Jigsaw bias : {len(jigsaw_bias_df)} texts")
print(f"Validation : {len(valid_df)} texts")
print(f"Test : {len(sub_df)} texts")


# In[ ]:


sns.countplot(valid_df['lang'])
plt.show()


# ## Laser Embedding

# In[ ]:


laser_toxic = read_embed(laser_path, 'laser', 'train1_em.pkl')
# laser_bias = read_embed(laser_path, 'laser', 'train2_em.pkl')

laser_test = read_embed(laser_path, 'laser', 'test_em.pkl')
laser_test_en = read_embed(test_en_path, 'laser', 'test_en_em.pkl')

laser_val = read_embed(laser_path, 'laser', 'valid_em.pkl')
laser_val_en = read_embed(test_en_path, 'laser', 'valid_en_em.pkl')

laser_columns = laser_toxic.columns
n_lasers = len(laser_columns)
print("Laser embedding dimension :", n_lasers)


# ## USE Embeddings

# In[ ]:


use_toxic = read_embed(use_path, 'use', 'train1_em.pkl')
# use_bias = read_embed(use_path, 'use', 'train2_em.pkl')

use_test = read_embed(use_path, 'use', 'test_em.pkl')
use_test_en = read_embed(use_path, 'use', 'test_en_em.pkl')

use_val = read_embed(use_path, 'use', 'valid_em.pkl')
use_val_en = read_embed(use_path, 'use', 'valid_en_em.pkl')

use_columns = use_toxic.columns
n_uses = len(use_columns)
print("Use embedding dimension :", n_uses)


# ## Remove Outliers
# - Senences that have a far-reaching meaning

# In[ ]:


# train_df = jigsaw_toxic_df.copy()
# train_df['check_embed'] = 0
# count_old = len(train_df)
# l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)

# for i in range(2):
#     index0 = np.where((train_df['toxic'].values==0) & (train_df['check_embed'].values==0))
#     index1 = np.where((train_df['toxic'].values==1) & (train_df['check_embed'].values==0))
    
#     toxic_ave_0 = np.average(use_toxic.values[index0], axis=0)
#     toxic_ave_1 = np.average(use_toxic.values[index1], axis=0)
    
#     train_df['toxic_0_l2'] = np.array(l2_dist(toxic_ave_0, use_toxic.values))
#     train_df['toxic_1_l2'] = np.array(l2_dist(toxic_ave_1, use_toxic.values))   
    
#     select_0 = train_df['toxic_0_l2'].values[index0].argsort()[::-1][:20]
#     select_1 = train_df['toxic_1_l2'].values[index1].argsort()[::-1][:5]
    
#     select_0 = [index0[0][x] for x in select_0]
#     select_1 = [index1[0][x] for x in select_1]
    
#     train_df.loc[select_0, 'check_embed'] = 1
#     train_df.loc[select_1, 'check_embed'] = 1  
    
# train_df = train_df[train_df.check_embed==0].copy()
# train_df.reset_index(drop=True, inplace=True)
# train_df.drop(['check_embed', 'toxic_0_l2', 'toxic_1_l2'], axis=1, inplace=True)    

# count_new = len(train_df)
# print(f"Removed {count_old - count_new} texts")

# jigsaw_toxic_df = train_df


# # Optimizer

# In[ ]:


import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python import ops, math_ops, state_ops, control_flow_ops
from tensorflow.python.keras import backend_config


class AdamWarmup(OptimizerV2):
    """Adam optimizer with warmup."""

    def __init__(self,
                 decay_steps,
                 warmup_steps,
                 min_lr=0.0,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 weight_decay=0.,
                 weight_decay_pattern=None,
                 amsgrad=False,
                 name='Adam',
                 **kwargs):
        r"""Construct a new Adam optimizer.
        Args:
            decay_steps: Learning rate will decay linearly to zero in decay steps.
            warmup_steps: Learning rate will increase linearly to lr in first warmup steps.
            lr: float >= 0. Learning rate.
            beta_1: float, 0 < beta < 1. Generally close to 1.
            beta_2: float, 0 < beta < 1. Generally close to 1.
            epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
            weight_decay: float >= 0. Weight decay.
            weight_decay_pattern: A list of strings. The substring of weight names to be decayed.
                                  All weights will be decayed if it is None.
            amsgrad: boolean. Whether to apply the AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                Beyond".
        """

        super(AdamWarmup, self).__init__(name, **kwargs)
        self._set_hyper('decay_steps', float(decay_steps))
        self._set_hyper('warmup_steps', float(warmup_steps))
        self._set_hyper('min_lr', min_lr)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('weight_decay', weight_decay)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad
        self._initial_weight_decay = weight_decay
        self._weight_decay_pattern = weight_decay_pattern
        
        self.current_lr = self.lr

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdamWarmup, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)

        decay_steps = self._get_hyper('decay_steps', var_dtype)
        warmup_steps = self._get_hyper('warmup_steps', var_dtype)
        min_lr = self._get_hyper('min_lr', var_dtype)
        lr_t = tf.where(
            local_step <= warmup_steps,
            lr_t * (local_step / warmup_steps),
            min_lr + (lr_t - min_lr) * (1.0 - tf.minimum(local_step, decay_steps) / decay_steps),
        )
        lr_t = (lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))

        m_t = state_ops.assign(m,
                               beta_1_t * m + (1.0 - beta_1_t) * grad,
                               use_locking=self._use_locking)

        v_t = state_ops.assign(v,
                               beta_2_t * v + (1.0 - beta_2_t) * math_ops.square(grad),
                               use_locking=self._use_locking)

        if self.amsgrad:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            var_update = m_t / (math_ops.sqrt(v_hat_t) + epsilon_t)
        else:
            var_update = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        if self._initial_weight_decay > 0.0:
            weight_decay = self._get_hyper('weight_decay', var_dtype)
            var_update += weight_decay * var
        var_update = state_ops.assign_sub(var, lr_t * var_update, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(v_hat_t)
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)

        decay_steps = self._get_hyper('decay_steps', var_dtype)
        warmup_steps = self._get_hyper('warmup_steps', var_dtype)
        min_lr = self._get_hyper('min_lr', var_dtype)
        lr_t = tf.where(
            local_step <= warmup_steps,
            lr_t * (local_step / warmup_steps),
            min_lr + (lr_t - min_lr) * (1.0 - tf.minimum(local_step, decay_steps) / decay_steps),
        )
        lr_t = (lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))

        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            var_update = m_t / (math_ops.sqrt(v_hat_t) + epsilon_t)
        else:
            var_update = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        if self._initial_weight_decay > 0.0:
            weight_decay = self._get_hyper('weight_decay', var_dtype)
            var_update += weight_decay * var
        var_update = state_ops.assign_sub(var, lr_t * var_update, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(v_hat_t)
        return control_flow_ops.group(*updates)

    def get_config(self):
        config = super(AdamWarmup, self).get_config()
        config.update({
            'decay_steps': self._serialize_hyperparameter('decay_steps'),
            'warmup_steps': self._serialize_hyperparameter('warmup_steps'),
            'min_lr': self._serialize_hyperparameter('min_lr'),
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        })
        return config


# # Model

# In[ ]:


def mixed_loss(y_true, y_pred, beta=0.10):
    loss = beta*focal_loss(y_true,y_pred) + (1-beta)*tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return loss


# In[ ]:


import tensorflow.keras.layers as KL


def nn_block(input_layer, size, dropout_rate, activation):
    out_layer = KL.Dense(size, activation=None)(input_layer)
    #out_layer = KL.BatchNormalization()(out_layer)
    out_layer = KL.Activation(activation)(out_layer)
    out_layer = KL.Dropout(dropout_rate)(out_layer)
    return out_layer

def cnn_block(input_layer, size, dropout_rate, activation):
    out_layer = KL.Conv1D(size, 1, activation=None)(input_layer)
    #out_layer = KL.LayerNormalization()(out_layer)
    out_layer = KL.Activation(activation)(out_layer)
    out_layer = KL.Dropout(dropout_rate)(out_layer)
    return out_layer

def build_model(transformer, config):
    # transformer
    input_ids = Input(shape=(config.max_len,), dtype=tf.int64, name="input_ids")
    input_masks = Input(shape=(config.max_len,), dtype=tf.int64, name="input_masks")
    input_segments = Input(shape=(config.max_len,), dtype=tf.int64, name="input_segments")
    
    sequence_output = transformer(input_ids, attention_mask=input_masks, token_type_ids=input_segments)[0]
    
    ave_pool = GlobalAveragePooling1D()(sequence_output)
    max_pool = GlobalMaxPooling1D()(sequence_output)
    
    # lasers
    lasers = Input(shape=(config.n_laser,), dtype=tf.float32, name="lasers") 
    lasers_output = nn_block(lasers,config.laser_ft,0.1,'tanh')
    #lasers_output = Dense(config.laser_ft, activation='tanh')(lasers)
    
     # uses
    uses = Input(shape=(config.n_use,), dtype=tf.float32, name="uses") 
    uses_output = nn_block(uses,config.use_ft,0.1,'tanh')

  #  uses_output = Dense(config.use_ft, activation='tanh')(uses)   
    
    features = Concatenate()([ave_pool, max_pool, KL.BatchNormalization()(lasers_output), KL.BatchNormalization()(uses_output)])
    
    outs = []
    for _ in range(5):
        x = Dropout(0.5)(features)
        x = tf.keras.layers.Dense(config.logit_ft, activation='tanh')(x)
        x = Dense(1, activation='sigmoid')(x)
        outs.append(x)
    
    out = Average()(outs)
    
    model = Model(inputs=[input_ids, input_masks, input_segments, lasers, uses], outputs=out)
#     model.compile(optimizer, loss=loss, metrics=[AUC()])
    
    return model


# In[ ]:


model = build_model(TFRobertaModel.from_pretrained(PRETRAINED_MODEL), config)

model.summary()


# # Tokenizer

# In[ ]:


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=True, 
        return_token_type_ids=True,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return [np.asarray(enc_di['input_ids'], dtype=np.int64), 
            np.asarray(enc_di['attention_mask'], dtype=np.int64), 
            np.asarray(enc_di['token_type_ids'], dtype=np.int64)]


# ## Datasets

# In[ ]:


def prepare_dataset(x, laser, use, y=None, mode="train", batch_size=16):
    if y is None:
        y = np.zeros(len(x[0]))
        
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_ids": x[0], 
            "input_masks": x[1],
            "input_segments": x[2], 
            "lasers": laser,
            "uses": use
        }, 
        y
    ))
    if mode == "train":
        dataset = dataset.repeat().shuffle(2048).batch(batch_size).prefetch(AUTO)
    elif mode == "val":
        dataset = dataset.batch(batch_size)#.cache().prefetch(AUTO)
    else: #test
        dataset = dataset.batch(batch_size)
        
    return dataset


# # Training

# In[ ]:


COLUMNS = ['id', 'comment_text', 'toxic', 'lang']


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER)


# ### Test data

# In[ ]:


x_test = regular_encode(test_df['content'].values, tokenizer, maxlen=config.max_len)
x_en_test = regular_encode(test_df['content_en'].values, tokenizer, maxlen=config.max_len)


# In[ ]:


test_dataset = prepare_dataset(x_test, laser_test.values, use_test.values, batch_size=config.batch_size, mode='test')
test_en_dataset = prepare_dataset(x_en_test, laser_test_en.values, use_test_en.values, batch_size=config.batch_size, mode='test')


# ### $k$-fold

# In[ ]:


WEIGHTS = [(1, 0), (0.9, 0.1),(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]


# In[ ]:


SELECTED_FOLDS = [2,3]


# In[ ]:


with strategy.scope():
    model = build_model(TFRobertaModel.from_pretrained(PRETRAINED_MODEL), config)
model.save_weights('model_checkpoint.h5')


# In[ ]:


train_history_list = []

test_preds  = np.zeros(len(test_df))
test_preds_en  = np.zeros(len(test_df))
pred_oof = np.zeros(len(valid_df))
pred_oof_en = np.zeros(len(valid_df))

splits = list(StratifiedKFold(n_splits=config.k, shuffle=True, random_state=config.seed).split(valid_df['toxic'].values, valid_df['toxic'].values))

for k, (train_idx, val_idx) in enumerate(splits):
    
    if k not in SELECTED_FOLDS:
        continue
        
    seed_everything(config.seed + k)
    print(f'\n\t -> Fold {k+1}\n')
    
    print(f' - Data Preparation \n')
    
    df_train = pd.concat([jigsaw_toxic_df[COLUMNS], valid_df.iloc[train_idx]])
    use_train =  pd.concat([use_toxic, use_val.iloc[train_idx]])
    laser_train =  pd.concat([laser_toxic, laser_val.iloc[train_idx]])
    
    x_train = regular_encode(df_train['comment_text'].values, tokenizer, maxlen=config.max_len)
    y_train = df_train['toxic'].values
    
    df_val = valid_df.iloc[val_idx]
    use_val_ = use_val.iloc[val_idx]
    use_val_en_ = use_val_en.iloc[val_idx]
    laser_val_ = laser_val.iloc[val_idx]
    laser_val_en_ = laser_val_en.iloc[val_idx]
    
    x_val = regular_encode(df_val['comment_text'].values, tokenizer, maxlen=config.max_len)
    x_val_en = regular_encode(df_val['comment_text_en'].values, tokenizer, maxlen=config.max_len)
    y_val = df_val['toxic'].values

    train_dataset = prepare_dataset(x_train, laser_train, use_train, y=y_train, batch_size=config.batch_size, mode='train')
    val_dataset = prepare_dataset(x_val, laser_val_, use_val_, y=y_val, batch_size=config.batch_size, mode='val')
    val_dataset_en = prepare_dataset(x_val_en, laser_val_en_, use_val_en_, y=y_val, batch_size=config.batch_size, mode='val')

    print(' - Model Preparation \n')
    
    steps_per_epoch = len(x_train[0]) // config.batch_size
    steps = steps_per_epoch * config.epochs
    
    optimizer = AdamWarmup(
        lr=config.lr, 
        min_lr=config.min_lr,
        decay_steps=steps, 
        warmup_steps=int(steps * config.warmup_prop),
        weight_decay=config.weight_decay
    )
    
    with strategy.scope():
        model.compile(optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
        model.load_weights('model_checkpoint.h5')

    print(f' - Training \n')

    train_history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        epochs=config.epochs
    )
    
    train_history_list.append(train_history)
#     model.save_weights(f'checkpoint_{k+1}.h5')
    
    print(f'\n - Predicting \n')
    
    pred_val = model.predict(val_dataset, verbose=0).reshape(-1)
    pred_oof[val_idx] = pred_val
    
    pred_val_en = model.predict(val_dataset_en, verbose=0).reshape(-1)
    pred_oof_en[val_idx] = pred_val_en 
    
    for weights in WEIGHTS:
        pred = pred_val * weights[0] + pred_val_en * weights[1]
        score = roc_auc_score(y_val, pred)
        print(f'Scored {score:.4f} with weights {weights}\n')
    
    test_preds += model.predict(test_dataset, verbose=1).reshape(-1) / 2.0
    test_preds_en += model.predict(test_en_dataset, verbose=1).reshape(-1) / 2.0
    
    del train_dataset, val_dataset, val_dataset_en, x_train, x_val, x_val_en, 
    del use_val_, use_val_en_, laser_val_, laser_val_en_, use_train, laser_train, df_train, df_val
    gc.collect()
    tf.tpu.experimental.initialize_tpu_system(tpu)
#     break


# In[ ]:


for weights in WEIGHTS:
    pred = pred_oof * weights[0] + pred_oof_en * weights[1]        
    score = roc_auc_score(valid_df['toxic'].values, pred)

    print(f' -> Local CV score is {score:.4f} for weights {weights} \n')


# In[ ]:


def history_to_dataframe( history ):
    columns = list(history.history.keys())
    datas   = list(history.history.values())
    return pd.DataFrame(np.array(datas).T, columns=columns)
   
for k, history in enumerate(train_history_list):
    df = history_to_dataframe( history )
    print('*' * 20)
    print('K:', k+1)
    print(df)


# In[ ]:


np.save("pred_test.npy", test_preds)
np.save("pred_test_en.npy", test_preds_en)
np.save("pred_oof_en.npy", pred_oof_en)
np.save("pred_oof.npy", pred_oof)


# ## Submission

# In[ ]:


for i, weights in enumerate(WEIGHTS):
    preds = test_preds * weights[0] + test_preds_en * weights[1]      
    
    sub_df['toxic'] = preds
    sub_df.to_csv(f'submission_{i}_{config.seed}.csv', index=False)


# In[ ]:


get_ipython().system('ls')

