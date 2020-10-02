#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Based on  https://www.kaggle.com/siavrez/wavenet-keras and https://www.kaggle.com/ragnar123/wavenet-with-1-more-feature

import os
import pandas as pd
import numpy as np
import random


import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, Dense, Add, Multiply
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models, optimizers

from numba import cuda

import gc

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn import preprocessing


# In[ ]:


""" TF-Keras SWA: callback utility for performing stochastic weight averaging (SWA).
"""

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization

class SWA(Callback):
    """ Stochastic Weight Averging.
    # Paper
        title: Averaging Weights Leads to Wider Optima and Better Generalization
        link: https://arxiv.org/abs/1803.05407
    # Arguments
        start_epoch:   integer, epoch when swa should start.
        lr_schedule:   string, type of learning rate schedule.
        swa_lr:        float, learning rate for swa sampling.
        swa_lr2:       float, upper bound of cyclic learning rate.
        swa_freq:      integer, length of learning rate cycle.
        verbose:       integer, verbosity mode, 0 or 1.
    """
    def __init__(self,
                 start_epoch,
                 lr_schedule='manual',
                 swa_lr='auto',
                 swa_lr2='auto',
                 swa_freq=1,
                 verbose=0):
                 
        super(SWA, self).__init__()
        self.start_epoch = start_epoch - 1
        self.lr_schedule = lr_schedule
        self.swa_lr = swa_lr
        self.swa_lr2 = swa_lr2
        self.swa_freq = swa_freq
        self.verbose = verbose

        if start_epoch < 2:
            raise ValueError('"swa_start" attribute cannot be lower than 2.')

        schedules = ['manual', 'constant', 'cyclic']

        if self.lr_schedule not in schedules:
            raise ValueError('"{}" is not a valid learning rate schedule'                              .format(self.lr_schedule))

        if self.lr_schedule == 'cyclic' and self.swa_freq < 2:
            raise ValueError('"swa_freq" must be higher than 1 for cyclic schedule.')

        if self.swa_lr == 'auto' and self.swa_lr2 != 'auto':
            raise ValueError('"swa_lr2" cannot be manually set if "swa_lr" is automatic.') 
            
        if self.lr_schedule == 'cyclic' and self.swa_lr != 'auto'            and self.swa_lr2 != 'auto' and self.swa_lr > self.swa_lr2:
            raise ValueError('"swa_lr" must be lower than "swa_lr2".')

    def on_train_begin(self, logs=None):

        self.epochs = self.params.get('epochs')

        if self.start_epoch >= self.epochs - 1:
            raise ValueError('"swa_start" attribute must be lower than "epochs".')

        self.init_lr = K.eval(self.model.optimizer.lr)

        # automatic swa_lr
        if self.swa_lr == 'auto':
            self.swa_lr = 0.1*self.init_lr
        
        if self.init_lr < self.swa_lr:
            raise ValueError('"swa_lr" must be lower than rate set in optimizer.')

        # automatic swa_lr2 between initial lr and swa_lr   
        if self.lr_schedule == 'cyclic' and self.swa_lr2 == 'auto':
            self.swa_lr2 = self.swa_lr + (self.init_lr - self.swa_lr)*0.25

        self._check_batch_norm()

    def on_epoch_begin(self, epoch, logs=None):

        self.current_epoch = epoch
        self._scheduler(epoch)

        # constant schedule is updated epoch-wise
        if self.lr_schedule == 'constant' or self.is_batch_norm_epoch:
            self._update_lr(epoch)

        if self.is_swa_start_epoch:
            self.swa_weights = self.model.get_weights()

            if self.verbose > 0:
                print('\nEpoch %05d: starting stochastic weight averaging'
                      % (epoch + 1))

        if self.is_batch_norm_epoch:
            self._set_swa_weights(epoch)

            if self.verbose > 0:
                print('\nEpoch %05d: reinitializing batch normalization layers'
                      % (epoch + 1))

            self._reset_batch_norm()

            if self.verbose > 0:
                print('\nEpoch %05d: running forward pass to adjust batch normalization'
                      % (epoch + 1))

    def on_batch_begin(self, batch, logs=None):

        # update lr each batch for cyclic lr schedule
        if self.lr_schedule == 'cyclic':
            self._update_lr(self.current_epoch, batch)

        if self.is_batch_norm_epoch:

            batch_size = self.params['samples']
            momentum = batch_size / (batch*batch_size + batch_size)

            for layer in self.batch_norm_layers:
                layer.momentum = momentum

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.eval(self.model.optimizer.lr)
        for k, v in logs.items():
            if k == 'lr':
                self.model.history.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):

        if self.is_swa_start_epoch:
            self.swa_start_epoch = epoch

        if self.is_swa_epoch and not self.is_batch_norm_epoch:
            self.swa_weights = self._average_weights(epoch)

    def on_train_end(self, logs=None):

        if not self.has_batch_norm:
            self._set_swa_weights(self.epochs)
        else:
            self._restore_batch_norm()

    def _scheduler(self, epoch):

        swa_epoch = (epoch - self.start_epoch)

        self.is_swa_epoch = epoch >= self.start_epoch and swa_epoch % self.swa_freq == 0
        self.is_swa_start_epoch = epoch == self.start_epoch
        self.is_batch_norm_epoch = epoch == self.epochs - 1 and self.has_batch_norm

    def _average_weights(self, epoch):

        return [(swa_w * (epoch - self.start_epoch) + w)
                / ((epoch - self.start_epoch) + 1)
                for swa_w, w in zip(self.swa_weights, self.model.get_weights())]

    def _update_lr(self, epoch, batch=None):

        if self.is_batch_norm_epoch:
            lr = 0
            K.set_value(self.model.optimizer.lr, lr)
        elif self.lr_schedule == 'constant':
            lr = self._constant_schedule(epoch)
            K.set_value(self.model.optimizer.lr, lr)
        elif self.lr_schedule == 'cyclic':
            lr = self._cyclic_schedule(epoch, batch)
            K.set_value(self.model.optimizer.lr, lr)

    def _constant_schedule(self, epoch):

        t = epoch / self.start_epoch
        lr_ratio = self.swa_lr / self.init_lr
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return self.init_lr * factor

    def _cyclic_schedule(self, epoch, batch):
        """ Designed after Section 3.1 of Averaging Weights Leads to
        Wider Optima and Better Generalization(https://arxiv.org/abs/1803.05407)
        """
        # steps are mini-batches per epoch, equal to training_samples / batch_size
        steps = self.params.get('steps')
        
        #occasionally steps parameter will not be set. We then calculate it ourselves
        if steps == None:
            steps = self.params['samples'] // self.params['batch_size']
        
        swa_epoch = (epoch - self.start_epoch) % self.swa_freq
        cycle_length = self.swa_freq * steps

        # batch 0 indexed, so need to add 1
        i = (swa_epoch * steps) + (batch + 1)
        if epoch >= self.start_epoch:
            t = (((i-1) % cycle_length) + 1)/cycle_length
            return (1-t)*self.swa_lr2 + t*self.swa_lr
        else:
            return self._constant_schedule(epoch)

    def _set_swa_weights(self, epoch):

        self.model.set_weights(self.swa_weights)

        if self.verbose > 0:
            print('\nEpoch %05d: final model weights set to stochastic weight average'
                  % (epoch + 1))

    def _check_batch_norm(self):

        self.batch_norm_momentums = []
        self.batch_norm_layers = []
        self.has_batch_norm = False
        self.running_bn_epoch = False

        for layer in self.model.layers:
            if issubclass(layer.__class__, BatchNormalization):
                self.has_batch_norm = True
                self.batch_norm_momentums.append(layer.momentum)
                self.batch_norm_layers.append(layer)

        if self.verbose > 0 and self.has_batch_norm:
            print('Model uses batch normalization. SWA will require last epoch '
                  'to be a forward pass and will run with no learning rate')

    def _reset_batch_norm(self):

        for layer in self.batch_norm_layers:

            # to get properly initialized moving mean and moving variance weights
            # we initialize a new batch norm layer from the config of the existing
            # layer, build that layer, retrieve its reinitialized moving mean and
            # moving var weights and then delete the layer
            bn_config = layer.get_config()
            new_batch_norm = BatchNormalization(**bn_config)
            new_batch_norm.build(layer.input_shape)
            new_moving_mean, new_moving_var = new_batch_norm.get_weights()[-2:]
            # get rid of the new_batch_norm layer
            del new_batch_norm
            # get the trained gamma and beta from the current batch norm layer
            trained_weights = layer.get_weights()
            new_weights = []
            # get gamma if exists
            if bn_config['scale']:
                new_weights.append(trained_weights.pop(0))
            # get beta if exists
            if bn_config['center']:
                new_weights.append(trained_weights.pop(0))
            new_weights += [new_moving_mean, new_moving_var]
            # set weights to trained gamma and beta, reinitialized mean and variance
            layer.set_weights(new_weights)

    def _restore_batch_norm(self):

        for layer, momentum in zip(self.batch_norm_layers, self.batch_norm_momentums):
            layer.momentum = momentum


# In[ ]:


# configurations and main hyperparammeters
EPOCHS = 250
NNBATCHSIZE = 16
GROUP_BATCH_SIZE = 4000
SEED = 321
LR = 0.001
SPLITS = 5


# In[ ]:


def read_data():
    train = pd.read_csv('../input/1-remove-drift-ac/train_feat_100k.csv.gz', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv('../input/1-remove-drift-ac/test_feat_100k.csv.gz', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})
    return train, test, sub

# create batches of 4000 observations
def batching(df, batch_size):
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

def normalize(train,test,features):
    scal = preprocessing.StandardScaler()
    scal.fit(train[features])
    train[features] = scal.transform(train[features])
    test[features] = scal.transform(test[features])
    return train, test


# In[ ]:


def reduce_mem_usage(df: pd.DataFrame,
                     verbose: bool = True) -> pd.DataFrame:
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if (c_min > np.iinfo(np.int32).min
                      and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem

    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'
    if verbose:
        print(msg)

    return df


# In[ ]:


def Classifier(shape_):
    
    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = Add()([res_x, x])
        return res_x
    
    inp = Input(shape = (shape_))
    
    x = wave_block(inp, 16, 3, 12)
    x1 = wave_block(x, 32, 3, 8)
    x2 = wave_block(x1, 64, 3, 4)
    x3 = wave_block(x2, 128, 3, 1)
    
    
    out = Dense(11, activation = 'softmax', name = 'out')(x3)
    
    model = models.Model(inputs = inp, outputs = out)
    
    opt = Adam(lr = LR)
    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])
    return model

# function that decrease the learning as epochs increase (i also change this part of the code)
def lr_schedule(epoch):
    if epoch < 30:
        lr = LR
    elif epoch < 40:
        lr = LR / 3
    elif epoch < 50:
        lr = LR / 5
    elif epoch < 60:
        lr = LR / 7
    elif epoch < 70:
        lr = LR / 9
    elif epoch < 80:
        lr = LR / 11
    elif epoch < 90:
        lr = LR / 13
    elif epoch < 120:
        lr = LR / 15
    else:
        lr = LR / 17
    return lr

# class to get macro f1 score. This is not entirely necessary but it's fun to check f1 score of each epoch (be carefull, if you use this function early stopping callback will not work)
class MacroF1(Callback):
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis = 2).reshape(-1)
        
    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis = 2).reshape(-1)
        score = f1_score(self.targets, pred, average = 'macro')
        print(f'F1 Macro Score: {score:.5f}')
        

start_epoch = 200

# define swa callback
swa = SWA(start_epoch=start_epoch, 
          lr_schedule='manual',
          swa_lr=0.001, 
          verbose=1)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


# In[ ]:


# main function to perfrom groupkfold cross validation (we have 1000 vectores of 4000 rows and 8 features (columns)). Going to make 5 groups with this subgroups.
def run_cv_model_by_batch(train, test, splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size):
    seed_everything(SEED)
    K.clear_session()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
      
    oof_ = np.zeros((len(train), 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)
    preds_ = np.zeros((len(test), 11))
    target = ['open_channels']
    group = train['group']
    kf = GroupKFold(n_splits=5)
    splits = [x for x in kf.split(train, train[target], group)]

    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])    
        new_splits.append(new_split)
    # pivot target columns to transform the net to a multiclass classification estructure (you can also leave it in 1 vector with sparsecategoricalcrossentropy loss function)
    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)

    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']
    target_cols = ['target_'+str(i) for i in range(11)]
    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values.astype(np.float32))))
    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))
    
    gc.collect()
    
    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):
        train_x, train_y = train[tr_idx], train_tr[tr_idx]
        valid_x, valid_y = train[val_idx], train_tr[val_idx]
        print(f'Our training dataset shape is {train_x.shape}')
        print(f'Our validation dataset shape is {valid_x.shape}')

        shape_ = (None, train_x.shape[2]) # input is going to be the number of feature we are using (dimension 2 of 0, 1, 2)
        model = Classifier(shape_)
        # using our lr_schedule function
        cb_lr_schedule = LearningRateScheduler(lr_schedule)
        model.fit(train_x,train_y,
                  epochs = nn_epochs,
                  callbacks = [swa,cb_lr_schedule],# MacroF1(model, valid_x, valid_y)], # adding custom evaluation metric for each epoch
                  batch_size = nn_batch_size,verbose = 2,
                  validation_data = (valid_x,valid_y))
        gc.collect()
        preds_f = model.predict(valid_x)
        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro') # need to get the class with the biggest probability
        print(f'Training fold {n_fold + 1} completed. macro f1 score : {f1_score_ :1.5f}')
        preds_f = preds_f.reshape(-1, preds_f.shape[-1])
        oof_[val_orig_idx,:] += preds_f
        te_preds = model.predict(test)
        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           
        preds_ += te_preds / SPLITS
    
        
    # calculate the oof macro f1_score
    f1_score_ = f1_score(np.argmax(train_tr, axis = 2).reshape(-1),  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)
    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')
    
    np.save(f'./oof.{f1_score_:1.5f}.npy',oof_)
    np.save(f'./preds.{f1_score_:1.5f}.npy',preds_)
    
    
    sample_submission['open_channels'] = np.argmax(preds_, axis = 1).astype(int)
    sample_submission.to_csv(f'./submission_wavenet.{f1_score_:1.5f}.csv', index=False, float_format='%.4f')


# In[ ]:


print('Reading Data Started...')
train, test, sample_submission = read_data()
train=train[0:5004000]
gc.collect()


# In[ ]:


features=['n','pt', 'm2am_50', 'm2am_100', 'm2am_500', 'P3', 'm2am_1000', 'signal', 'm2am_10', 'm2am_10000',
     'P0', 'P1', 'P4', 'sdam_10000', 'P2', 'm2am_5', 'P6', 'P9', 'P5', 'P8', 'P7', 'P10', 'sdm_5000',
     'sdm_500', 'sdam_1000', 'lma_2', 'lmi_3', 'lmi_2', 'sdam_5000', 'm2am_5000', 'm2_2', 'lmi_5', 
     'sdm_10000', 'lmed_2', 'lead_1', 'lma_3', 'm2am_4', 'pt1', 'lmi_4', 'leadm_1', 'vp', 'lmim_2',
     'lma_50', 'lmi_10', 'lma_10', 'lmim_10', 'lmed_3', 'sdm_10', 'vp1', 'm2m_10000', 'sdam_50']

GROUP_BATCH_SIZE = 4000

train, test = normalize(train,test,['signal'])
gc.collect()

test=test[features]
features.append('open_channels')
train=train[features]

train = batching(train, batch_size = GROUP_BATCH_SIZE)
test  = batching(test , batch_size = GROUP_BATCH_SIZE)

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]

gc.collect()


# In[ ]:


print(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started...')
run_cv_model_by_batch(train, test, SPLITS, 'group', features, sample_submission, EPOCHS, NNBATCHSIZE)
print('Training completed...')

