#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import pyarrow.parquet as pq
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import gc

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Creating a CNN that works using TensorFlow's Estimator and Dataset API. First let's make some constants available.

# In[ ]:


class Config:
    META_TRAIN='../input/metadata_train.csv'
    META_TEST='../input/metadata_test.csv'
    TRAIN_SRC='../input/train.parquet'
    TEST_SRC='../input/test.parquet'
    TRAIN_ARRAYS_STRATIFIED = 'train.npz'
    TEST_ARRAYS = 'test.npz'


# Let's define the Matthews correlation coefficient (MCC) in both numpy and tensorlfow

# In[ ]:


# in all functions below use keyword argument use_np to use numpy, 
# if False then tensorflow is used!

def _is_equal(a, b, use_np=False):
    # Equality or not, returned in type Float
    if use_np:
        return np.equal(a, b).astype(np.float32)
    return tf.cast(tf.equal(a, b), tf.float32)

def _is_not_equal(a, b, use_np=False):
    # non-Equality or not, returned in type Float
    if use_np:
        return np.not_equal(a, b).astype(np.float32)
    return tf.cast(tf.not_equal(a, b), tf.float32)

def true_positives(y, y_preds, use_np=False):
    correct_preds = _is_equal(y, y_preds, use_np=use_np)
    poss = _is_equal(y, 1, use_np=use_np)
    if use_np:
        return np.sum(correct_preds * poss)
    return tf.reduce_sum(correct_preds * poss)

def true_negatives(y, y_preds, use_np=False):
    correct_preds = _is_equal(y, y_preds, use_np=use_np)
    negs = _is_equal(y, 0, use_np=use_np)
    if use_np:
        return np.sum(correct_preds * negs)
    return tf.reduce_sum(correct_preds * negs)

def false_positives(y, y_preds, use_np=False):
    incorrect_preds = _is_not_equal(y, y_preds, use_np=use_np)
    negs = _is_equal(y, 0, use_np=use_np)
    if use_np:
        return np.sum(incorrect_preds * negs)
    return tf.reduce_sum(incorrect_preds * negs)

def false_negatives(y, y_preds, use_np=False):
    incorrect_preds = _is_not_equal(y, y_preds, use_np=use_np)
    poss = _is_equal(y, 1, use_np=use_np)
    if use_np:
        return np.sum(incorrect_preds * poss)
    return tf.reduce_sum(incorrect_preds * poss)

def _get_inter_metrics(y, y_preds, use_np=False):
    tp = true_positives(y, y_preds, use_np=use_np)
    tn = true_negatives(y, y_preds, use_np=use_np)
    fp = false_positives(y, y_preds, use_np=use_np)
    fn = false_negatives(y, y_preds, use_np=use_np)
    return tp, tn, fp, fn

def mcc(y, y_preds, use_np=False):
    tp, tn, fp, fn = _get_inter_metrics(y, y_preds, use_np=use_np)
    num = (tp * tn) - (fp * fn)
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if use_np:
        den_sqrt = np.sqrt(den + 1e-7)
    else:
        den_sqrt = tf.sqrt(den + 1e-7)
    return num / den_sqrt


# Preprocessing the data. This is inspired from https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694
# 
# I have tweaked the code to make sure that each phase gives a separate target. Since we have 3 phases so we will have three channels in train input and 3 labels in each data point.
# 
# In my base model my stratefied split got comparable performance on public score. So, I'll be using stratified splitting. I have three classes in stratified split. class 0 all the three labels are 0, class 1 all the three labels are 1 and class 2 the three labels are different

# In[ ]:


def read_table(src, start, end):
    df = pq.read_pandas(src, columns=[str(i) for i in range(start, end)]).to_pandas()
    return df.values

def _standardize(data, min, max):
    return (data - min) / (max - min)

def normalize(data, cur_min_max, new_min_max):
    standardized = _standardize(data,
                                cur_min_max[0],
                                cur_min_max[1])
    new_min, new_max = new_min_max
    return standardized * (new_max - new_min) + new_min

def transform_and_binify(data,
                         sample_size,
                         bins=160,
                         cur_min_max=(-128, 127),
                         min_max=(-1, 1)):
    min, max = cur_min_max

    data_normed = normalize(data, (min, max), min_max)
    bucket_size = int(sample_size / bins)

    new_data = []
    for i in range(0, sample_size, bucket_size):
        data_slice = data_normed[i:i+bucket_size]
        mean = np.expand_dims(data_slice.mean(axis=0), axis=0)
        std = np.expand_dims(data_slice.std(axis=0), axis=0)
        std_1_away_right = mean + std
        std_1_away_left = mean - std

        percentile_calc = np.percentile(data_slice, [0, 1, 25, 50, 75, 99, 100], axis=0)
        percentile_range = np.expand_dims(percentile_calc[-1] - percentile_calc[0], axis=0)
        relative_percentile = percentile_calc - mean

        new_data.append(np.concatenate([
            mean, std, std_1_away_right,
            std_1_away_left, percentile_range,
            percentile_calc, relative_percentile
        ]))
    return np.expand_dims(np.asarray(new_data), axis=0)

def load_train_meta(path):
    meta_df = pd.read_csv(path)
    ids = [i for i in range(meta_df.shape[0] // 3)]
    meta_df.set_index(['id_measurement', 'phase'], inplace=True)
    targets = np.array([np.array([meta_df.loc[id].loc[0]['target'], 
                         meta_df.loc[id].loc[1]['target'], 
                         meta_df.loc[id].loc[2]['target']])
                        for id in ids])
    return ids, targets

def load_and_preprocess(meta_path,
                        src_path,
                        is_train=True,
                        do_strat_split=True,
                        val_size=0.1,
                        random_state=42):
    if do_strat_split:
        assert is_train, 'do_strat_split can be True only if is_train is True.'

    if is_train:
        ids, targets = load_train_meta(meta_path)
        start, end = 0, 8712
    else:
        meta_df = pd.read_csv(meta_path)
        signal_ids = meta_df['signal_id'].values
        start, end = 8712, meta_df.shape[0] + 8712

    X = []
    for i in tqdm.tqdm(range(start, end, 3)):
        pq_data = read_table(src_path, i, i+3)
        X.append(transform_and_binify(pq_data, 800000))
    X = np.concatenate(X)

    if do_strat_split:
        ((X_train, Y_train),
         (X_val, Y_val)) = _strata_split_helper(X, targets, val_size, random_state)
        return (X_train, Y_train), (X_val, Y_val)

    if is_train:
        return X, targets
    return X, signal_ids

def _strata_split_helper(X, Y, val_size, random_state):
    tmp_targets = []
    # each target in targets is a 3 elelment array, one for each phase
    for target in Y:
        # if any 1 target is non-zero
        if np.sum(target) != 0:
            # if the targets isn't all 1 then give it class 2 else give class 1
            if np.sum(target) != 3:
                tmp_targets.append(2)
            else:
                tmp_targets.append(1)
        else:
            tmp_targets.append(0)
    train_indices, val_indices = stratified_split(X, tmp_targets,
                                                  val_size, random_state)
    return (X[train_indices], Y[train_indices]), (X[val_indices], Y[val_indices])

def stratified_split(X, Y, val_size, random_state):
    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=val_size,
                                 random_state=random_state)
    return next(sss.split(X, Y))

def np_save(path, **arrays):
    np.savez(path, **arrays)


# Preprocessing data using functions defined above

# In[ ]:


((X_train, Y_train), (X_val, Y_val)) = load_and_preprocess(Config.META_TRAIN, Config.TRAIN_SRC)
np_save(Config.TRAIN_ARRAYS_STRATIFIED,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val)

del X_train
del Y_train
del Y_val
del X_val
gc.collect()

X, signal_id = load_and_preprocess(Config.META_TEST, Config.TEST_SRC, False, False)
np_save(Config.TEST_ARRAYS, X=X, signal_id=signal_id)
del X
del signal_id
gc.collect()


# Now, we are at the fun stage. There are two parts to a TensorFlow estimator:
# 1. Estimator model
# 2. Estimator Input function (read input pipeline)
# 
# We will first fo through the pipeline.
# 
# The pipeline is the function `input_fn`. It will take in our desired settings and give us an iterator that can be used by the estimator model to run through the data.
# 
# In case of training:
# * Input pipeline will need both X and Y along with batch size.
# * We will allow shuffling for better training
# * We will allow the iterator to repeat data. This will make sure that the iterator iterates over data for  infinite number of times (till we train the model).
# * We will have buffer_size, which will make sure that only required number of data is pushed into memory.
# 
# In case of evaluation, everything will be same except that we do not repeat the data. This means the iterator will iterate over data only once.
# 
# In case of prediction everything will be same except that we neither repeat nor shuffle the data.
# 
# The iterators returned by the function are in form of dictionaries. This makes sure that we can access them using sensible naming conventions.

# In[ ]:


def shuffle_repeat_applier(dataset, buffer_size, shuffle, repeat):
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    else:
        dataset = dataset.prefetch(buffer_size)

    if repeat:
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(1)

    return dataset

def input_fn(X, Y=None, batch_size=32, buffer_size=2000,
             shuffle=True, repeat=True):
    if Y is not None:
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, ))

    dataset = shuffle_repeat_applier(dataset, buffer_size, shuffle, repeat)
    dataset = dataset.batch(batch_size)

    if Y is not None:
        X, Y = dataset.make_one_shot_iterator().get_next()
    else:
        X = dataset.make_one_shot_iterator().get_next()[0]

    features_dic = {'signal': tf.cast(X, tf.float32)}
    if Y is not None:
        labels_dic = {
            'phase1': tf.one_hot(Y[:, 0], depth=2),
            'phase2': tf.one_hot(Y[:, 1], depth=2),
            'phase3': tf.one_hot(Y[:, 2], depth=2)
        }
        return features_dic, labels_dic
    return features_dic


# Let's define our model.
# 
# We will be defining a 9 layers model (excluding Dropout and Global Pool):
# * The first 5 layers will be simple convolution, batchnormalization and relu stack. So we will define a function to encapsulate this stack. The Sequential model of keras will come handy here.
# * Then we will apply a Global Pool. This will result in a output of (Batch Size, 512). 
# * The above output will pass through a dense layer which then passes outputs into a dropout layer, with drop rate = 0.4
# * Finally we will have three dense layers, one for each phase, with 2 outputs (0 and 1).
# 
# The estimator model is defined in a function, `model_fn` in this case. This function always have 4 args: `features`, `labels`, `mode`, `params`:
# * `feartures`: The input tensor to the model is passed using this arg. It is usually a dictionary that has its values as the feature or features that the model needs to learn on.
# * `labels`: As the name suggests, this will be the argument that holds our labels.
# * `mode`; The estimator API has three mode keys `tf.estimator.ModeKeys.TRAIN`, `tf.estimator.ModeKeys.EVAL` and `tf.estimator.ModeKeys.PREDICT`. Depending on which key is passed here we need to define the output spec (`tf.estimator.EstimatorSpec`) of our model.
# * `params`: Our model will be in need of special case sensitive values (hyperparams), like learning rate, drop rate and so on. These will passed as a dictionary in ths arg.
# 
# I'll also make use of tensorflow summaries to plot values of special metrics in the tensorboard.

# In[ ]:


from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2

def conv_bn_relu(in_tensor,
                 filters,
                 kernel_size,
                 strides,
                 padding='valid',
                 weight_decay=5e-4):
    return models.Sequential([
        layers.Conv2D(filters, kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation('relu')
    ])(in_tensor)

def _make_summaries_helper(labels,
                           phase1_logits,
                           phase2_logits,
                           phase3_logits,
                           metric_func,
                           summary_name):
    phase1 = metric_func(tf.argmax(labels['phase1'], axis=1),
                         tf.argmax(phase1_logits, axis=1))
    phase2 = metric_func(tf.argmax(labels['phase2'], axis=1),
                         tf.argmax(phase2_logits, axis=1))
    phase3 = metric_func(tf.argmax(labels['phase3'], axis=1),
                         tf.argmax(phase3_logits, axis=1))

    tf.summary.scalar(summary_name + '_phase1', phase1)
    tf.summary.scalar(summary_name + '_phase2', phase2)
    tf.summary.scalar(summary_name + '_phase3', phase3)
    return phase1, phase2, phase3

def make_summaries(labels,
                   phase1_logits,
                   phase2_logits,
                   phase3_logits):
    _make_summaries_helper(labels, phase1_logits,
                           phase2_logits, phase3_logits,
                           true_positives, 'tp')
    _make_summaries_helper(labels, phase1_logits,
                           phase2_logits, phase3_logits,
                           true_negatives, 'tn')
    _make_summaries_helper(labels, phase1_logits,
                           phase2_logits, phase3_logits,
                           false_positives, 'fp')
    _make_summaries_helper(labels, phase1_logits,
                           phase2_logits, phase3_logits,
                           false_negatives, 'fn')
    mcc1, mcc2, mcc3 = _make_summaries_helper(labels, phase1_logits,
                                              phase2_logits, phase3_logits,
                                              mcc, 'mcc')
    return mcc1, mcc2, mcc3

def model_fn(features, labels, mode, params):
    conv_bn_relu1 = conv_bn_relu(features['signal'], 32, (7, 3), (2, 1))
    conv_bn_relu2 = conv_bn_relu(conv_bn_relu1, 64, (7, 3), (2, 1))
    conv_bn_relu3 = conv_bn_relu(conv_bn_relu2, 128, (7, 3), (2, 1))
    conv_bn_relu4 = conv_bn_relu(conv_bn_relu3, 256, (3, 3), (2, 2))
    conv_bn_relu5 = conv_bn_relu(conv_bn_relu4, 512, (3, 3), (2, 2))

    pool = tf.keras.layers.GlobalAveragePooling2D()(conv_bn_relu5)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(pool)
    dropout = tf.keras.layers.Dropout(rate=params['drop_rate'])(dense1)

    phase1_logits = tf.keras.layers.Dense(2)(dropout)
    phase2_logits = tf.keras.layers.Dense(2)(dropout)
    phase3_logits = tf.keras.layers.Dense(2)(dropout)

    if mode == tf.estimator.ModeKeys.PREDICT:
        preds = {
            'phase1': tf.argmax(phase1_logits, axis=1),
            'phase2': tf.argmax(phase2_logits, axis=1),
            'phase3': tf.argmax(phase3_logits, axis=1)
        }
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=preds)
    else:
        
        mcc1, mcc2, mcc3 = make_summaries(labels, phase1_logits, phase2_logits, phase3_logits)
        logging_hook = tf.train.LoggingTensorHook({
            "mcc_phase1": mcc1,
            "mcc_phase2": mcc2,
            "mcc_phase3": mcc3
        }, every_n_iter=15)

        costs_pahse1 = tf.losses.softmax_cross_entropy(labels['phase1'], phase1_logits)
        costs_pahse2 = tf.losses.softmax_cross_entropy(labels['phase2'], phase2_logits)
        costs_pahse3 = tf.losses.softmax_cross_entropy(labels['phase3'], phase3_logits)

        costs = costs_pahse1 + costs_pahse2 + costs_pahse3
        loss = tf.reduce_mean(costs)

        global_step = tf.train.get_global_step()
        optimizer = tf.train.AdamOptimizer(params['lr'])
        train_op = optimizer.minimize(loss,
                                      global_step=global_step)

        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op, training_hooks=[logging_hook])
    return spec


# Now let's have some helper functions for training, evaluation, mcc evaluation and prediction.
# 
# The estimator API returns predictions in form a generator. As we returned a dictionary of predictions for each phase, we will get a generator each value of which will be a dictionary.

# In[ ]:


def get_preds(pred_generator):
    preds = []
    for pred in pred_generator:
        preds.append(pred['phase1'])
        preds.append(pred['phase2'])
        preds.append(pred['phase3'])
    return preds

def train(model, X, Y, steps):
    train_input_fn = lambda: input_fn(X, Y)
    model.train(input_fn=train_input_fn, steps=steps)

def eval(model, X, Y):
    eval_input_fn = lambda: input_fn(X, Y, shuffle=False, repeat=False)
    return model.evaluate(input_fn=eval_input_fn)

def predict(model, X):
    predict_input_fn = lambda: input_fn(X, shuffle=False, repeat=False)
    return get_preds(model.predict(predict_input_fn))

def eval_mcc(model, X, Y):
    preds = predict(model, X)
    return mcc(Y, preds, True)


# Let's define some constants that will help us at time of training.
# 
# I've used `tf.estimator.RunConfig` to define extra configurations for the model. Estimator API saves checkpoints. The default configurations were causing out of memory issues on kaggle kernel Disk. \\_(^_^)_/

# In[ ]:


BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-5
DROP_RATE = 0.4

RUN_CONFIG = tf.estimator.RunConfig(tf_random_seed=42,
                                    save_summary_steps=200,
                                    keep_checkpoint_max=3)


# In[ ]:


model = tf.estimator.Estimator(model_fn=model_fn,
                               params={
                                   'lr': LR,
                                   'drop_rate': DROP_RATE
                               }, model_dir='./model/',
                               config=RUN_CONFIG)


# In[ ]:


npzfile = np.load(Config.TRAIN_ARRAYS_STRATIFIED)
npzfile.files

X_train = npzfile['X_train']
Y_train = npzfile['Y_train']
X_val = npzfile['X_val']
Y_val = npzfile['Y_val']


# The training of an estimator is done on number of steps, not on epochs, so we will calculate number of steps here

# In[ ]:


STEPS = EPOCHS * (X_train.shape[0] // BATCH_SIZE)
STEPS


# Finally, let's train this model!

# In[ ]:


train(model, X_train, Y_train, STEPS)


# Our model needs to have a drop rate of 0 to predict. Hence defining the model again. Don't worry, it will load the settings from the checkpoints

# In[ ]:


del model
model = tf.estimator.Estimator(model_fn=model_fn,
                               params={
                                   'lr': LR,
                                   'drop_rate': 0
                               }, model_dir='./model/',
                               config=RUN_CONFIG)


# In[ ]:


def flatten(Y):
    new_y = []
    for y in Y:
        new_y.extend(list(y))
    return np.array(new_y)


# In[ ]:


print('Loss on Train Data:{}'.format(eval(model, X_train, Y_train)))
print('MCC on Train Data:{}'.format(eval_mcc(model, X_train, flatten(Y_train))))


# In[ ]:


print('Loss on Val Data:{}'.format(eval(model, X_val, Y_val)))
print('MCC on Val Data:{}'.format(eval_mcc(model, X_val, flatten(Y_val))))


# Let's train model on the left out data, for final submission

# In[ ]:


del model
model = tf.estimator.Estimator(model_fn=model_fn,
                               params={
                                   'lr': LR,
                                   'drop_rate': 0.4
                               }, model_dir='./model/',
                               config=RUN_CONFIG)


# In[ ]:


EPOCHS = 1
STEPS = X_val.shape[0] // BATCH_SIZE

train(model, X_val, Y_val, STEPS)


# Time for final prediction and submission!

# In[ ]:


del X_train
del Y_train
del X_val
del Y_val
gc.collect()

del model
model = tf.estimator.Estimator(model_fn=model_fn,
                               params={
                                   'lr': LR,
                                   'drop_rate': 0
                               }, model_dir='./model/',
                               config=RUN_CONFIG)


# In[ ]:


npzfile3 = np.load(Config.TEST_ARRAYS)
X = npzfile3['X']
signal_ids = npzfile3['signal_id']


preds = predict(model, X)


# In[ ]:


df = pd.DataFrame({
    'signal_id': list(signal_ids),
    'target': list(preds)
})

df.to_csv('submission.csv', index=False)

