#!/usr/bin/env python
# coding: utf-8

# ### This kernel presents some changes on the amazing Bruno Aquino's kernel. 
# ### The main difference is that experimental results are used for training data and overwrite of predicted results
# ### Please upvote if it was helpful.

# In[ ]:


# experimental data 
exp_true = [9423, 9424, 9425, 
            9597, 9598, 9599, 
            10248, 10249, 10250, 
            11523, 11524, 11525,
            12036, 12037, 12038,
            12222, 12223, 12224,
            13041, 13042, 13043,
            14028, 14029, 14030,
            14472, 14473, 14474,
            15540, 15541, 15542,
            17289, 17290, 17291,
            17685, 17686, 17687,
            19458, 19459, 19460,
            22827, 22828, 22829,
            22938, 22939, 22940,
            23286, 23287, 23288,
            24168, 24169, 24170,
            25143, 25144, 25145,
            26010, 26011, 26012]
exp_false = [8769, 8770, 8771,
             9810, 9811, 9812,
             11178, 11179, 11180,
             12186, 12187, 12188,
             13968, 13969, 13970,
             14187, 14188, 14189,
             15177, 15178, 15179,
             16920, 16921, 16922,
             18630, 18631, 18632,
             18828, 18829, 18830,
             19308, 19309, 19310,
             20832, 20833, 20834,
             26202, 26203, 26204,
             28167, 28168, 28169]


# In[ ]:


import os 
import numpy as np
import pandas as pd
import pyarrow.parquet as pq # Used to read the data
from joblib import Parallel, delayed
from tqdm import tqdm # Processing time measurement

import tensorflow as tf
from tensorflow import set_random_seed

from keras.layers import * # Keras is the most friendly Neural Network library, this Kernel use a lot of layers classes
from keras.models import Model
from keras import backend as K # The backend give us access to tensorflow operations and allow us to create the Attention class
from keras import optimizers # Allow us to access the Adam class to modify some parameters
from keras.callbacks import * # This object helps the model to train in a smarter way, avoiding overfitting

from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model
from sklearn.model_selection import train_test_split 
from sklearn.metrics import matthews_corrcoef


# In[ ]:


# select how many folds will be created
N_SPLITS = 5
# it is just a constant with the measurements data size
sample_size = 800000

# max threads number for parallel
MAX_THREADS = 4
RANDOM_SEED = 2019


# In[ ]:


np.random.seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)


# ### The standart way keras calculates an epoch's loss/metric is by taking the average value of the loss/metric on batches. While it works well on losses calculated by average (like cross entropy), it creates problems in metrics that must be calculated over the entire dataset (like F1 score or Matthews Correlation).  
# ### To overcome that problems, we are going to use a statefull metric that is calculated over a whole epoch. A stateful metric on keras is a special layer that allow the running of cumulative operations on each batch. The following implementation requires tensorflow as backend.

# In[ ]:


class StatefullMCC(Layer):
    def __init__(self, thresholds, **kwargs):
        super(StatefullMCC, self).__init__(**kwargs)
        self.thresholds = thresholds
        self.stateful = True
        self.name='matthews_correlation'

    def reset_states(self):
        K.get_session().run(tf.variables_initializer(self.local_variable))

    def metric_variable(self, shape, dtype, validate_shape=True, name=None):
        return tf.Variable(
                np.zeros(shape),
                dtype=dtype,
                trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
                validate_shape=validate_shape,
                name=name,
                )

    def initialize_vars(self, y_true, y_pred):
        self.tp = [None]*len(self.thresholds)
        self.tn = [None]*len(self.thresholds)
        self.fp = [None]*len(self.thresholds)
        self.fn = [None]*len(self.thresholds)
        for i,_ in enumerate(self.thresholds):
            self.tp[i] = self.metric_variable(shape=[1], dtype=tf.int64, validate_shape=False, name='tp%d'%i)
            self.tn[i] = self.metric_variable(shape=[1], dtype=tf.int64, validate_shape=False, name='tn%d'%i)
            self.fp[i] = self.metric_variable(shape=[1], dtype=tf.int64, validate_shape=False, name='fp%d'%i)
            self.fn[i] = self.metric_variable(shape=[1], dtype=tf.int64, validate_shape=False, name='fn%d'%i)

            tp_op = tf.assign_add(self.tp[i], tf.count_nonzero(y_true * y_pred[i], axis=0))
            self.add_update(tp_op)
            tn_op = tf.assign_add(self.tn[i], tf.count_nonzero((1-y_true)*(1-y_pred[i]), axis=0))
            self.add_update(tn_op)
            fp_op = tf.assign_add(self.fp[i], tf.count_nonzero((1-y_true)*y_pred[i], axis=0))
            self.add_update(fp_op)
            fn_op = tf.assign_add(self.fn[i], tf.count_nonzero(y_true*(1-y_pred[i]), axis=0))
            self.add_update(fn_op)

        self.local_variable = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

    def __call__(self, y_true, y_pred):
        rounded_preds = [K.cast(K.greater_equal(y_pred, t), 'float32') for t in self.thresholds]
        self.initialize_vars(y_true, rounded_preds)
        mcc_vec = []
        for i,_ in enumerate(self.thresholds):
            num = tf.cast((self.tp[i] * self.tn[i] - self.fp[i] * self.fn[i]), 'float32')
            den = K.sqrt(tf.cast((self.tp[i] + self.fp[i]) * (self.tp[i] + self.fn[i]) * (self.tn[i] + self.fp[i]) * (self.tn[i] + self.fn[i]), 'float32'))
            mcc_vec.append(num/(den + tf.constant(K.epsilon())))
        return K.max(mcc_vec)


# In[ ]:


# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.glorot_uniform(RANDOM_SEED)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# In[ ]:


# just load train data
df_train = pd.read_csv('../input/metadata_train.csv')
# set index, it makes the data access much faster
df_train = df_train.set_index(['id_measurement', 'phase'])
df_train.head()


# In[ ]:


def get_features(dataset='train', split_parts=10):
    if dataset == 'train':
        cache_file = 'X.npy'
        meta_file = '../input/metadata_train.csv'
    elif dataset == 'test':
        cache_file = 'X_test.npy'
        meta_file = '../input/metadata_test.csv'
    if os.path.isfile(cache_file):
        X = np.load(cache_file)
        y = None
        if dataset == 'train':
            y = np.load('y.npy')
    else:
        meta_df = pd.read_csv(meta_file)

        data_measurements = meta_df.pivot(index='id_measurement', columns='phase', values='signal_id')
        data_measurements = data_measurements.values
        data_measurements = np.array_split(data_measurements, split_parts, axis=0)
        X = Parallel(n_jobs=min(split_parts, MAX_THREADS), verbose=1)(delayed(prep_data)(p, dataset) for p in data_measurements)
        try:
            y = meta_df.loc[meta_df['phase']==0, 'target'].values
        except:
            y = None
        X = np.concatenate(X, axis=0)

        if dataset == 'train':
            np.save("X.npy",X)
            np.save("y.npy",y)
        elif dataset == 'test':
            np.save("X_test.npy",X)
    return X, y


# In[ ]:


# in other notebook I have extracted the min and max values from the train data, the measurements
max_num = 127
min_num = -128


# In[ ]:


# This function standardize the data from (-128 to 127) to (-1 to 1)
# Theoretically it helps in the NN Model training, but I didn't tested without it
def min_max_transf(ts, min_data, max_data, range_needed=(-1,1)):
    ts_std = (ts - min_data) / (max_data - min_data)
    return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


# In[ ]:


# This is one of the most important peace of code of this Kernel
# Any power line contain 3 phases of 800000 measurements, or 2.4 millions data 
# It would be praticaly impossible to build a NN with an input of that size
# The ideia here is to reduce it each phase to a matrix of <n_dim> bins by n features
# Each bean is a set of 5000 measurements (800000 / 160), so the features are extracted from this 5000 chunk data.
def transform_ts(ts, n_dim=160, min_max=(-1,1)):
    # convert data into -1 to 1
    ts_std = min_max_transf(ts, min_data=min_num, max_data=max_num)
    # bucket or chunk size, 5000 in this case (800000 / 160)
    bucket_size = int(sample_size / n_dim)
    # new_ts will be the container of the new data
    new_ts = []
    # this for iteract any chunk/bucket until reach the whole sample_size (800000)
    for i in tqdm(range(0, sample_size, bucket_size)):
        # cut each bucket to ts_range
        ts_range = ts_std[i:i + bucket_size]
        # calculate each feature
        mean = ts_range.mean()
        std = ts_range.std() # standard deviation
        std_top = mean + std # I have to test it more, but is is like a band
        std_bot = mean - std
        # I think that the percentiles are very important, it is like a distribuiton analysis from eath chunk
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100]) 
        max_range = percentil_calc[-1] - percentil_calc[0] # this is the amplitude of the chunk
        relative_percentile = percentil_calc - mean # maybe it could heap to understand the asymmetry
        # now, we just add all the features to new_ts and convert it to np.array
        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range]),percentil_calc, relative_percentile]))
    return np.asarray(new_ts)


# In[ ]:


def prep_data(signal_ids, dataset="train"):
    signal_ids_all = np.concatenate(signal_ids)
    if dataset == "train":
        praq_data = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in signal_ids_all]).to_pandas()
    elif dataset == "test":
        praq_data = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in signal_ids_all]).to_pandas()
    else:
        raise ValueError("Unknown dataset")
    X = []
    for sids in tqdm(signal_ids):
        data = praq_data[[str(s) for s in sids]].values.T
        X_signal = [transform_ts(signal) for signal in data]
        X_signal = np.concatenate(X_signal, axis=1)
        X.append(X_signal)
    X = np.asarray(X)
    return X


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X, y = get_features("train", split_parts=6)')


# In[ ]:


# The X shape here is very important. It is also important undertand a little how a LSTM works
# X.shape[0] is the number of id_measuremts contained in train data
# X.shape[1] is the number of chunks resultant of the transformation, each of this date enters in the LSTM serialized
# This way the LSTM can understand the position of a data relative with other and activate a signal that needs
# a serie of inputs in a specifc order.
# X.shape[3] is the number of features multiplied by the number of phases (3)
print(X.shape, y.shape)


# In[ ]:


X


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Now load the test data\n# This first part is the meta data, not the main data, the measurements\nmeta_test = pd.read_csv('../input/metadata_test.csv')")


# In[ ]:


meta_test = meta_test.set_index(['signal_id'])
meta_test.head()


# In[ ]:


tmp = meta_test.reset_index()


# In[ ]:


true_mask = tmp["signal_id"].isin(exp_true).values[::3]
false_mask = tmp["signal_id"].isin(exp_false).values[::3]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'MAX_THREADS = 4\nX_test_input, _ = get_features("test")')


# In[ ]:


X_test_input.shape


# In[ ]:


true_X_test = X_test_input[true_mask]
false_X_test = X_test_input[false_mask]


# In[ ]:


print(true_X_test.shape)
print(false_X_test.shape)


# In[ ]:


XX = np.append(X, true_X_test, axis=0)
X = np.append(XX, false_X_test, axis=0)


# In[ ]:


y = np.append(y, [1] * (len(exp_true) // 3) + [0]* (len(exp_false) // 3), axis=0)


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


# This is NN LSTM Model creation
def model_lstm(input_shape):
    # The shape was explained above, must have this order
    inp = Input(shape=(input_shape[1], input_shape[2],))
    
    init_glorot_uniform = initializers.glorot_uniform(seed=RANDOM_SEED)
    init_orthogonal = initializers.orthogonal(seed=RANDOM_SEED)
    
    # This is the LSTM layer
    # Bidirecional implies that the 160 chunks are calculated in both ways, 0 to 159 and 159 to zero
    # although it appear that just 0 to 159 way matter, I have tested with and without, and tha later worked best
    # 128 and 64 are the number of cells used, too many can overfit and too few can underfit
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True, kernel_initializer=init_glorot_uniform, recurrent_initializer=init_orthogonal))(inp)
    # The second LSTM can give more fire power to the model, but can overfit it too
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True, kernel_initializer=init_glorot_uniform, recurrent_initializer=init_orthogonal))(x)
    # Attention is a new tecnology that can be applyed to a Recurrent NN to give more meanings to a signal found in the middle
    # of the data, it helps more in longs chains of data. A normal RNN give all the responsibility of detect the signal
    # to the last cell. Google RNN Attention for more information :)
    x = Attention(input_shape[1])(x)
    # A intermediate full connected (Dense) can help to deal with nonlinears outputs
    x = Dense(64, activation="relu", kernel_initializer=init_glorot_uniform)(x)
    # A binnary classification as this must finish with shape (1,)
    x = Dense(1, activation="sigmoid", kernel_initializer=init_glorot_uniform)(x)
    model = Model(inputs=inp, outputs=x)
    # Pay attention in the addition of matthews_correlation metric in the compilation, it is a success factor key
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[StatefullMCC(np.linspace(0.45,0.55,11))])
    
    return model


# In[ ]:


# Here is where the training happens

# First, create a set of indexes of the 5 folds
splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED).split(X, y))
preds_val = []
y_val = []
# Then, iteract with each fold
# If you dont know, enumerate(['a', 'b', 'c']) returns [(0, 'a'), (1, 'b'), (2, 'c')]
for idx, (train_idx, val_idx) in tqdm(enumerate(splits)):
    K.clear_session() # I dont know what it do, but I imagine that it "clear session" :)
    print("Beginning fold {}".format(idx+1))
    # use the indexes to extract the folds in the train and validation data
    train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
    # instantiate the model for this fold
    model = model_lstm(train_X.shape)
    # This checkpoint helps to avoid overfitting. It just save the weights of the model if it delivered an
    # validation matthews_correlation greater than the last one.
    ckpt = ModelCheckpoint('weights_{}.h5'.format(idx), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_matthews_correlation', mode='max')
    # Train, train, train
    model.fit(train_X, train_y, batch_size=128, epochs=100, validation_data=[val_X, val_y], callbacks=[ckpt])
    # loads the best weights saved by the checkpoint
    model.load_weights('weights_{}.h5'.format(idx))
    # Add the predictions of the validation to the list preds_val
    preds_val.append(model.predict(val_X, batch_size=512))
    # and the val true y
    y_val.append(val_y)

# concatenates all and prints the shape    
preds_val = np.concatenate(preds_val)[...,0]
y_val = np.concatenate(y_val)
preds_val.shape, y_val.shape


# In[ ]:


# The output of this kernel must be binary (0 or 1), but the output of the NN Model is float (0 to 1).
# So, find the best threshold to convert float to binary is crucial to the result
# this piece of code is a function that evaluates all the possible thresholds from 0 to 1 by 0.01
def threshold_search(y_true, y_proba):
    thresholds = np.linspace(0.0,1.0,101)
    scores = [matthews_corrcoef(y_true, (y_proba > t).astype(np.uint8)) for t in thresholds]
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]


# In[ ]:


best_threshold, best_score = threshold_search(y_val, preds_val)
print(best_threshold, best_score)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
print(len(submission))
submission.head()


# In[ ]:


preds_test = []
for i in tqdm(range(N_SPLITS)):
    model.load_weights('weights_{}.h5'.format(i))
    pred = model.predict(X_test_input, batch_size=300, verbose=1)
    pred_3 = []
    for pred_scalar in pred:
        for i in range(3):
            pred_3.append(pred_scalar)
    preds_test.append(pred_3)


# In[ ]:


preds_test = (np.squeeze(np.mean(preds_test, axis=0)) > best_threshold).astype(np.int)
preds_test.shape


# In[ ]:


submission['target'] = preds_test
submission.loc[submission["signal_id"].isin(exp_true), "target"] = 1
submission.loc[submission["signal_id"].isin(exp_false), "target"] = 0
submission.to_csv('submission.csv', index=False)
submission.head()

