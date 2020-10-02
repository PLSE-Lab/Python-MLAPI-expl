#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

get_ipython().system('pip install -q nilearn')
get_ipython().system('pip install -q dltk')
get_ipython().system('pip install -q pycaret')
#!pip install -q tensorflow-io
#import tensorflow_io as tfio
from pycaret.regression import *
import nilearn
from nilearn import plotting, masking, input_data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy as sp
import scipy.stats
import pickle
import numpy.ma as ma
from tensorflow.python.keras import backend as K
import copy
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import seaborn as sns
import dltk
import SimpleITK as sitk
from dltk.io.preprocessing import *
import random
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dirname = "/kaggle/input/trends-assessment-prediction"
read_test_data = False
read_train_data = False
download_train_data = True
download_test_data = False
select_column_pval_threshold = 0.1 #-1
plot = True


# In[ ]:


mask = nilearn.image.load_img(dirname + "/fMRI_mask.nii")
mask.shape
nilearn.plotting.plot_img(mask)


# In[ ]:


m = h5py.File(dirname + "/fMRI_train/10001.mat")['SM_feature'][()]
print("hdf5 mat shape:", m.shape)
m = np.swapaxes(m, 1, 3)
m = np.moveaxis(m, 0, 3)
print("after axis moves:", m.shape)
im_4d = nilearn.image.new_img_like(mask, m)
im_first = nilearn.image.new_img_like(mask, m[:,:,:,0])
plotting.plot_img(im_first)
im_filtered = nilearn.masking.apply_mask(im_4d, mask)
print("after apply_mask:", type(im_filtered), im_filtered.shape)
im_mean = nilearn.image.mean_img(im_4d)
plotting.plot_img(im_mean)


# In[ ]:


print("after apply_mask", im_filtered.shape)
plt.plot(im_filtered[:53, :2])


# In[ ]:


fnc = pd.read_csv(dirname + "/fnc.csv")
icn = pd.read_csv(dirname + "/ICN_numbers.csv")
loading = pd.read_csv(dirname + "/loading.csv")
tscores = pd.read_csv(dirname + "/train_scores.csv")
data = fnc.merge(loading, on = "Id")
data_columns = list(data.columns)
data_columns.remove('Id')
output_columns = list(tscores.columns)
output_columns.remove('Id')
data = data.merge(tscores, on = "Id", how = "outer")
del fnc, icn, tscores
#data['filename'] = np.where(data['age'].isnull(), dirname + '/fMRI_test/', dirname + '/fMRI_train/')
#data['filename'] = data['filename'] + data['Id'].astype(str) + ".mat"


# In[ ]:


r2 = pd.read_csv(dirname + "/reveal_ID_site2.csv")
r2['site'] = "site2"
a = data.merge(r2, on = "Id", how = "outer")
a['site'] = a['site'].fillna("site1")
a.head(15)
s1 = a[a['site'] == "site1"]
s2 = a[a['site'] == "site2"]
print("site1:",s1.shape, "site2:", s2.shape)
all_col_types = set([a[i].dtype for i in data_columns])
print("feature column types:", all_col_types)
numeric_cols = [i for i in data_columns if a[i].dtype == 'int64' or a[i].dtype == 'float64']
print("number of numeric feature cols:", len(numeric_cols))


# In[ ]:


def plot_site1_v_site2(s1, s2, numeric_cols, start_index = 0, end_index = None):
    end_index = len(numeric_cols) if end_index is None else min(end_index, len(numeric_cols))
    ncol = 4.0
    rows = np.ceil(len(numeric_cols[start_index:end_index]) / ncol) 
    f, axes = plt.subplots(int(rows), int(ncol), figsize = (ncol * 5, rows * 5))
    data = pd.concat([s1, s2], axis = 0)
    for count, col in enumerate(numeric_cols[start_index:end_index]):
        i = count // int(ncol)
        j = count % int(ncol)
        if rows > 1:
            sns.boxplot(data = data, x = 'site', y = col, ax = axes[i][j])
        else:
            sns.boxplot(data = data, x = 'site', y = col, ax = axes[j])
if plot:
    plot_site1_v_site2(s1, s2, numeric_cols, start_index = 2, end_index= 8)


# In[ ]:


def compute_test_stats(s1, s2, numeric_cols):
    test_results = []
    for col in numeric_cols:
        t = ttest_ind(s1[col], s2[col], equal_var = False)
        mw = mannwhitneyu(s1[col], s2[col], alternative = 'two-sided')
        test_results.append([col, t[1], mw[1]])
    test_results = pd.DataFrame(test_results, columns = ['name', 'ttest_pval', 'mannwhitney_pval'])
    return test_results
test_results = compute_test_stats(s1, s2, numeric_cols)
fig, axes = plt.subplots(1, 2)
_ = axes[0].hist(test_results['ttest_pval'])
_ = axes[1].hist(test_results['mannwhitney_pval'])
_ = axes[0].set_title("t-test Pvalue distribution")
_ = axes[1].set_title("Mann-Whitney Pvalue distribution")


# In[ ]:


print("number of data colums before filtering:", len(data_columns))
if select_column_pval_threshold > 0:
    if read_train_data or download_train_data:
        select_columns = list(test_results[test_results['ttest_pval'] > select_column_pval_threshold]['name'])
    else:
        print("using pre-selected columns")
    data_columns = list(set(data_columns).intersection(select_columns))
print("number of data colums after applying", select_column_pval_threshold, "threshold on t-test pvalue:", len(data_columns))
removed_columns = test_results[test_results['ttest_pval'] <= select_column_pval_threshold].sort_values('ttest_pval', ascending = True)['name']
plot_site1_v_site2(s1, s2, removed_columns, start_index = 2, end_index= 6)
del a, s1, s2, all_col_types, numeric_cols, removed_columns


# In[ ]:


data = data[['Id'] + data_columns + output_columns]
train_data = data[~data['age'].isnull()]
test_data = data[data['age'].isnull()]
test_data.drop(output_columns, axis = 1, inplace = True)
print("columns with missing values and their counts:")
columns_w_nan = list(train_data.columns[np.where(train_data.isnull().sum()>0)[0]])
print(train_data[columns_w_nan].isnull().sum())
train_data.dropna(axis = 0, inplace = True)
#train_data = train_data.fillna(train_data.median())
#print("NAs filled with median")


# In[ ]:


train_data.reset_index(drop = True, inplace = True)
trainx = train_data[data_columns]
trainy = train_data[output_columns]
test_data.reset_index(drop = True, inplace = True)
test_ids = test_data['Id']
testx = test_data[data_columns]


# In[ ]:


output_order = [1,2,0,4,3]
for i in output_order:
    outname = output_columns[i]
    d = setup(data = pd.concat([trainx, trainy[outname]], axis = 1), target = outname, silent = True)
    br = tune_model('br', optimize = 'mse')
    br = finalize_model(br)
    out_train = predict_model(br, data = trainx).rename({'Label': outname}, axis = 1)
    out_test = predict_model(br, data = testx).rename({'Label': outname}, axis = 1)
    trainx = pd.concat([trainx, out_train[outname]], axis = 1)
    testx = pd.concat([testx, out_test[outname]], axis = 1)


# In[ ]:


output = pd.concat([testx, test_ids], axis = 1)
output = output[['Id'] + output_columns]
output.head()


# In[ ]:


output = pd.melt(output, id_vars = "Id", value_vars = output_columns, var_name = "target", value_name = "Predicted")
output['Id'] = output['Id'].astype(str) + "_" + output['target']
output.drop('target', axis = 1, inplace = True)
output.to_csv("submission.csv", index = False)
output.head()


# In[ ]:


train_data, valid_data = train_test_split(train_data, test_size = 0.075, random_state = 0)


# In[ ]:


print(train_data.shape, valid_data.shape)


# In[ ]:


train_data = train_data.to_numpy()
valid_data = valid_data.to_numpy()
test_data = test_data.to_numpy()
train_data[:,0] = train_data[:, 0].astype(int)
valid_data[:,0] = valid_data[:, 0].astype(int)
test_data[:,0] = test_data[:, 0].astype(int)


# In[ ]:


print(train_data.shape, valid_data.shape, test_data.shape)


# In[ ]:


use_fnc_loading = False
use_mri_images = True

NORMALIZE_SM = True
TRANSFORM = True
MASK_IMG = nilearn.image.load_img(dirname + "/fMRI_mask.nii")
AVERAGE_SPACE = True
AVERAGE_ON = 8
APPLY_MASK = False
BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

class hdf5_reader:
    def read_hdf5(self, number, set_type):
        #print(str(number))
        if set_type == b"test":
            m = h5py.File(dirname + "/fMRI_test/" + str(number) + ".mat", 'r')['SM_feature'][()]    
        else:
            m = h5py.File(dirname + "/fMRI_train/" + str(number) + ".mat", 'r')['SM_feature'][()]
        m = np.swapaxes(m, 1, 3)
        m = np.moveaxis(m, 0, 3)
        return m #.reshape(-1)
    
    def collect_non_sm_features(self, number, set_type):
        if set_type == b"test":
            df = test_data[np.where(test_data[:,0] == number),1:]
        elif set_type == b"train":
            df = train_data[np.where(train_data[:,0] == number),1:]
        if set_type == b"valid":
            df = valid_data[np.where(valid_data[:,0] == number),1:]
        return df.reshape(-1)
    
    def transform_sm_map(self, m):
        m = nilearn.image.new_img_like(MASK_IMG, m)
        if AVERAGE_SPACE:
            if AVERAGE_ON > 0:
                imgs = [nilearn.image.index_img(m, range(i,min(i+AVERAGE_ON, m.shape[-1]))) for i in range(0, m.shape[-1], AVERAGE_ON)]
                #for i in imgs:
                #    print(i.shape)
                #imgs = [nilearn.image.new_img_like(MASK_IMG, m[:,:,:,i:min(i+AVERAGE_ON, m.shape[-1])]) for i in range(0, m.shape[-1], AVERAGE_ON)]
                temp = [nilearn.image.mean_img(i) for i in imgs]
                m = nilearn.image.concat_imgs(temp)
            else:
                m = nilearn.image.mean_img(m)
        if APPLY_MASK:
            m = nilearn.masking.apply_mask(m, MASK_IMG)
        else:
            m = nilearn.image.get_data(m)
        if NORMALIZE_SM:
            m = dltk.io.preprocessing.normalise_one_one(m)
        #print(m.shape)
        #print(m[:3,:3])
        m = np.expand_dims(m, -1)
        #print(m.shape)
        return m.reshape(-1)
        
    def __call__(self, id_num, set_type):
        id_num = int(id_num)
        non_sm_features = self.collect_non_sm_features(id_num, set_type)
        if use_mri_images:
            sm_map = self.read_hdf5(id_num, set_type)
            if TRANSFORM:
                sm_map = self.transform_sm_map(sm_map)
            else:
                sm_map = sm_map.reshape(-1)
        if use_mri_images and use_fnc_loading:
            m = np.concatenate([sm_map, non_sm_features])
        elif use_mri_images:
            if set_type != b"test":
                m = np.concatenate([sm_map, non_sm_features[-5:]])
            else:
                m = sm_map
        elif use_fnc_loading:
            m = non_sm_features
        #print(non_sm_features.shape)
        #print(sm_map.shape)
        #print(m.shape)
        m = m.reshape((1,-1))
        return m


# In[ ]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
num_avgs = (53 // AVERAGE_ON) + 1 if AVERAGE_ON > 0 else 53
if TRANSFORM:
    if APPLY_MASK and AVERAGE_SPACE:
        sm_output_size = 58869
        sms_shape, non_sm_len = ((58869,), len(data_columns))
    elif APPLY_MASK:
        sm_output_size = 3120057
        sms_shape, non_sm_len = ((53, 58869), len(data_columns)) #time is the first dimension
    elif AVERAGE_SPACE:
        sm_output_size = 173628 * num_avgs
        sms_shape, non_sm_len = ((53, 63, 52, num_avgs), len(data_columns)) #there is no time; averaged over time
    else:
        sm_output_size = 9202284
        sms_shape, non_sm_len = ((53, 63, 52, 53), len(data_columns)) #time is the last dimension
else:
    sm_output_size = 9202284
    sms_shape, non_sm_len = ((53, 63, 52, 53), len(data_columns)) #time is the last dimension

if use_mri_images and use_fnc_loading:
    ds_output_size = sm_output_size + len(data_columns)
elif use_mri_images:
    ds_output_size = sm_output_size
elif use_fnc_loading:
    ds_output_size = len(data_columns)
else:
    raise Exception("At least one of 'use_mri_images' or 'use_fnc_loading' should be True.")
    
train_ds = tf.data.Dataset.from_tensor_slices(train_data[:,0]).shuffle(1024)
train_ds = train_ds.interleave(lambda filename: tf.data.Dataset.from_generator(
    hdf5_reader(), tf.float32, tf.TensorShape((ds_output_size + 5,)), args=(filename, "train")), cycle_length = AUTOTUNE)
train_ds = (train_ds.map(lambda x: (x[:-5], x[-5:]))
.cache()
.repeat()
.batch(BATCH_SIZE)
)


valid_ds = tf.data.Dataset.from_tensor_slices(valid_data[:,0])#.shuffle(1024)
valid_ds = valid_ds.interleave(lambda filename: tf.data.Dataset.from_generator(
    hdf5_reader(), tf.float32, tf.TensorShape((ds_output_size + 5,)), args=(filename, "valid")), cycle_length = AUTOTUNE)
valid_ds = (valid_ds.map(lambda x: (x[:-5], x[-5:]))
.cache()
.repeat()
.batch(VALID_BATCH_SIZE)
)


test_ds = tf.data.Dataset.from_tensor_slices(test_data[:,0])
test_ds = (test_ds.interleave(lambda filename: tf.data.Dataset.from_generator(
    hdf5_reader(), tf.float32, tf.TensorShape((ds_output_size,)), args=(filename, "test")), cycle_length = AUTOTUNE)
.cache()
.repeat()
.batch(BATCH_SIZE)
)


# In[ ]:


h = list(train_ds.take(1).as_numpy_iterator())
h2 = list(valid_ds.take(1).as_numpy_iterator())
h3 = list(test_ds.take(1).as_numpy_iterator())
print(h[0][0].shape, h[0][1].shape, sms_shape)
print(h3[0].shape)
del h, h2, h3


# In[ ]:


class LRSelector(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, factor = 2, varname = "lr", method = None):
        self.history = {}
        self.iteration = 0
        self.minlr = min_lr
        self.maxlr = max_lr
        self.factor = factor
        self.currentlr = self.minlr
        self.varname = varname
        self.method = method
        #self.num_iterations = math.log(self.maxlr / self.minlr, self.factor)
        #print(self.num_iterations)
        
    def on_train_begin(self, logs):
        if self.varname == "lr":
            K.set_value(self.model.optimizer.lr, self.minlr)
        elif self.varname == "momentum":
            K.set_value(self.model.optimizer.momentum, self.minlr)
        #self.num_iterations = math.log(self.maxlr / self.minlr, self.factor)
        
    def compute_lr(self):
        if self.method == "sum":
            lr = self.minlr + self.iteration * self.factor
        else:
            lr = self.minlr * np.power(self.factor, self.iteration)
        return lr
        
    def on_batch_end(self, batch, logs = {}):
        self.iteration += 1
        self.history.setdefault('loss', []).append(logs['loss'])
        self.history.setdefault('lr', []).append(self.currentlr)
        self.currentlr = self.compute_lr()
        if self.varname == "lr":
            K.set_value(self.model.optimizer.lr, self.currentlr)
        elif self.varname == "momentum":
            K.set_value(self.model.optimizer.momentum, self.currentlr)
        #print("in cycler")
        #print(logs.keys())
        #print(logs['loss'])
        #print(batch)
        
class LRScheduler2(tf.keras.callbacks.Callback):
    def __init__(self, start_lr, max_lr, epochs, min_lr = 1e-3, stable_percent = 0.2, stable_factor = 0.2, constant = 0):
        self.stable_steps_start = epochs - np.ceil(epochs * stable_percent)
        self.constant_start = self.stable_steps_start - constant
        self.mid_step = self.constant_start
        self.slope = (max_lr - start_lr) / self.mid_step
        self.startlr = start_lr
        self.maxlr = max_lr
        self.stable_factor = stable_factor
        self.minlr = min_lr
    
    def on_train_begin(self, logs):
        if hasattr(self, "model"):
            K.set_value(self.model.optimizer.lr, self.startlr)
    
    def on_epoch_end(self, epoch, logs = {}):
        if epoch <= self.mid_step:
            lr = self.startlr + (epoch * self.slope)
        elif epoch < self.stable_steps_start:
            lr = self.maxlr
        else:
            lr = min((self.maxlr - self.minlr) * self.stable_factor**(epoch - self.stable_steps_start) + self.minlr, self.maxlr)
        logs.setdefault('lr', []).append(lr)
        #return(lr)
        if hasattr(self, "model"):
            K.set_value(self.model.optimizer.lr, lr)
            print(self.model.optimizer.lr.numpy())
        else:
            return lr


# In[ ]:


def build_model(model_name, use_fnc_loading, use_mri_images):
    if model_name == "1d_model":
        model = build_1d_model(use_fnc_loading, use_mri_images)
    elif model_name == "lstm_on_masked":
        model = build_lstm_on_masked_seq(use_fnc_loading, use_mri_images)
    elif model_name == "3d_on_avg_space":
        #model = build_3d_mode(sms_shape, non_sm_feature_count)
        model = build_3d_model_avg_space(use_fnc_loading, use_mri_images)
    elif model_name == "3d_on_mri_sequences":
        model = build_3d_model_mri_seq(use_fnc_loading, use_mri_images)
    return model


# In[ ]:


def get_non_sm_structure(non_sm_features_layer):
    x = tf.keras.layers.Dense(1024)(non_sm_features_layer)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(768)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(768)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    return x


# In[ ]:


def build_1d_model(use_fnc_loading, use_mri_images):
    features = ds_output_size
    sms_feature_len = np.prod(sms_shape)
    print(features)
    
    input_layer = tf.keras.layers.Input(shape=(features,))
    
    '''
    if(use_fnc_loading and use_mri_images):
        input_layer2 = input_layer
    elif use_fnc_loading:
        #input_layer = tf.keras.layers.Lambda(lambda x: x[:,-non_sm_len:])(input_layer)
        input_layer2 = input_layer
    elif use_mri_images:
        #input_layer = tf.keras.layers.Lambda(lambda x: x[:,:-non_sm_len])(input_layer)
        input_layer2 = input_layer
    else:
        raise Exception("At least one of use_fnc_loading or use_mri_images should be True")
    '''
    x = get_non_sm_structure(input_layer)
    x = tf.keras.layers.Dense(5)(x)
    output_layer = tf.keras.layers.Activation('relu')(x)

    model = tf.keras.Model(input_layer, output_layer)
    
    optimizer = tf.optimizers.Adam()
    
    model.compile(optimizer, loss='mse', metrics=['mae', 'mape'])
    return model


# In[ ]:


def build_lstm_on_masked_seq(use_fnc_loading, use_mri_images):
    number_of_series = 53
    series_len = sms_shape[1]
    
    features = ds_output_size
    sms_feature_len = np.prod(sms_shape)
    
    input_layer = tf.keras.layers.Input(shape=(features,))
    
    if use_fnc_loading and use_mri_images:
        sm_features_layer = tf.keras.layers.Lambda(lambda x: x[:,:-non_sm_len])(input_layer)
        non_sm_features_layer = tf.keras.layers.Lambda(lambda x: x[:,-non_sm_len:])(input_layer)
    else:
        sm_features_layer = input_layer
        non_sm_features_layer = input_layer
    
    sm_features_layer = tf.keras.layers.Reshape(sms_shape)(sm_features_layer)
    sm_features_layer = tf.keras.layers.GRU(16, activation = "tanh", return_sequences=True)(sm_features_layer)
    sm_features_layer = tf.keras.layers.GRU(32, activation = "tanh", return_sequences=True)(sm_features_layer)
    sm_features_layer = tf.keras.layers.GRU(32, activation = "tanh", return_sequences=False)(sm_features_layer)
    sm_features_layer = tf.keras.layers.Dense(512, activation = "relu")(sm_features_layer)
    print("sm_layer created")
    
    if use_fnc_loading:
        non_sm_features_layer = get_non_sm_structure(non_sm_features_layer)
    print("non_sm created")
    
    if use_fnc_loading and use_mri_images:
        print("concating both")
        squeeze_layer = tf.keras.layers.concatenate([sm_features_layer, non_sm_features_layer])
    elif use_fnc_loading:
        print("onluy fnc loading")
        squeeze_layer = non_sm_features_layer
    elif use_mri_images:
        print("only images")
        squeeze_layer = sm_features_layer
    else:
        raise Exception("At least one of use_fnc_loading or use_mri_images should be True")
    
    x = tf.keras.layers.Dense(1024, activation = "relu")(squeeze_layer)
    x = tf.keras.layers.Dense(512, activation = "relu")(x)
    output_layer = tf.keras.layers.Dense(5, activation = "relu")(x)
    
    model = tf.keras.Model(input_layer, output_layer)
    
    optimizer = tf.optimizers.Adam()
    
    model.compile(optimizer, loss='mse', metrics=['mae', 'mape'])
    return model


# In[ ]:


def build_3d_model_avg_space(use_fnc_loading, use_mri_images):
    features = ds_output_size
    sms_feature_len = np.prod(sms_shape)
    
    input_layer = tf.keras.layers.Input(shape=(features,))
    if use_fnc_loading and use_mri_images:
        sm_features_layer = tf.keras.layers.Lambda(lambda x: x[:,:-non_sm_len])(input_layer)
        non_sm_features_layer = tf.keras.layers.Lambda(lambda x: x[:,-non_sm_len:])(input_layer)
    else:
        sm_features_layer = input_layer
        non_sm_features_layer = input_layer
    

    sm_features_layer = tf.keras.layers.Reshape(sms_shape)(sm_features_layer)
    sm_features_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(sm_features_layer)
    sm_features_layer = tf.keras.layers.Conv3D(32, (3,3,3), activation = "relu")(sm_features_layer)
    sm_features_layer = tf.keras.layers.AveragePooling3D((2,2,2))(sm_features_layer)
    sm_features_layer = tf.keras.layers.Conv3D(32, (3,3,3), activation = "relu")(sm_features_layer)
    sm_features_layer = tf.keras.layers.AveragePooling3D((2,2,2))(sm_features_layer)
    sm_features_layer = tf.keras.layers.Conv3D(16, (3,3,3), activation = "relu")(sm_features_layer)
    #sm_features_layer = tf.keras.layers.AveragePooling3D((2,2,2))(sm_features_layer)
    sm_features_layer = tf.keras.layers.Flatten()(sm_features_layer)
    sm_features_layer = tf.keras.layers.Dense(1024, activation = "relu")(sm_features_layer)
    #sm_features_layer = tf.keras.layers.Dense(128, activation = "relu")(sm_features_layer)  
    
    if use_fnc_loading:
        non_sm_features_layer = get_non_sm_structure(non_sm_features_layer)
    
    if use_fnc_loading and use_mri_images:
        squeeze_layer = tf.keras.layers.concatenate([sm_features_layer, non_sm_features_layer])
    elif use_fnc_loading:
        squeeze_layer = non_sm_features_layer
    elif use_mri_images:
        squeeze_layer = sm_features_layer
    else:
        raise Exception("At least one of use_fnc_loading or use_mri_images should be True")
        
    x = tf.keras.layers.Dense(512, activation = "relu")(squeeze_layer)
    x = tf.keras.layers.Dense(256, activation = "relu")(x)
    output_layer = tf.keras.layers.Dense(5, activation = 'linear')(x)
    model = tf.keras.Model(input_layer, output_layer)
    
    optimizer = tf.optimizers.Adam()
    
    model.compile(optimizer, loss='mse', metrics=['mae', 'mape'])
    return model


# In[ ]:


def build_3d_model_mri_seq(use_fnc_loading, use_mri_images):
    number_of_pictures = sms_shape[-1]
    single_pic_shape = sms_shape[:-1]
    single_pic_features = np.prod(single_pic_shape)
    print(single_pic_shape)
    features = ds_output_size
    sms_feature_len = np.prod(sms_shape)
    
    input_layer = tf.keras.layers.Input(shape=(features,))
    if use_fnc_loading and use_mri_images:
        sm_features_layer = tf.keras.layers.Lambda(lambda x: x[:,:-non_sm_len])(input_layer)
        non_sm_features_layer = tf.keras.layers.Lambda(lambda x: x[:,-non_sm_len:])(input_layer)
    else:
        sm_features_layer = input_layer
        non_sm_features_layer = input_layer
    
    sm_outputs = []
    for i in range(number_of_pictures):
        single_image_layer = tf.keras.layers.Lambda(lambda x: x[:,(i*single_pic_features):((i+1)*single_pic_features)])(sm_features_layer)
        single_image_layer = tf.keras.layers.Reshape(single_pic_shape)(single_image_layer)
        single_image_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(single_image_layer)
        single_image_layer = tf.keras.layers.Conv3D(4, (7,7,7), activation = "relu", data_format="channels_last")(single_image_layer)
        single_image_layer = tf.keras.layers.AveragePooling3D((3,3,3))(single_image_layer)
        single_image_layer = tf.keras.layers.Conv3D(4, (3,3,3), activation = "relu", data_format="channels_last")(single_image_layer)
        single_image_layer = tf.keras.layers.Flatten()(single_image_layer)
        single_image_layer = tf.keras.layers.Dense(5, activation = "relu")(single_image_layer)
        sm_outputs.append(single_image_layer)
    tf.print(len(sm_outputs))
    sm_features_layer = tf.keras.layers.concatenate(sm_outputs)
    
    if use_fnc_loading:
        non_sm_features_layer = get_non_sm_structure(non_sm_features_layer)
    
    if use_fnc_loading and use_mri_images:
        squeeze_layer = tf.keras.layers.concatenate([sm_features_layer, non_sm_features_layer])
    elif use_fnc_loading:
        squeeze_layer = non_sm_features_layer
    elif use_mri_images:
        squeeze_layer = sm_features_layer
    else:
        raise Exception("At least one of use_fnc_loading or use_mri_images should be True")
    
    x = tf.keras.layers.Dense(32, activation = "relu")(squeeze_layer)
    x = tf.keras.layers.Dense(32, activation = "relu")(x)
    x = tf.keras.layers.Dense(16, activation = "relu")(x)
    output_layer = tf.keras.layers.Dense(5, activation = "relu")(x)
    
    model = tf.keras.Model(input_layer, output_layer)
    
    optimizer = tf.optimizers.Adam()
    
    model.compile(optimizer, loss='mse', metrics=['mae', 'mape'])
    return model


# In[ ]:


#model_name = "3d_on_avg_space"
model_name = "3d_on_mri_sequences"
#model_name = "lstm_on_masked"
#model_name = "1d_model"

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    with tpu_strategy.scope():
        model = build_model(model_name, use_fnc_loading, use_mri_images)
    print("TPU successfully set up!")
except:
    print("running without TPU")
    model = build_model(model_name, use_fnc_loading, use_mri_images)


# In[ ]:


#model.summary()
tf.keras.utils.plot_model(model)


# In[ ]:


epochs = 20
STEP_SIZE_TRAIN = len(train_data[0]) // BATCH_SIZE
STEP_SIZE_VALID = len(valid_data[0]) // VALID_BATCH_SIZE
lr_selector = LRSelector(min_lr = 1e-3, max_lr = 10, factor = 1.05)
a = LRScheduler2(start_lr = 0.003, max_lr = 0.003, epochs = epochs, min_lr = 0.003, constant = 2, stable_percent = 0.5, stable_factor = 0.8)


ls = [a.on_epoch_end(i) for i in range(epochs)]
#plt.plot(ls)
#plt.show()
scheduler = copy.copy(a)

'''
model.fit(train_ds, 
            #class_weight = (0.3, 0.175, 0.175, 0.175, 175), 
            steps_per_epoch = STEP_SIZE_TRAIN,
            #callbacks = [lr_selector],
            callbacks = [scheduler],
            epochs = epochs,
            validation_data = valid_ds,
            validation_steps = STEP_SIZE_VALID
            )
'''


# In[ ]:


#limit = 100
#print(len(lr_selector.history['lr']))
#plt.plot(lr_selector.history['lr'][:limit], (lr_selector.history['loss'][:limit]))

